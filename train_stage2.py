"""
SPIGA Training Script - Stage 2
Shape Preserving Facial Landmarks with Graph Attention Networks

Paper: "Shape Preserving Facial Landmarks with Graph Attention Networks"
       Prados-Torreblanca et al. (BMVC 2022)
       https://arxiv.org/abs/2210.07233

Stage 2: Fine-tuning with both landmark detection and pose estimation (150 epochs)

Multi-GPU Training:
    - Uses torch.nn.parallel.DistributedDataParallel (DDP)
    - Compatible with SLURM sbatch and srun
    - Automatic rank/world_size detection from environment
    - Gradient synchronization across GPUs
    - Proper handling of batch size scaling

Usage with SLURM:
    srun python train_stage2.py --dataset wflw --pretrained ./checkpoints/checkpoint_stage1/best_model.pth
    
    Or submit with sbatch:
    sbatch train_stage2.sh

Usage with torchrun:
    torchrun --nproc_per_node=4 train_stage2.py --dataset wflw

Usage with single GPU:
    python train_stage2.py --dataset wflw

Environment Variables (set by SLURM or torchrun):
    RANK: Global rank of current process (0 to world_size-1)
    WORLD_SIZE: Total number of processes
    MASTER_ADDR: Master node address (default: 127.0.0.1)
    MASTER_PORT: Master node port (default: 29500)
    LOCAL_RANK: Local rank on node
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb

MANUAL_SEED = 42

from utils import setup_memory_optimizations, setup_wandb, setup, cleanup, is_main_process, setup_logging
from loss import SmoothL1Loss, LandmarkLoss, CombinedLoss

from spiga.models.spiga import SPIGA
from spiga.data.loaders.dl_config import AlignConfig
from spiga.data.loaders.dataloader import get_dataloader



# ======================== Training Functions ========================
def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                logger: logging.Logger,
                rank: int,
                world_size: int,
                scaler: torch.cuda.amp.GradScaler = None,
                use_amp: bool = True,
                num_landmarks: int = 98,
                wandb_log=None) -> Tuple[float, Dict]:
    """
    Train for one epoch with distributed support
    
    Args:
        model: Neural network model (should be wrapped with DistributedDataParallel)
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        avg_loss: Average loss over the epoch
        loss_details_accum: Accumulated loss details
    """
    model.train()
    total_loss = 0.0
    loss_details_accum = {}
    num_batches = 0

    # Set sampler epoch for proper shuffling in distributed training
    if hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    # track epochs is progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=True) if is_main_process() else dataloader
    
    # main loop
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move data to device
            images = batch["image"].to(device, dtype=torch.float32)
            
            # Ensure images have correct shape: [B, C, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # Prepare targets (ensure correct dtype)
            targets = {"landmarks": batch["landmarks"].to(device, dtype=torch.float32)}
            
            # Add optional targets
            if "heatmaps_points" in batch:
                targets["heatmaps_points"] = batch["heatmaps_points"].to(device, dtype=torch.float32)
            if "heatmaps_edges" in batch:
                targets["heatmaps_edges"] = batch["heatmaps_edges"].to(device, dtype=torch.float32)
            if "pose" in batch:
                targets["pose"] = batch["pose"].to(device, dtype=torch.float32)
                if batch_idx == 0 and is_main_process():
                    logger.info(f"[DEBUG] Pose found in batch, shape: {targets['pose'].shape}")
            elif batch_idx == 0 and is_main_process():
                logger.warning(f"[DEBUG] No pose in batch! Available keys: {batch.keys()}")

            # Delete batch from memory after moving to device
            torch.cuda.empty_cache()

            # Prepare model inputs
            model3d = None
            cam_matrix = None
            if "model3d" in batch:
                model3d = batch["model3d"].to(device, dtype=torch.float32)
                # Ensure model3d has shape [B, L, 3]
                # Handle all cases: 2D [L, 3] -> add batch, 3D [B, L, 3] is correct
                while model3d.dim() < 3:
                    model3d = model3d.unsqueeze(0)
            if "cam_matrix" in batch:
                cam_matrix = batch["cam_matrix"].to(device, dtype=torch.float32)
                # Ensure cam_matrix has shape [B, 3, 3]
                # Handle all cases: 2D [3, 3] -> add batch, 3D [B, 3, 3] is correct
                while cam_matrix.dim() < 3:
                    cam_matrix = cam_matrix.unsqueeze(0)

            
            optimizer.zero_grad()

            # shape images correctly
            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)

            # Forward pass with AMP - Stage 2: CNN + Pose (with regression heads, like Stage 1)
            # Call visual_cnn directly + pose_fc, use regression heads for landmarks
            actual_model = model.module if hasattr(model, 'module') else model
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Direct CNN forward - returns {'VisualField': [...], 'HGcore': [...]}
                cnn_predictions = actual_model.visual_cnn(images)
                
                # Add pose prediction (extract from last HGcore, same as backbone_forward)
                pose_raw = cnn_predictions['HGcore'][-1]  # Last hourglass core features
                B, L, _, _ = pose_raw.shape
                pose_flat = pose_raw.reshape(B, L)  # Flatten spatial dimensions
                pose = actual_model.pose_fc(pose_flat)
                cnn_predictions['Pose'] = pose
                
                # Compute loss: landmarks (regression heads) + pose
                loss, loss_details = criterion(cnn_predictions, targets)
                
                # Debug logging for first batch
                if batch_idx == 0 and is_main_process():
                    logger.info(f"[DEBUG] Pose in predictions: {'Pose' in cnn_predictions}")
                    logger.info(f"[DEBUG] Pose in targets: {'pose' in targets}")
                    if 'Pose' in cnn_predictions:
                        logger.info(f"[DEBUG] Pose pred shape: {cnn_predictions['Pose'].shape}")
                    if 'pose' in targets:
                        logger.info(f"[DEBUG] Pose target shape: {targets['pose'].shape}")
                    logger.info(f"[DEBUG] Loss details: {loss_details}")

            # Check for NaN/Inf before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process():
                    logger.error(f"NaN/Inf detected in training loss at epoch {epoch}, batch {batch_idx}")
                    logger.error(f"Loss value: {loss.item()}")
                    logger.error(f"Skipping batch...")
                continue

            # Save loss values before deletion
            loss_val = loss.item()
            loss_details_copy = {k: v for k, v in loss_details.items()}

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for exploding gradients
            if grad_norm > 10.0 and is_main_process():
                logger.warning(f"Large gradient norm: {grad_norm:.2f} at epoch {epoch}, batch {batch_idx}")
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Aggressive memory cleanup
            torch.cuda.empty_cache()
            
            # Accumulate losses
            total_loss += loss_val
            num_batches += 1
            for key, val in loss_details_copy.items():
                if key not in loss_details_accum:
                    loss_details_accum[key] = 0.0
                loss_details_accum[key] += val

            # Update progress bar and wandb (main process only)
            if is_main_process():
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': f"{loss_val:.6f}",
                        'avg': f"{total_loss / num_batches:.6f}",
                    })
                
                # Log to wandb (main process only)
                if is_main_process() and wandb_log is not None and batch_idx % 10 == 0:
                    log_dict = {
                        'train/loss': loss_val,
                        'train/avg_loss': total_loss / num_batches,
                        'train/batch': batch_idx,
                        'train/epoch': epoch,
                    }
                    for key, val in loss_details_copy.items():
                        log_dict[f'train/{key}'] = val
                    wandb_log.log(log_dict)

        except Exception as e:
            if is_main_process():
                logger.error(f"Error on line {e.__traceback__.tb_lineno} in batch {batch_idx}: {e}")
                # exit program
                sys.exit(1)


    # Synchronize metrics across processes
    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor)
        total_loss = total_loss_tensor.item() / world_size
        
        num_batches_tensor = torch.tensor(num_batches, device=device)
        dist.all_reduce(num_batches_tensor)
        num_batches = int(num_batches_tensor.item() / world_size)

    # Average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for key in loss_details_accum:
        loss_details_accum[key] /= num_batches if num_batches > 0 else 1

    return avg_loss, loss_details_accum


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             logger: logging.Logger,
             rank: int,
             world_size: int,
             epoch: int = 0,
             use_amp: bool = True,
             num_landmarks: int = 98,
             wandb_log=None) -> Tuple[float, Dict]:
    """
    Validate the model with distributed support
    """
    model.eval()
    total_loss = 0.0
    loss_details_accum = {}
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=True) if is_main_process() else dataloader
        
        for batch in pbar:
            try:
                # Move data to device
                images = batch["image"].to(device, dtype=torch.float32)
                
                # Ensure images have correct shape: [B, C, H, W]
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                
                # Prepare targets (ensure correct dtype)
                targets = {"landmarks": batch["landmarks"].to(device, dtype=torch.float32)}
                
                if "heatmaps_points" in batch:
                    targets["heatmaps_points"] = batch["heatmaps_points"].to(device, dtype=torch.float32)
                if "heatmaps_edges" in batch:
                    targets["heatmaps_edges"] = batch["heatmaps_edges"].to(device, dtype=torch.float32)
                if "headpose" in batch:
                    targets["pose"] = batch["headpose"].to(device, dtype=torch.float32)

                # Prepare model inputs
                model3d = None
                cam_matrix = None
                if "model3d" in batch:
                    model3d = batch["model3d"].to(device, dtype=torch.float32)
                    # Ensure model3d has shape [B, L, 3]
                    # Handle all cases: 2D [L, 3] -> add batch, 3D [B, L, 3] is correct
                    while model3d.dim() < 3:
                        model3d = model3d.unsqueeze(0)
                if "cam_matrix" in batch:
                    cam_matrix = batch["cam_matrix"].to(device, dtype=torch.float32)
                    # Ensure cam_matrix has shape [B, 3, 3]
                    # Handle all cases: 2D [3, 3] -> add batch, 3D [B, 3, 3] is correct
                    while cam_matrix.dim() < 3:
                        cam_matrix = cam_matrix.unsqueeze(0)

                # shape images correctly
                if images.shape[1] != 3:
                    images = images.permute(0, 3, 1, 2)

                # Forward pass with AMP - Stage 2: CNN + Pose (with regression heads)
                actual_model = model.module if hasattr(model, 'module') else model
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Direct CNN forward
                    cnn_predictions = actual_model.visual_cnn(images)
                    
                    # Add pose prediction (extract from last HGcore)
                    if model3d is not None and cam_matrix is not None:
                        pose_raw = cnn_predictions['HGcore'][-1]
                        B, L, _, _ = pose_raw.shape
                        pose_flat = pose_raw.reshape(B, L)
                        pose = actual_model.pose_fc(pose_flat)
                        cnn_predictions['Pose'] = pose
                    
                    loss, loss_details = criterion(cnn_predictions, targets)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process():
                        logger.error(f"NaN/Inf in validation loss at batch {num_batches}")
                    continue

                # Accumulate losses
                total_loss += loss.item()
                num_batches += 1
                for key, val in loss_details.items():
                    if key not in loss_details_accum:
                        loss_details_accum[key] = 0.0
                    loss_details_accum[key] += val

                if is_main_process() and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'loss': f"{loss.item():.6f}"})

            except Exception as e:
                if is_main_process():
                    logger.error(f"Validation error on line {e.__traceback__.tb_lineno}: {e}")
                    # exit program
                    sys.exit(1)

    # Synchronize metrics across processes
    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor)
        total_loss = total_loss_tensor.item() / world_size
        
        num_batches_tensor = torch.tensor(num_batches, device=device)
        dist.all_reduce(num_batches_tensor)
        num_batches = int(num_batches_tensor.item() / world_size)

    # Average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for key in loss_details_accum:
        loss_details_accum[key] /= num_batches if num_batches > 0 else 1

    # Log to wandb (main process only)
    if is_main_process() and wandb_log is not None:
        log_dict = {
            'val/loss': avg_loss,
            'val/epoch': epoch,
        }
        for key, val in loss_details_accum.items():
            log_dict[f'val/{key}'] = val
        wandb_log.log(log_dict)

    return avg_loss, loss_details_accum


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   loss: float,
                   checkpoint_dir: str,
                   filename: str,
                   logger: logging.Logger,
                   rank: int):
    """Save model checkpoint (main process only)"""
    if rank != 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Handle DistributedDataParallel model
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str,
                   model: nn.Module,
                   optimizer: Optional[optim.Optimizer] = None,
                   device: torch.device = None,
                   logger: Optional[logging.Logger] = None) -> int:
    """Load model checkpoint with proper DDP handling"""
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    
    if logger:
        logger.debug(f"Checkpoint keys sample: {list(model_state_dict.keys())[:3]}")
    
    # Handle state dict key mismatch between DDP and non-DDP models
    is_ddp_model = hasattr(model, 'module')
    state_dict_has_module = any(k.startswith('module.') for k in model_state_dict.keys())
    
    if logger:
        logger.info(f"Model is DDP: {is_ddp_model}, State dict has 'module.': {state_dict_has_module}")
    
    if is_ddp_model and not state_dict_has_module:
        # Model is DDP but checkpoint doesn't have 'module.' prefix - add it
        if logger:
            logger.info("Converting state dict keys: adding 'module.' prefix for DDP")
        model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
    elif not is_ddp_model and state_dict_has_module:
        # Model is not DDP but checkpoint has 'module.' prefix - remove it
        if logger:
            logger.info("Converting state dict keys: removing 'module.' prefix for non-DDP")
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as e:
        if logger:
            logger.error(f"Error loading state dict: {e}")
            # Try with strict=False to see if it's just extra keys
            logger.warning("Attempting to load with strict=False...")
        try:
            model.load_state_dict(model_state_dict, strict=False)
            if logger:
                logger.warning("Successfully loaded checkpoint with strict=False (some keys may be missing)")
        except Exception as e2:
            if logger:
                logger.error(f"Failed even with strict=False: {e2}")
            raise
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            if logger:
                logger.warning(f"Could not load optimizer state: {e}")
    
    epoch = checkpoint.get("epoch", 0)
    if logger:
        logger.info(f"Checkpoint loaded from: {checkpoint_path} (epoch {epoch})")
    
    return epoch


def train_stage2(args, logger: logging.Logger, device: torch.device, rank: int, world_size: int, pretrained_path: str, wandb_log=None):
    """Stage 2: Fine-tuning CNN backbone + Pose (Freeze GAT regressors)
    
    Paper: "Once the model has been pre-trained with landmarks, it is fine-tuned 
    with both tasks, pose and landmarks. Sharing the same hyperparameter configuration 
    as in the previous pretraining stage during 150 epochs, with a step decay from 
    10^-3 to 10^-4 at epoch 100."
    
    Training: Same as Stage 1 (CNN + regression heads) + Pose estimation
    - visual_cnn (CNN backbone)
    - pose_fc (pose estimation head)
    - Use regression heads for landmark prediction (no GAT)
    
    Frozen: GAT regressors (gcn, shape_encoder, conv_window) - not used in Stage 2
    """
    if is_main_process():
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: CNN + Pose Fine-tuning (Regression Heads for Landmarks)")
        logger.info("="*80)
        logger.info(f"Epochs: {args.epochs_stage2}")
        logger.info(f"Learning Rate: {args.lr_stage2} (decay to {args.lr_stage2 * 0.1} at epoch {args.decay_epoch_stage2})")
        logger.info(f"Pretrained CNN weights: {pretrained_path}")

    # Data loading
    logger.info(f"Loading {args.dataset} dataset...")
    train_cfg = AlignConfig(args.dataset, mode='train')
    val_cfg = AlignConfig(args.dataset, mode='test')
    
    # Create sampler config object for distributed training
    class SamplerConfig:
        def __init__(self, world_size, rank):
            self.world_size = world_size
            self.rank = rank
    
    sampler_cfg = SamplerConfig(world_size, rank) if world_size > 1 else None
    train_loader, _ = get_dataloader(args.batch_size, train_cfg, sampler_cfg=sampler_cfg)
    val_loader, _ = get_dataloader(args.batch_size, val_cfg)

    if is_main_process():
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Create SPIGA model
    model = SPIGA(
        num_landmarks=args.num_landmarks,
        num_edges=args.num_edges
    ).to(device)
    
    # Load Stage 1 pretrained CNN weights (best_model.pth from checkpoint_stage1)
    if os.path.exists(pretrained_path):
        if is_main_process():
            logger.info(f"Loading Stage 1 pretrained CNN weights from: {pretrained_path}")
        load_checkpoint(pretrained_path, model, device=device, logger=logger)
    else:
        if is_main_process():
            logger.warning(f"Pretrained Stage 1 weights not found: {pretrained_path}")
    
    # STAGE 2: Freeze GAT components, train CNN + Pose
    # Freeze: gcn, shape_encoder, conv_window (GAT regressors for Stage 3)
    # Train: visual_cnn (CNN backbone) + pose_fc (pose estimation)
    for param in model.gcn.parameters():
        param.requires_grad = False
    for param in model.shape_encoder.parameters():
        param.requires_grad = False
    for param in model.conv_window.parameters():
        param.requires_grad = False
    
    # Ensure CNN and pose_fc are trainable
    for param in model.visual_cnn.parameters():
        param.requires_grad = True
    for param in model.pose_fc.parameters():
        param.requires_grad = True
    
    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ SPIGA model created with {args.num_landmarks} landmarks")
        logger.info(f"✓ GAT components FROZEN (not used in Stage 2)")
        logger.info(f"✓ Training CNN + Pose (visual_cnn, pose_fc) with regression heads")
        logger.info(f"✓ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Wrap with DistributedDataParallel
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # GAT params are frozen
        )
        if is_main_process():
            logger.info(f"✓ Model wrapped with DistributedDataParallel")

    # Loss: Combined landmark (with regression heads) + pose loss (Paper Equation 2)
    # Same as Stage 1 but with added pose loss
    from loss import CombinedLoss
    criterion = CombinedLoss(
        num_stages=args.num_stages,
        lambda_coord=args.lambda_coord,
        lambda_att=args.lambda_att,
        lambda_p=args.lambda_p  # Paper: λ_p = 1.0
    ).to(device)
    
    # Optimizer: CNN + pose_fc + regression heads
    actual_model = model.module if hasattr(model, 'module') else model
    trainable_params_list = (
        list(actual_model.visual_cnn.parameters()) + 
        list(actual_model.pose_fc.parameters()) +
        list(criterion.landmark_loss.stage1_heads.parameters())  # Regression heads from loss
    )
    optimizer = optim.Adam(trainable_params_list, lr=args.lr_stage2)
    scheduler = StepLR(optimizer, step_size=args.decay_epoch_stage2, gamma=0.1)
    
    # AMP: Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    if is_main_process():
        logger.info(f"✓ Optimizer: Adam with lr={args.lr_stage2} (CNN + Pose + Regression Heads)")
        logger.info(f"✓ Scheduler: StepLR (decay 0.1 at epoch {args.decay_epoch_stage2})")
        logger.info(f"✓ Loss: Landmark (regression heads) + Pose, λ_p={args.lambda_p}")
        logger.info(f"✓ AMP: {'Enabled' if args.use_amp else 'Disabled'}")

    best_loss = float('inf')
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoint_stage2')

    # Training loop
    for epoch in range(args.epochs_stage2):
        if is_main_process():
            logger.info(f"\nEpoch {epoch+1}/{args.epochs_stage2}")
        
        # Train
        train_loss, train_details = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch+1, logger, rank, world_size, scaler=scaler,
            use_amp=args.use_amp, num_landmarks=args.num_landmarks,
            wandb_log=wandb_log
        )
        
        if is_main_process():
            logger.info(f"Train Loss: {train_loss:.6f}")
            if 'loss_pose' in train_details:
                logger.info(f"  - Landmark Loss: {train_details.get('loss_landmark', 0):.6f}")
                logger.info(f"  - Pose Loss: {train_details.get('loss_pose', 0):.6f}")

        # Validate
        val_loss, val_details = validate(
            model, val_loader, criterion, device, logger, rank, world_size,
            epoch=epoch+1, use_amp=args.use_amp, num_landmarks=args.num_landmarks,
            wandb_log=wandb_log
        )
        
        if is_main_process():
            logger.info(f"Val Loss: {val_loss:.6f}")

        # Scheduler step
        scheduler.step()

        # Save best checkpoint
        if is_main_process():
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch+1, val_loss, checkpoint_dir,
                              'best_model.pth', logger, rank)

            # Save periodic checkpoints
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(model, optimizer, epoch+1, val_loss, checkpoint_dir,
                              f'checkpoint_epoch_{epoch+1}.pth', logger, rank)

    if is_main_process():
        logger.info(f"Stage 2 completed! Best validation loss: {best_loss:.6f}")
    
    return os.path.join(checkpoint_dir, 'best_model.pth')


def main(rank, args):
    """Main training function for SLURM-based distributed multi-GPU training.
    
    Rank and world_size are passed as parameters from torchrun launcher.
    For single GPU testing, rank defaults to 0 and world_size to 1.
    """

    torch.manual_seed(MANUAL_SEED)
    
    # Note: rank and world_size are passed as parameters, not extracted here
    # This allows torchrun to properly manage the distributed environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = rank

    # Setup memory optimizations (must be before GPU initialization)
    setup_memory_optimizations()

    # Setup distributed training if multi-GPU
    if world_size > 1:
        setup(rank, world_size)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")
    
    # Setup wandb (main process only)
    wandb_log = setup_wandb(args, rank, "stage2") if rank == 0 else None

    # Setup logging
    logger = setup_logging(args.log_dir, 'stage2', rank)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoint_stage2')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration (main process only)
    if rank == 0:
        config_path = os.path.join(checkpoint_dir, "config_stage2.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Configuration saved to: {config_path}")

        # Print configuration
        logger.info("\n" + "="*80)
        logger.info("SPIGA Training Stage 2 Configuration (Distributed Multi-GPU)")
        logger.info("="*80)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Number of landmarks: {args.num_landmarks}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Total batch size: {args.batch_size * world_size}")
        logger.info(f"World size (GPUs): {world_size}")
        logger.info("="*80)
    

    # Determine pretrained model path
    if args.resume:
        pretrained_path = args.resume
    else:
        pretrained_path = os.path.join(args.checkpoint_dir, 'checkpoint_stage1', 'best_model.pth')
    
    # Run training
    train_stage2(args, logger, device, rank, world_size, pretrained_path, wandb_log=wandb_log)

    if is_main_process():
        logger.info("\n" + "="*80)
        logger.info("Stage 2 training complete!")
        logger.info("="*80)

    # Cleanup wandb
    if wandb_log is not None and is_main_process():
        wandb_log.finish()
    
    # Cleanup distributed training
    if world_size > 1:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPIGA Training Script - Stage 2: Fine-tuning with Landmarks + Pose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="wflw",
        choices=["300wpublic", "300wprivate", "wflw", "cofw68", "merlrav"],
        help="Dataset name",
    )
    parser.add_argument(
        "--num_landmarks",
        type=int,
        default=98,
        help="Number of facial landmarks",
    )
    parser.add_argument(
        "--num_edges",
        type=int,
        default=15,
        help="Number of edge types for graph structure",
    )

    # Stage 2 arguments
    parser.add_argument("--epochs_stage2", type=int, default=150,
                       help="Number of epochs for Stage 2")
    parser.add_argument("--lr_stage2", type=float, default=1e-3,
                       help="Learning rate for Stage 2")
    parser.add_argument("--decay_epoch_stage2", type=int, default=100,
                       help="Epoch to decay learning rate in Stage 2")

    # Model arguments
    parser.add_argument("--num_stages", type=int, default=4,
                       help="Number of Hourglass stages")

    # Loss weights
    parser.add_argument("--lambda_coord", type=float, default=4.0,
                       help="Weight for coordinate loss")
    parser.add_argument("--lambda_att", type=float, default=50.0,
                       help="Weight for attention (heatmap) loss")
    parser.add_argument("--lambda_p", type=float, default=1.0,
                       help="Weight for pose loss")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=24,
                       help="Batch size (paper uses 24 total)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use Automatic Mixed Precision (AMP) training")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                       help="Disable AMP training")
    
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to train on")
    parser.add_argument("--empty_cache", action="store_true", default=True,
                       help="Empty CUDA cache after each batch to reduce fragmentation")

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=50,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from or path to Stage 1 checkpoint")

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory to save logs")

    args = parser.parse_args()
    
    # For SLURM: rank and world_size come from environment variables set by torchrun
    # For single GPU: they default to 0 and 1
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"THIS PROCESS IS RANK {rank} BEFORE MAIN")

    if rank == 0:
        print(f"✓ SLURM Distributed Training Configuration:")
        print(f"  - Rank: {rank}")
        print(f"  - World Size: {world_size}")
        print(f"  - GPUs Available: {torch.cuda.device_count()}")
        print(f"  - Master Addr: {os.environ.get('MASTER_ADDR', '127.0.0.1')}")
        print(f"  - Master Port: {os.environ.get('MASTER_PORT', '29500')}")
        print()
    
    # Call main directly (SLURM + torchrun handles process spawning, NOT mp.spawn)
    main(rank, args)
    
    # Ensure proper cleanup even in edge cases
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning: Error during final cleanup: {e}")
