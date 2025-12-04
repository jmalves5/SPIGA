"""
SPIGA Training Script - Stage 1
Shape Preserving Facial Landmarks with Graph Attention Networks

Paper: "Shape Preserving Facial Landmarks with Graph Attention Networks"
       Prados-Torreblanca et al. (BMVC 2022)
       https://arxiv.org/abs/2210.07233

Stage 1: Pre-training CNN backbone with landmark detection (450 epochs)

Multi-GPU Training:
    - Uses torch.nn.parallel.DistributedDataParallel (DDP)
    - Compatible with SLURM sbatch and srun
    - Automatic rank/world_size detection from environment
    - Gradient synchronization across GPUs
    - Proper handling of batch size scaling

Usage with SLURM:
    srun python train_stage1.py --dataset wflw
    
    Or submit with sbatch:
    sbatch train_stage1.sh

Usage with torchrun:
    torchrun --nproc_per_node=4 train_stage1.py --dataset wflw

Usage with single GPU:
    python train_stage1.py --dataset wflw

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

from utils import setup_memory_optimizations, setup_wandb, setup, cleanup, is_main_process, setup_logging

MANUAL_SEED = 42

from spiga.models.spiga import SPIGA
from spiga.data.loaders.dl_config import AlignConfig
from spiga.data.loaders.dataloader import get_dataloader

from loss import LandmarkLoss, CombinedLoss

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
            
            # Add optional targets (no pose in stage 1 - landmark detection only)
            if "heatmaps_points" in batch:
                targets["heatmaps_points"] = batch["heatmaps_points"].to(device, dtype=torch.float32)
            if "heatmaps_edges" in batch:
                targets["heatmaps_edges"] = batch["heatmaps_edges"].to(device, dtype=torch.float32)

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

            # Forward pass with AMP - Stage 1: Train CNN only (no pose, no GNN)
            # Call visual_cnn directly instead of backbone_forward to avoid pose computation
            actual_model = model.module if hasattr(model, 'module') else model
            
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # Direct CNN forward - returns {'VisualField': [...], 'HGcore': [...]}
                predictions = actual_model.visual_cnn(images)
                loss, loss_details = criterion(predictions, targets)

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
            
            # Unscale and clip gradients to prevent explosion
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                # Note: No pose in stage 1 (landmark detection only)

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

                # Forward pass with AMP - Stage 1: Use CNN only (no pose, no GNN)
                actual_model = model.module if hasattr(model, 'module') else model
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Direct CNN forward - no pose computation in Stage 1
                    predictions = actual_model.visual_cnn(images)
                    loss, loss_details = criterion(predictions, targets)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process():
                        logger.error(f"NaN/Inf detected in validation loss at batch {num_batches}")
                        logger.error(f"Loss value: {loss.item()}")
                        logger.error(f"Predictions sample: {predictions['VisualField'][0].min()}, {predictions['VisualField'][0].max()}")
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


def train_stage1(args, logger: logging.Logger, device: torch.device, rank: int, world_size: int, wandb_log=None):
    """Stage 1: Pre-training CNN backbone with landmark detection"""    
    if is_main_process():
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: CNN Backbone Pre-training (Landmarks Only)")
        logger.info("="*80)
        logger.info(f"Epochs: {args.epochs_stage1}")
        logger.info(f"Learning Rate: {args.lr_stage1}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"World Size (GPUs): {world_size}")

    # Data loading with distributed sampler
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

    # Stage 1: Use SPIGA model but only train CNN backbone (visual_cnn)
    # Freeze all GNN components (gcn, shape_encoder, conv_window, pose_fc)
    model = SPIGA(
        num_landmarks=args.num_landmarks,
        num_edges=args.num_edges
    ).to(device)
    
    # CRITICAL: Freeze GNN components for Stage 1 backbone pretraining
    # Only the CNN backbone (visual_cnn) should be trained
    for param in model.gcn.parameters():
        param.requires_grad = False
    for param in model.shape_encoder.parameters():
        param.requires_grad = False
    for param in model.conv_window.parameters():
        param.requires_grad = False
    for param in model.pose_fc.parameters():
        param.requires_grad = False
    
    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ SPIGA model created with {args.num_landmarks} landmarks")
        logger.info(f"✓ GNN components frozen (gcn, shape_encoder, conv_window, pose_fc)")
        logger.info(f"✓ Training CNN backbone (visual_cnn) ONLY")
        logger.info(f"✓ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Wrap with DistributedDataParallel
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # GNN params are frozen but still in graph
        )
    
    if is_main_process() and world_size > 1:
        logger.info(f"✓ Model wrapped with DistributedDataParallel")

    # Loss, optimizer, scheduler
    criterion = LandmarkLoss(
        num_stages=args.num_stages,
        lambda_coord=args.lambda_coord,
        lambda_att=args.lambda_att
    ).to(device)
    
    # Optimizer: optimize CNN backbone + regression heads in criterion
    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if hasattr(model, 'module') else model
    backbone_params = list(actual_model.visual_cnn.parameters()) + list(criterion.stage1_heads.parameters())
    optimizer = optim.Adam(backbone_params, lr=args.lr_stage1)
    scheduler = StepLR(optimizer, step_size=args.decay_epoch_stage1, gamma=0.1)
    
    # AMP: Mixed precision training with gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    if is_main_process():
        logger.info(f"✓ Optimizer: Adam with lr={args.lr_stage1} (backbone only)")
        logger.info(f"✓ Scheduler: StepLR (decay at epoch {args.decay_epoch_stage1})")
        logger.info(f"✓ AMP: {'Enabled' if args.use_amp else 'Disabled'}")

    # Resume from checkpoint if available
    start_epoch = 0
    if args.resume and is_main_process():
        start_epoch = load_checkpoint(args.resume, model, optimizer, device, logger)

    # Broadcast start_epoch to all processes
    if world_size > 1:
        start_epoch_tensor = torch.tensor(start_epoch, device=device, dtype=torch.long)
        dist.broadcast(start_epoch_tensor, 0)
        start_epoch = int(start_epoch_tensor.item())

    best_loss = float('inf')
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoint_stage1')

    # Training loop
    for epoch in range(start_epoch, args.epochs_stage1):

        
        if is_main_process():
            logger.info(f"\nEpoch {epoch+1}/{args.epochs_stage1}")
        
        # Train
        train_loss, train_details = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch+1, logger, rank, world_size, scaler=scaler, 
            use_amp=args.use_amp, 
            num_landmarks=args.num_landmarks, wandb_log=wandb_log
        )
        
        if is_main_process():
            logger.info(f"Train Loss: {train_loss:.6f}")

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

        # Save best checkpoint (main process only)
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
        logger.info(f"Stage 1 completed! Best validation loss: {best_loss:.6f}")
        
        # Log best model to wandb
        if wandb_log is not None:
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            artifact = wandb.Artifact(
                name=f"spiga-stage1-{args.dataset}",
                type="model",
                description=f"Best SPIGA Stage 1 model trained on {args.dataset} dataset",
                metadata={
                    "dataset": args.dataset,
                    "stage": 1,
                    "epochs": args.epochs_stage1,
                    "best_val_loss": best_loss,
                    "num_landmarks": args.num_landmarks,
                    "learning_rate": args.lr_stage1,
                    "batch_size": args.batch_size,
                    "world_size": world_size
                }
            )
            artifact.add_file(best_model_path)
            wandb_log.log_artifact(artifact)
            logger.info(f"Best model logged to wandb: {best_model_path}")
    
    return os.path.join(checkpoint_dir, 'best_model.pth')


def main(rank, args):
    torch.manual_seed(MANUAL_SEED)
    """Main training function for SLURM-based distributed multi-GPU training.
    
    Rank and world_size are passed as parameters from torchrun launcher.
    For single GPU testing, rank defaults to 0 and world_size to 1.
    """
    
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
    wandb_log = setup_wandb(args, rank) if rank == 0 else None

    # Setup logging
    logger = setup_logging(args.log_dir, 'stage1', rank)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoint_stage1')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration (main process only)
    if rank == 0:
        config_path = os.path.join(checkpoint_dir, "config_stage1.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Configuration saved to: {config_path}")

        # Print configuration
        logger.info("\n" + "="*80)
        logger.info("SPIGA Training Stage 1 Configuration (Distributed Multi-GPU)")
        logger.info("="*80)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Number of landmarks: {args.num_landmarks}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Total batch size: {args.batch_size * world_size}")
        logger.info(f"World size (GPUs): {world_size}")
        logger.info("="*80)
    

    # Run training
    train_stage1(args, logger, device, rank, world_size, wandb_log=wandb_log)

    if is_main_process():
        logger.info("\n" + "="*80)
        logger.info("Stage 1 training complete!")
        logger.info("="*80)

    # Cleanup wandb
    if wandb_log is not None and is_main_process():
        wandb_log.finish()
    
    # Cleanup distributed training
    if world_size > 1:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPIGA Training Script - Stage 1: CNN Backbone Pre-training",
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

    # Stage 1 arguments
    parser.add_argument("--epochs_stage1", type=int, default=450,
                       help="Number of epochs for Stage 1")
    parser.add_argument("--lr_stage1", type=float, default=1e-4,
                       help="Learning rate for Stage 1")
    parser.add_argument("--decay_epoch_stage1", type=int, default=380,
                       help="Epoch to decay learning rate in Stage 1")

    # Model arguments
    parser.add_argument("--num_stages", type=int, default=4,
                       help="Number of Hourglass stages")

    # Loss weights
    parser.add_argument("--lambda_coord", type=float, default=4.0,
                       help="Weight for coordinate loss")
    parser.add_argument("--lambda_att", type=float, default=50.0,
                       help="Weight for attention (heatmap) loss")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=6,
                       help="Batch size per GPU (6 per GPU × 4 GPUs = 24 effective batch size)")
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
                       help="Path to checkpoint to resume training from")

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory to save logs")

    args = parser.parse_args()
    
    # For SLURM: rank and world_size come from environment variables set by torchrun
    # For single GPU: they default to 0 and 1
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
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
