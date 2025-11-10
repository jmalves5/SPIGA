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

# ======================== Memory Management ========================
def setup_memory_optimizations():
    """Enable memory-efficient CUDA settings"""
    
    # Set CUBLAS to allow non-deterministic algorithms (required for CUDA >= 10.2)
    # This is necessary because some CUDA operations don't support deterministic mode
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable cuDNN benchmark for faster convolutions (with fixed input sizes)
    torch.backends.cudnn.benchmark = True
    
    # Disable deterministic algorithms due to CUDA 10.2+ compatibility
    # Using deterministic_algorithms would require all operations to be deterministic,
    # but CuBLAS operations don't support it with CUDA >= 10.2
    torch.use_deterministic_algorithms(False)

# ======================== Weights & Biases Setup ========================
def setup_wandb(args, rank: int):
    """Initialize wandb logging (only on main process)"""
    if rank != 0:
        return None
    try:
        wandb_project = os.environ.get('WANDB_PROJECT', 'spiga-training')
        wandb_entity = os.environ.get('WANDB_ENTITY', None)
        wandb_run_name = os.environ.get('WANDB_RUN_NAME', f"spiga_{args.dataset}_stage2")
        
        # Set mode to offline if no API key is set
        wandb_mode = "online" if os.environ.get('WANDB_API_KEY') else "offline"
        print(f"Initializing wandb for rank {rank}")
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            tags=[args.dataset, "stage2", f"world_size_{args.world_size}"],
            mode=wandb_mode,
            reinit=False,  # Prevent multiple initializations
            dir="/tmp"  # Use temp directory to avoid conflicts
        )
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None

# SPIGA imports
try:
    from spiga.models.spiga import SPIGA
    from spiga.data.loaders.dl_config import AlignConfig
    from spiga.data.loaders.dataloader import get_dataloader
except ImportError as e:
    print(f"Error importing SPIGA modules: {e}")
    print("Please ensure SPIGA is installed: pip install -e .")
    sys.exit(1)

def setup(rank, world_size):
    """Initialize distributed training with NCCL backend.
    
    For SLURM: RANK and WORLD_SIZE come from srun environment.

    """
    # Check if already initialized (important for multi-stage training)
    if dist.is_available() and not dist.is_initialized():
        # Use environment variables for SLURM compatibility
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        if rank == 0:
            print(f"✓ Initialized process group with rank {rank} and world size {world_size}")
    elif rank == 0:
        print(f"✓ Process group already initialized (rank {rank}, world_size {world_size})")
    
    # Note: Don't call torch.cuda.set_device() here
    # It's called after device initialization in main()
    print(f"[DEBUG] Rank {rank}: setup() returning", flush=True)

def cleanup():
    """Safely destroy distributed process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is main process (rank 0)"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """Get current process rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total number of processes"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# ======================== Logging Setup ========================
def setup_logging(log_dir: str, stage: str, rank: int) -> logging.Logger:
    """Setup logging configuration (only on main process)"""
    if not is_main_process():
        logging.disable(logging.CRITICAL)
        return logging.getLogger(__name__)
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir, 
        f"train_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


# ======================== Loss Functions ========================
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for coordinate regression"""
    def __init__(self, beta: float = 1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta, 
            0.5 * diff**2 / self.beta, 
            diff - 0.5 * self.beta
        )
        return loss.mean()


class LandmarkLoss(nn.Module):
    """Combined loss for landmark detection"""
    def __init__(self, num_stages: int = 4, lambda_coord: float = 4.0, lambda_att: float = 50.0):
        super(LandmarkLoss, self).__init__()
        self.num_stages = num_stages
        self.lambda_coord = lambda_coord
        self.lambda_att = lambda_att
        self.coord_loss = SmoothL1Loss()

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute landmark loss"""
        losses_detail = {}
        
        # Simple L1 loss on landmarks if available
        total_loss = None
        if "Landmarks" in predictions and "landmarks" in targets:
            # predictions["Landmarks"] is a list of refined landmark predictions
            landmarks_pred = predictions["Landmarks"][-1]  # Take last refinement
            landmarks_target = targets["landmarks"]
            
            loss = self.coord_loss(landmarks_pred, landmarks_target)
            total_loss = self.lambda_coord * loss
            losses_detail["landmarks"] = loss.item()
        
        # If no landmarks loss, return zero loss with gradient
        if total_loss is None:
            total_loss = torch.tensor(0.0, requires_grad=True)
        
        losses_detail["loss_landmark"] = total_loss.item()
        return total_loss, losses_detail


class CombinedLoss(nn.Module):
    """Combined loss for landmark detection and pose estimation"""
    def __init__(self, num_stages: int = 4, lambda_coord: float = 4.0, 
                 lambda_att: float = 50.0, lambda_p: float = 1.0):
        super(CombinedLoss, self).__init__()
        self.num_stages = num_stages
        self.lambda_p = lambda_p
        self.landmark_loss = LandmarkLoss(num_stages, lambda_coord, lambda_att)
        self.pose_loss = nn.MSELoss()
        self.coord_loss = SmoothL1Loss()

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute combined landmark and pose loss"""
        # Landmark loss
        lnd_loss, lnd_details = self.landmark_loss(predictions, targets)
        losses_detail = lnd_details.copy()

        # Pose loss (if predictions include pose)
        pose_loss_total = torch.tensor(0.0, device=lnd_loss.device, dtype=lnd_loss.dtype)
        
        if 'Pose' in predictions and 'pose' in targets:
            pose_pred = predictions['Pose']
            pose_loss_total = self.lambda_p * self.pose_loss(pose_pred, targets['pose'])
            losses_detail['pose'] = pose_loss_total.item()

        total_loss = lnd_loss + pose_loss_total
        losses_detail['loss_pose'] = pose_loss_total.item()
        losses_detail['loss_total'] = total_loss.item()

        return total_loss, losses_detail


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
            if "headpose" in batch:
                targets["pose"] = batch["headpose"].to(device, dtype=torch.float32)

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

            # Forward pass
            predictions = model([images, model3d, cam_matrix])
            loss, loss_details = criterion(predictions, targets)

            # Save loss values before deletion
            loss_val = loss.item()
            loss_details_copy = {k: v for k, v in loss_details.items()}

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
            optimizer.step()
            
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

                # Forward pass
                if model3d is not None and cam_matrix is not None:
                    predictions = model([images, model3d, cam_matrix])
                else:
                    predictions = model(images)
                loss, loss_details = criterion(predictions, targets)

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
    """Stage 2: Fine-tuning with both landmark detection and pose estimation"""
    if is_main_process():
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: Fine-tuning (Landmarks + Pose)")
        logger.info("="*80)
        logger.info(f"Epochs: {args.epochs_stage2}")
        logger.info(f"Learning Rate: {args.lr_stage2}")
        logger.info(f"Pretrained weights: {pretrained_path}")

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

    # Model with pre-trained weights
    model = SPIGA(
        num_landmarks=args.num_landmarks,
        num_edges=args.num_edges
    ).to(device)
    
    # Wrap with DistributedDataParallel
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )
    
    if os.path.exists(pretrained_path):
        load_checkpoint(pretrained_path, model, device=device, logger=logger)
    else:
        if is_main_process():
            logger.warning(f"Pretrained weights not found: {pretrained_path}")

    if is_main_process():
        logger.info(f"Model created with {args.num_landmarks} landmarks")
        if world_size > 1:
            logger.info(f"Model wrapped with DistributedDataParallel")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(
        num_stages=args.num_stages,
        lambda_coord=args.lambda_coord,
        lambda_att=args.lambda_att,
        lambda_p=args.lambda_p
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr_stage2)
    scheduler = StepLR(optimizer, step_size=args.decay_epoch_stage2, gamma=0.1)

    best_loss = float('inf')
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoint_stage2')

    # Training loop
    for epoch in range(args.epochs_stage2):
        if is_main_process():
            logger.info(f"\nEpoch {epoch+1}/{args.epochs_stage2}")
        
        # Train
        train_loss, train_details = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch+1, logger, rank, world_size, wandb_log=wandb_log
        )
        
        if is_main_process():
            logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss, val_details = validate(model, val_loader, criterion, device, logger, rank, world_size, epoch=epoch+1, wandb_log=wandb_log)
        
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
    wandb_log = setup_wandb(args, rank) if rank == 0 else None

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
    parser.add_argument("--batch_size", type=int, default=12,
                       help="Batch size per GPU (very small for large models)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
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
