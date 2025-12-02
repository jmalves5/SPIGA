import os
import torch
import wandb
import torch.distributed as dist
import logging
from datetime import datetime

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
        wandb_run_name = os.environ.get('WANDB_RUN_NAME', f"spiga_{args.dataset}_stage1")
        
        # Set mode to offline if no API key is set
        wandb_mode = "online" if os.environ.get('WANDB_API_KEY') else "offline"
        print(f"Initializing wandb for rank {rank}")
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            tags=[args.dataset, "stage1", f"world_size_{args.world_size}"],
            mode=wandb_mode,
            reinit=False,  # Prevent multiple initializations
            dir="/tmp"  # Use temp directory to avoid conflicts
        )
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None

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
