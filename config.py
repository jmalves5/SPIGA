#!/usr/bin/env python3
"""
SPIGA Training Configuration Templates
======================================

This file contains pre-configured training setups that can be easily loaded
and executed. Users can modify these templates to match their specific needs.

Usage:
    from train_spiga_config import get_config_wflw_full, get_config_cofw68_quick
    
    # Get a configuration
    config_dict = get_config_wflw_full()
    
    # Or create your own by modifying an existing template
    config = get_config_cofw68_quick()
    config['num_landmarks'] = 68
    config['batch_size'] = 16
"""

def get_config_wflw_full():
    """
    Full three-stage training on WFLW (98 landmarks).
    This matches the configuration from the SPIGA paper exactly.
    
    Expected training time: ~3-5 days on single GPU (RTX3090/A100)
    Memory requirement: ~24GB VRAM
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'all',
        
        # Epochs per stage
        'epochs_stage1': 450,
        'epochs_stage2': 150,
        'epochs_gat': 150,
        
        # Learning rates
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        # Loss weights
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        # Training parameters
        'batch_size': 24,
        'num_workers': 4,
        'device': 'cuda',
        'seed': 42,
        
        # Optimization
        'lr_decay': 'step',
        'lr_decay_epoch': 50,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': True,
        
        # Output
        'checkpoint_dir': './checkpoints/wflw_full',
        'log_dir': './logs/wflw_full',
        'save_frequency': 10,
        'validate_frequency': 1,
    }


def get_config_cofw68_full():
    """
    Full training on COFW68 (68 landmarks).
    COFW (Caltech Occluded Faces in the Wild) - occluded face dataset.
    
    Expected training time: ~2-4 days on single GPU
    Memory requirement: ~20GB VRAM
    """
    return {
        'dataset': 'cofw68',
        'num_landmarks': 68,
        'stage': 'all',
        
        'epochs_stage1': 450,
        'epochs_stage2': 150,
        'epochs_gat': 150,
        
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 24,
        'num_workers': 4,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 50,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/cofw68_full',
        'log_dir': './logs/cofw68_full',
        'save_frequency': 10,
        'validate_frequency': 1,
    }


def get_config_quick_debug():
    """
    Quick debug configuration for testing the training pipeline.
    Uses minimal epochs and small batch size.
    
    Expected training time: ~5-10 minutes on single GPU
    Memory requirement: ~4GB VRAM
    Perfect for: Testing code changes, verifying data pipeline, GPU check
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'all',
        
        # Very short training for testing
        'epochs_stage1': 3,
        'epochs_stage2': 2,
        'epochs_gat': 2,
        
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 4,  # Small batch for memory efficiency
        'num_workers': 2,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 1,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/debug',
        'log_dir': './logs/debug',
        'save_frequency': 1,
        'validate_frequency': 1,
    }


def get_config_cpu_test():
    """
    CPU-only configuration for testing on machines without GPU.
    
    Expected training time: ~30-60 minutes per epoch (DO NOT USE FOR PRODUCTION)
    Memory requirement: ~8GB RAM
    Perfect for: Development, debugging, CI/CD testing
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'all',
        
        'epochs_stage1': 2,
        'epochs_stage2': 1,
        'epochs_gat': 1,
        
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 2,
        'num_workers': 0,
        'device': 'cpu',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 1,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': False,  # AMP doesn't work well on CPU
        
        'checkpoint_dir': './checkpoints/cpu_test',
        'log_dir': './logs/cpu_test',
        'save_frequency': 1,
        'validate_frequency': 1,
    }


def get_config_stage1_only():
    """
    Stage 1 only: CNN backbone pre-training.
    
    Use this when:
    - Pre-training backbone for other tasks
    - Extending training from a checkpoint
    - Testing CNN architecture changes
    
    Expected training time: ~1-2 days on single GPU
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'stage1',
        
        'epochs_stage1': 450,
        'epochs_stage2': 0,  # Ignored
        'epochs_gat': 0,     # Ignored
        
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 24,
        'num_workers': 4,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 50,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/stage1_only',
        'log_dir': './logs/stage1_only',
        'save_frequency': 10,
        'validate_frequency': 1,
    }


def get_config_high_accuracy():
    """
    High-accuracy training configuration with longer training.
    
    Expected training time: ~7-10 days on single GPU
    Memory requirement: ~24GB VRAM
    Tuning: More epochs, lower learning rates, stricter validation
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'all',
        
        'epochs_stage1': 600,  # +150 from default
        'epochs_stage2': 200,  # +50 from default
        'epochs_gat': 200,     # +50 from default
        
        'lr_stage1': 5e-4,     # Lower initial LR
        'lr_stage2': 2.5e-4,
        'lr_gat': 2.5e-5,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 24,
        'num_workers': 4,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 100,  # Slower decay
        'lr_decay_gamma': 0.95, # Gentler decay
        'max_grad_norm': 0.5,   # Stricter gradient clipping
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/high_accuracy',
        'log_dir': './logs/high_accuracy',
        'save_frequency': 20,   # Save more frequently
        'validate_frequency': 1,
    }


def get_config_fast_training():
    """
    Fast training configuration for quick results.
    Trades some accuracy for speed.
    
    Expected training time: ~1-2 days on single GPU
    Memory requirement: ~16GB VRAM
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'all',
        
        'epochs_stage1': 300,   # -150 from default
        'epochs_stage2': 100,   # -50 from default
        'epochs_gat': 100,      # -50 from default
        
        'lr_stage1': 2e-3,      # Higher initial LR
        'lr_stage2': 1e-3,
        'lr_gat': 1e-4,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 32,       # Larger batch
        'num_workers': 6,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 30,   # Faster decay
        'lr_decay_gamma': 0.85, # More aggressive decay
        'max_grad_norm': 2.0,   # More lenient gradient clipping
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/fast',
        'log_dir': './logs/fast',
        'save_frequency': 5,
        'validate_frequency': 2,  # Validate less frequently
    }


def get_config_300w_public():
    """
    Training on 300W Public dataset (68 landmarks).
    300W: challenging in-the-wild face detection dataset.
    
    Expected training time: ~2-3 days on single GPU
    """
    return {
        'dataset': '300wpublic',
        'num_landmarks': 68,
        'stage': 'all',
        
        'epochs_stage1': 450,
        'epochs_stage2': 150,
        'epochs_gat': 150,
        
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        'lambda_coord': 4.0,
        'lambda_att': 50.0,
        'lambda_p': 1.0,
        
        'batch_size': 24,
        'num_workers': 4,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 50,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/300w_public',
        'log_dir': './logs/300w_public',
        'save_frequency': 10,
        'validate_frequency': 1,
    }


def get_config_custom_loss_weights():
    """
    Configuration with custom loss weights for experimentation.
    
    Adjust these weights to emphasize different aspects:
    - lambda_coord: Higher = stricter coordinate regression
    - lambda_att: Higher = stricter attention/heatmap regression
    - lambda_p: Higher = stricter pose estimation
    """
    return {
        'dataset': 'wflw',
        'num_landmarks': 98,
        'stage': 'all',
        
        'epochs_stage1': 450,
        'epochs_stage2': 150,
        'epochs_gat': 150,
        
        'lr_stage1': 1e-3,
        'lr_stage2': 5e-4,
        'lr_gat': 5e-5,
        
        # Custom loss weights - adjust these!
        'lambda_coord': 2.0,    # Was 4.0 - lower weight on coordinates
        'lambda_att': 100.0,    # Was 50.0 - higher weight on heatmaps
        'lambda_p': 0.5,        # Was 1.0 - lower weight on pose
        
        'batch_size': 24,
        'num_workers': 4,
        'device': 'cuda',
        'seed': 42,
        
        'lr_decay': 'step',
        'lr_decay_epoch': 50,
        'lr_decay_gamma': 0.9,
        'max_grad_norm': 1.0,
        'use_amp': True,
        
        'checkpoint_dir': './checkpoints/custom_loss',
        'log_dir': './logs/custom_loss',
        'save_frequency': 10,
        'validate_frequency': 1,
    }


def dict_to_args(config_dict):
    """
    Convert a configuration dictionary to command-line arguments.
    
    Usage:
        config = get_config_wflw_full()
        args_str = dict_to_args(config)
        print(f"python train_spiga_complete.py {args_str}")
    """
    args = []
    for key, value in config_dict.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key} {value}")
    return " ".join(args)


if __name__ == "__main__":
    """
    Display all available configurations with their command-line equivalents.
    """
    configs = {
        'wflw_full': get_config_wflw_full,
        'cofw68_full': get_config_cofw68_full,
        'quick_debug': get_config_quick_debug,
        'cpu_test': get_config_cpu_test,
        'stage1_only': get_config_stage1_only,
        'high_accuracy': get_config_high_accuracy,
        'fast_training': get_config_fast_training,
        '300w_public': get_config_300w_public,
        'custom_loss_weights': get_config_custom_loss_weights,
    }
    
    print("=" * 70)
    print("SPIGA Training Configuration Templates")
    print("=" * 70)
    print()
    
    for name, config_func in configs.items():
        config = config_func()
        args_str = dict_to_args(config)
        
        print(f"Configuration: {name}")
        print(f"Dataset: {config['dataset']}")
        print(f"Command:")
        print(f"  python train_spiga_complete.py {args_str}")
        print()
