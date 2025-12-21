import argparse

def parse_args():
    """Parse command line arguments for OpenSERGE training."""
    ap = argparse.ArgumentParser(description='Train OpenSERGE road graph extraction model')

    # Config file
    ap.add_argument('--config', type=str, default=None,
                    help='Path to config JSON file (overrides other args)')

    # Data
    ap.add_argument('--data_root', type=str, required=False,
                    help='Path to dataset root directory')
    ap.add_argument('--img_size', type=int, default=512,
                    help='Input image size')
    ap.add_argument('--preload', action='store_true',
                    help='Preload entire dataset into memory for faster training')

    # Model
    ap.add_argument('--backbone', type=str, default='resnet50',
                    choices=['resnet18', 'resnet50'],
                    help='Backbone architecture')
    ap.add_argument('--k', type=int, default=None,
                    help='k for k-NN graph prior; None=complete graph')
    ap.add_argument('--pretrained_cnn', type=str, default=None,
                    help='Path to pre-trained SingleShotRoadGraphNet checkpoint')
    ap.add_argument('--freeze_pretrained_cnn', action='store_true',
                    help='Freeze pre-trained CNN weights (only train GNN)')

    # Training
    ap.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
    ap.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
    ap.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
    ap.add_argument('--junction_thresh', type=float, default=0.5,
                    help='Junction threshold for inference')

    # Loss weights
    ap.add_argument('--loss_weight_junction', type=float, default=1.0,
                    help='Weight for junction loss')
    ap.add_argument('--loss_weight_offset', type=float, default=1.0,
                    help='Weight for offset loss')
    ap.add_argument('--loss_weight_edge', type=float, default=1.0,
                    help='Weight for edge loss')

    # Early stopping
    ap.add_argument('--early_stop_patience', type=int, default=10,
                    help='Early stopping patience (epochs)')
    ap.add_argument('--min_delta', type=float, default=1e-4,
                    help='Minimum change in validation loss for improvement')

    # Checkpointing
    ap.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help='Directory to save checkpoints')
    ap.add_argument('--save_freq', type=int, default=5,
                    help='Save checkpoint every N epochs')
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to full OpenSERGE checkpoint to resume training from')

    # Logging
    ap.add_argument('--log_dir', type=str, default='logs',
                    help='Directory for logs and tensorboard')
    ap.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name for logging')

    # Weights & Biases
    ap.add_argument('--wandb_project', type=str, default='openserge',
                    help='W&B project name')
    ap.add_argument('--wandb_entity', type=str, default=None,
                    help='W&B entity (team) name')
    ap.add_argument('--wandb_run_name', type=str, default=None,
                    help='W&B run name (default: auto-generated)')
    ap.add_argument('--wandb_tags', type=str, nargs='*', default=None,
                    help='W&B run tags')
    ap.add_argument('--disable_wandb', action='store_true',
                    help='Disable W&B logging')

    # System
    ap.add_argument('--device', type=str, default='cuda',
                    help='Device to use (cuda/cpu)')
    ap.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')

    return ap.parse_args()