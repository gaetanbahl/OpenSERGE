"""
Training script for junction detection (backbone + junction/offset head only).
This script trains a simplified version of OpenSERGE focusing only on the
junction detection and offset regression tasks, without the GNN edge prediction.
"""
import argparse
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from .data.dataset import CityScale
from .models.net import SingleShotRoadGraphNet
from .models.losses import sigmoid_focal_loss, masked_mse
from .utils.graph import collate_fn
from .utils.training import save_checkpoint, load_checkpoint, setup_logging, load_config, set_seed, save_config, EarlyStopping


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description='Train junction detection on CityScale dataset')

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
    ap.add_argument('--nfeat', type=int, default=256,
                    help='Number of feature channels')

    # Training
    ap.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
    ap.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
    ap.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')

    # Loss weights
    ap.add_argument('--loss_weight_junction', type=float, default=1.0,
                    help='Weight for junction loss')
    ap.add_argument('--loss_weight_offset', type=float, default=1.0,
                    help='Weight for offset loss')
    ap.add_argument('--focal_alpha', type=float, default=0.25,
                    help='Focal loss alpha parameter')
    ap.add_argument('--focal_gamma', type=float, default=2.0,
                    help='Focal loss gamma parameter')

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
                    help='Path to checkpoint to resume from')

    # Logging
    ap.add_argument('--log_dir', type=str, default='logs',
                    help='Directory for logs and tensorboard')
    ap.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name for logging')

    # Weights & Biases
    ap.add_argument('--wandb_project', type=str, default='openserge_junction',
                    help='W&B project name (default: openserge_junction)')
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


def compute_losses(outputs, targets, config):
    """
    Compute junction and offset losses.

    Args:
        outputs: Dict with 'junction_logits' [B,1,h,w] and 'offset' [B,2,h,w]
        targets: Dict with 'junction_map', 'offset_map', 'offset_mask'
        config: Config dict with loss hyperparameters

    Returns:
        Dict with individual losses and total loss
    """
    j_logits = outputs['junction_logits']
    offset_pred = outputs['offset']

    j_tgt = targets['junction_map'].to(j_logits.device)
    offset_tgt = targets['offset_map'].to(offset_pred.device)
    offset_mask = targets['offset_mask'].to(offset_pred.device)

    # Junction focal loss
    loss_junction = sigmoid_focal_loss(
        j_logits, j_tgt,
        alpha=config.get('focal_alpha', 0.25),
        gamma=config.get('focal_gamma', 2.0)
    )

    # Offset masked MSE loss
    loss_offset = masked_mse(offset_pred, offset_tgt, offset_mask)

    # Total weighted loss - only include components with non-zero weights
    total_loss = torch.tensor(0.0, device=j_logits.device, requires_grad=True)
    if config.get('loss_weight_junction', 1.0) > 0:
        total_loss = total_loss + config['loss_weight_junction'] * loss_junction
    if config.get('loss_weight_offset', 1.0) > 0:
        total_loss = total_loss + config['loss_weight_offset'] * loss_offset

    return {
        'total': total_loss,
        'junction': loss_junction,
        'offset': loss_offset
    }


def train_epoch(model, dataloader, optimizer, device, config, writer, epoch):
    """Train for one epoch."""
    model.train()

    epoch_losses = {'total': 0.0, 'junction': 0.0, 'offset': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config["epochs"]} [Train]')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = {
            'junction_map': batch['junction_map'],
            'offset_map': batch['offset_map'],
            'offset_mask': batch['offset_mask']
        }

        # Forward pass
        outputs = model(images)
        losses = compute_losses(outputs, targets, config)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        optimizer.step()

        # Accumulate losses
        for k in epoch_losses.keys():
            epoch_losses[k] += losses[k].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'L_j': f"{losses['junction'].item():.4f}",
            'L_o': f"{losses['offset'].item():.4f}",
            'L_tot': f"{losses['total'].item():.4f}"
        })

        # Log to tensorboard and W&B (every 10 batches)
        if batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx

            # TensorBoard logging
            if writer:
                writer.add_scalar('Train/Loss_Total', losses['total'].item(), global_step)
                writer.add_scalar('Train/Loss_Junction', losses['junction'].item(), global_step)
                writer.add_scalar('Train/Loss_Offset', losses['offset'].item(), global_step)

            # W&B logging
            if not config.get('disable_wandb', False):
                wandb.log({
                    'train/loss_total': losses['total'].item(),
                    'train/loss_junction': losses['junction'].item(),
                    'train/loss_offset': losses['offset'].item(),
                    'train/epoch': epoch,
                }, step=global_step)

    # Average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    return avg_losses


@torch.no_grad()
def validate_epoch(model, dataloader, device, config, writer, epoch):
    """Validate for one epoch."""
    model.eval()

    epoch_losses = {'total': 0.0, 'junction': 0.0, 'offset': 0.0}
    num_batches = 0

    # Metrics
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config["epochs"]} [Valid]')
    for batch in pbar:
        images = batch['image'].to(device)
        targets = {
            'junction_map': batch['junction_map'],
            'offset_map': batch['offset_map'],
            'offset_mask': batch['offset_mask']
        }

        # Forward pass
        outputs = model(images)
        losses = compute_losses(outputs, targets, config)

        # Accumulate losses
        for k in epoch_losses.keys():
            epoch_losses[k] += losses[k].item()
        num_batches += 1

        # Compute metrics (using threshold of 0.5 after sigmoid)
        j_pred = torch.sigmoid(outputs['junction_logits']) > 0.5
        j_gt = targets['junction_map'].to(device) > 0.5

        total_tp += (j_pred & j_gt).sum().item()
        total_fp += (j_pred & ~j_gt).sum().item()
        total_fn += (~j_pred & j_gt).sum().item()

    # Average losses
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

    # Compute F1 score
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Log to tensorboard
    if writer:
        writer.add_scalar('Valid/Loss_Total', avg_losses['total'], epoch)
        writer.add_scalar('Valid/Loss_Junction', avg_losses['junction'], epoch)
        writer.add_scalar('Valid/Loss_Offset', avg_losses['offset'], epoch)
        writer.add_scalar('Valid/Precision', precision, epoch)
        writer.add_scalar('Valid/Recall', recall, epoch)
        writer.add_scalar('Valid/F1', f1, epoch)

    # Log to W&B
    if not config.get('disable_wandb', False):
        wandb.log({
            'val/loss_total': avg_losses['total'],
            'val/loss_junction': avg_losses['junction'],
            'val/loss_offset': avg_losses['offset'],
            'val/precision': precision,
            'val/recall': recall,
            'val/f1': f1,
            'epoch': epoch,
        })  # No step parameter - let wandb auto-increment

    return avg_losses, metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load config from file if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = vars(args)

    # Set random seed
    set_seed(config['seed'])

    # Setup experiment name
    if config.get('experiment_name') is None:
        config['experiment_name'] = f"junction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Setup directories
    exp_dir = Path(config['log_dir']) / config['experiment_name']
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(exp_dir)
    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Save config
    save_config(config, exp_dir / 'config.json')

    # Setup tensorboard
    writer = SummaryWriter(exp_dir / 'tensorboard')

    # Setup Weights & Biases
    if not config.get('disable_wandb', False):
        wandb_config = {
            'data_root': config.get('data_root', 'N/A'),
            'img_size': config['img_size'],
            'backbone': config.get('backbone', 'resnet50'),
            'nfeat': config.get('nfeat', 256),
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'lr': config['lr'],
            'weight_decay': config.get('weight_decay', 1e-4),
            'loss_weight_junction': config.get('loss_weight_junction', 1.0),
            'loss_weight_offset': config.get('loss_weight_offset', 1.0),
            'focal_alpha': config.get('focal_alpha', 0.25),
            'focal_gamma': config.get('focal_gamma', 2.0),
            'early_stop_patience': config.get('early_stop_patience', 10),
            'min_delta': config.get('min_delta', 1e-4),
            'seed': config['seed'],
            'preload': config.get('preload', False),
        }

        wandb.init(
            project=config.get('wandb_project', 'openserge_junction'),
            entity=config.get('wandb_entity'),
            name=config.get('wandb_run_name', config['experiment_name']),
            tags=config.get('wandb_tags'),
            config=wandb_config,
            dir=str(exp_dir)
        )

        logger.info(f"W&B project: {config.get('wandb_project', 'openserge_junction')}")
        logger.info(f"W&B run: {wandb.run.name if wandb.run else 'N/A'}")
    else:
        logger.info("W&B logging disabled")

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets with skip_edges=True for faster training
    logger.info("Loading datasets...")
    preload = config.get('preload', False)
    train_dataset = CityScale(
        config['data_root'],
        split='train',
        img_size=config['img_size'],
        aug=True,
        preload=preload,
        skip_edges=True  # Skip edge computation for junction-only training
    )
    val_dataset = CityScale(
        config['data_root'],
        split='valid',
        img_size=config['img_size'],
        aug=False,
        preload=preload,
        skip_edges=True  # Skip edge computation for junction-only training
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Create model
    logger.info("Creating model...")
    model = SingleShotRoadGraphNet(
        backbone=config.get('backbone', 'resnet50'),
        nfeat=config.get('nfeat', 256)
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Watch model with W&B
    if not config.get('disable_wandb', False):
        wandb.watch(model, log='all', log_freq=100)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    logger.info(f"Optimizer created")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    if config.get('resume'):
        logger.info(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'val_losses' in checkpoint and checkpoint['val_losses']:
            best_val_loss = checkpoint['val_losses']['total']

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stop_patience'],
        min_delta=config['min_delta'],
        mode='min'
    )

    # Training loop
    logger.info("Starting training...")
    train_history = {'train': [], 'val': []}

    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start_time = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device,
                                   config, writer, epoch)

        # Validate
        val_losses, val_metrics = validate_epoch(model, val_loader, device,
                                                 config, writer, epoch)

        epoch_time = time.time() - epoch_start_time

        # Log epoch summary
        logger.info(
            f"Epoch {epoch}/{config['epochs']} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_losses['total']:.4f} "
            f"(J={train_losses['junction']:.4f}, "
            f"O={train_losses['offset']:.4f}) | "
            f"Val Loss: {val_losses['total']:.4f} "
            f"(J={val_losses['junction']:.4f}, "
            f"O={val_losses['offset']:.4f}) | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        # Save history
        train_history['train'].append(train_losses)
        train_history['val'].append(val_losses)

        # Check if best model
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

            # Log best validation loss to W&B
            if not config.get('disable_wandb', False):
                wandb.log({'val/best_loss': best_val_loss})

        # Save checkpoint
        if epoch % config['save_freq'] == 0 or is_best:
            save_path = checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, config, save_path,
                          is_best, train_losses, val_losses)
            logger.info(f"Saved checkpoint: {save_path}")

            # Log checkpoint to W&B
            if not config.get('disable_wandb', False):
                artifact = wandb.Artifact(
                    name=f"checkpoint_epoch{epoch}",
                    type='model',
                    description=f"Model checkpoint at epoch {epoch}"
                )
                artifact.add_file(str(save_path))
                wandb.log_artifact(artifact)

                # Log best model separately
                if is_best:
                    best_model_path = checkpoint_dir / 'best_model.pt'
                    if best_model_path.exists():
                        best_artifact = wandb.Artifact(
                            name='best_model',
                            type='model',
                            description=f"Best model (epoch {epoch}, val_loss={best_val_loss:.4f})"
                        )
                        best_artifact.add_file(str(best_model_path))
                        wandb.log_artifact(best_artifact)

        # Early stopping check
        if early_stopping(val_losses['total']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('Train/LearningRate', current_lr, epoch)
        if not config.get('disable_wandb', False):
            wandb.log({'train/learning_rate': current_lr})

    # Save final checkpoint
    final_path = checkpoint_dir / 'final_model.pt'
    save_checkpoint(model, optimizer, epoch, config, final_path,
                   False, train_losses, val_losses)

    # Save training history
    history_path = exp_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Results saved to: {exp_dir}")

    # Log final artifacts to W&B
    if not config.get('disable_wandb', False):
        # Log final model
        final_artifact = wandb.Artifact(
            name='final_model',
            type='model',
            description=f"Final model after {epoch} epochs"
        )
        final_artifact.add_file(str(final_path))
        wandb.log_artifact(final_artifact)

        # Log training history
        history_artifact = wandb.Artifact(
            name='training_history',
            type='results',
            description='Training and validation loss history'
        )
        history_artifact.add_file(str(history_path))
        wandb.log_artifact(history_artifact)

        # Log final summary metrics
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.summary['best_f1'] = max([h['f1'] if 'f1' in h else 0 for h in train_history['val']])
        wandb.summary['final_epoch'] = epoch
        wandb.summary['total_params'] = num_params
        wandb.summary['trainable_params'] = num_trainable

        # Finish W&B run
        wandb.finish()
        logger.info("W&B run finished successfully")

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
