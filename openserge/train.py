import argparse
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data.dataset import CityScale
from .models.wrapper import OpenSERGE
from .models.losses import openserge_losses
from .utils.graph import collate_fn, create_edge_labels
from .utils.training import save_checkpoint, load_checkpoint, setup_logging, load_config, set_seed, save_config, EarlyStopping


def parse_args():
    """Parse command line arguments."""
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
                    help='Path to checkpoint to resume from')

    # Logging
    ap.add_argument('--log_dir', type=str, default='logs',
                    help='Directory for logs and tensorboard')
    ap.add_argument('--experiment_name', type=str, default=None,
                    help='Experiment name for logging')

    # System
    ap.add_argument('--device', type=str, default='cuda',
                    help='Device to use (cuda/cpu)')
    ap.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')

    return ap.parse_args()


def train_epoch(model, dataloader, optimizer, device, config, writer, epoch):
    """Train for one epoch."""
    model.train()

    epoch_losses = {'total': 0, 'junction': 0, 'offset': 0, 'edge': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config["epochs"]} [Train]')

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # Create edge labels from ground truth
        edge_labels = create_edge_labels(
            batch['junction_map'].to(device),
            batch['edges'],
            device=device
        )

        targets = {
            'junction_map': batch['junction_map'].to(device),
            'offset_map': batch['offset_map'].to(device),
            'offset_mask': batch['offset_mask'].to(device),
            'edge_lists': edge_labels
        }

        # Forward pass
        out = model(images, j_thr=config['junction_thresh'])
        losses = openserge_losses(out, targets)

        # Weighted loss
        loss = (config['loss_weight_junction'] * losses['L_junction'] +
                config['loss_weight_offset'] * losses['L_offset'] +
                config['loss_weight_edge'] * losses['L_edge'])

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Accumulate losses
        epoch_losses['total'] += loss.item()
        epoch_losses['junction'] += losses['L_junction'].item()
        epoch_losses['offset'] += losses['L_offset'].item()
        epoch_losses['edge'] += losses['L_edge'].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'L_j': f"{losses['L_junction'].item():.4f}",
            'L_o': f"{losses['L_offset'].item():.4f}",
            'L_e': f"{losses['L_edge'].item():.4f}",
            'L_tot': f"{loss.item():.4f}"
        })

        # Log to tensorboard (every 10 batches)
        if writer and batch_idx % 10 == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
            writer.add_scalar('Train/Loss_Junction', losses['L_junction'].item(), global_step)
            writer.add_scalar('Train/Loss_Offset', losses['L_offset'].item(), global_step)
            writer.add_scalar('Train/Loss_Edge', losses['L_edge'].item(), global_step)

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def validate_epoch(model, dataloader, device, config, writer, epoch):
    """Validate for one epoch."""
    model.eval()

    epoch_losses = {'total': 0, 'junction': 0, 'offset': 0, 'edge': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config["epochs"]} [Valid]')

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)

            # Create edge labels from ground truth
            edge_labels = create_edge_labels(
                batch['junction_map'].to(device),
                batch['edges'],
                device=device
            )

            targets = {
                'junction_map': batch['junction_map'].to(device),
                'offset_map': batch['offset_map'].to(device),
                'offset_mask': batch['offset_mask'].to(device),
                'edge_lists': edge_labels
            }

            # Forward pass
            out = model(images, j_thr=config['junction_thresh'])
            losses = openserge_losses(out, targets)

            # Weighted loss
            loss = (config['loss_weight_junction'] * losses['L_junction'] +
                    config['loss_weight_offset'] * losses['L_offset'] +
                    config['loss_weight_edge'] * losses['L_edge'])

            # Accumulate losses
            epoch_losses['total'] += loss.item()
            epoch_losses['junction'] += losses['L_junction'].item()
            epoch_losses['offset'] += losses['L_offset'].item()
            epoch_losses['edge'] += losses['L_edge'].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'L_j': f"{losses['L_junction'].item():.4f}",
                'L_o': f"{losses['L_offset'].item():.4f}",
                'L_e': f"{losses['L_edge'].item():.4f}",
                'L_tot': f"{loss.item():.4f}"
            })

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    # Log to tensorboard
    if writer:
        writer.add_scalar('Valid/Loss_Total', epoch_losses['total'], epoch)
        writer.add_scalar('Valid/Loss_Junction', epoch_losses['junction'], epoch)
        writer.add_scalar('Valid/Loss_Offset', epoch_losses['offset'], epoch)
        writer.add_scalar('Valid/Loss_Edge', epoch_losses['edge'], epoch)

    return epoch_losses


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
        config['experiment_name'] = f"openserge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading datasets...")
    preload = config.get('preload', False)
    train_dataset = CityScale(config['data_root'], split='train',
                             img_size=config['img_size'], aug=True, preload=preload)
    val_dataset = CityScale(config['data_root'], split='valid',
                           img_size=config['img_size'], aug=False, preload=preload)

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
    model = OpenSERGE(backbone=config['backbone'], k=config.get('k')).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    if config.get('resume'):
        logger.info(f"Resuming from checkpoint: {config['resume']}")
        start_epoch, _, _, val_losses = load_checkpoint(config['resume'], model, optimizer)
        start_epoch += 1
        if val_losses:
            best_val_loss = val_losses['total']

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
        val_losses = validate_epoch(model, val_loader, device,
                                   config, writer, epoch)

        epoch_time = time.time() - epoch_start_time

        # Log epoch summary
        logger.info(
            f"Epoch {epoch}/{config['epochs']} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_losses['total']:.4f} "
            f"(J={train_losses['junction']:.4f}, "
            f"O={train_losses['offset']:.4f}, "
            f"E={train_losses['edge']:.4f}) | "
            f"Val Loss: {val_losses['total']:.4f} "
            f"(J={val_losses['junction']:.4f}, "
            f"O={val_losses['offset']:.4f}, "
            f"E={val_losses['edge']:.4f})"
        )

        # Save history
        train_history['train'].append(train_losses)
        train_history['val'].append(val_losses)

        # Check if best model
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        # Save checkpoint
        if epoch % config['save_freq'] == 0 or is_best:
            save_path = checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, config, save_path,
                          is_best, train_losses, val_losses)
            logger.info(f"Saved checkpoint: {save_path}")

        # Early stopping check
        if early_stopping(val_losses['total']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)

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

    writer.close()


if __name__ == '__main__':
    main()
