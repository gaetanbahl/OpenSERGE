"""
Training script for junction detection (backbone + junction/offset head only).
This script trains a simplified version of OpenSERGE focusing only on the
junction detection and offset regression tasks, without the GNN edge prediction.
"""
import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import CityScale
from .models.net import SingleShotRoadGraphNet
from .utils import sigmoid_focal_loss, masked_mse


def parse_args():
    ap = argparse.ArgumentParser(description='Train junction detection on CityScale dataset')

    # Data
    ap.add_argument('--data_root', type=str, required=True,
                   help='Path to Sat2Graph/data/data directory')
    ap.add_argument('--img_size', type=int, default=512,
                   help='Input image size (default: 512)')
    ap.add_argument('--stride', type=int, default=32,
                   help='Downsampling stride (default: 32)')

    # Model
    ap.add_argument('--backbone', type=str, default='resnet50',
                   choices=['resnet18', 'resnet50'],
                   help='Backbone architecture (default: resnet50)')
    ap.add_argument('--nfeat', type=int, default=256,
                   help='Number of feature channels (default: 256)')
    ap.add_argument('--pretrained', action='store_true',
                   help='Use pretrained backbone weights')

    # Training
    ap.add_argument('--epochs', type=int, default=50,
                   help='Number of training epochs (default: 50)')
    ap.add_argument('--batch_size', type=int, default=4,
                   help='Batch size (default: 4)')
    ap.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate (default: 1e-3)')
    ap.add_argument('--weight_decay', type=float, default=1e-4,
                   help='Weight decay (default: 1e-4)')
    ap.add_argument('--num_workers', type=int, default=4,
                   help='Number of data loading workers (default: 4)')

    # Loss weights
    ap.add_argument('--junction_weight', type=float, default=1.0,
                   help='Weight for junction loss (default: 1.0)')
    ap.add_argument('--offset_weight', type=float, default=1.0,
                   help='Weight for offset loss (default: 1.0)')
    ap.add_argument('--focal_alpha', type=float, default=0.25,
                   help='Focal loss alpha parameter (default: 0.25)')
    ap.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal loss gamma parameter (default: 2.0)')

    # Checkpointing
    ap.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                   help='Directory to save checkpoints (default: checkpoints)')
    ap.add_argument('--save_freq', type=int, default=5,
                   help='Save checkpoint every N epochs (default: 5)')
    ap.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')

    # Device
    ap.add_argument('--device', type=str, default='cuda',
                   help='Device to use (default: cuda)')

    return ap.parse_args()


def compute_losses(outputs, targets, args):
    """
    Compute junction and offset losses.

    Args:
        outputs: Dict with 'junction_logits' [B,1,h,w] and 'offset' [B,2,h,w]
        targets: Dict with 'junction_map', 'offset_map', 'offset_mask'
        args: Argument namespace with loss hyperparameters

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
        alpha=args.focal_alpha,
        gamma=args.focal_gamma
    )

    # Offset masked MSE loss
    loss_offset = masked_mse(offset_pred, offset_tgt, offset_mask)

    # Total weighted loss
    total_loss = (args.junction_weight * loss_junction +
                  args.offset_weight * loss_offset)

    return {
        'total': total_loss,
        'junction': loss_junction,
        'offset': loss_offset
    }


def train_epoch(model, dataloader, optimizer, device, args, epoch):
    """Train for one epoch."""
    model.train()

    total_losses = {'total': 0.0, 'junction': 0.0, 'offset': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
    for batch in pbar:
        images = batch['image'].to(device)
        targets = {
            'junction_map': batch['junction_map'],
            'offset_map': batch['offset_map'],
            'offset_mask': batch['offset_mask']
        }

        # Forward pass
        outputs = model(images)
        losses = compute_losses(outputs, targets, args)

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Accumulate losses
        for k in total_losses.keys():
            total_losses[k] += losses[k].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'j': f"{losses['junction'].item():.4f}",
            'o': f"{losses['offset'].item():.4f}"
        })

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, device, args):
    """Validate the model."""
    model.eval()

    total_losses = {'total': 0.0, 'junction': 0.0, 'offset': 0.0}
    num_batches = 0

    # Metrics
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        images = batch['image'].to(device)
        targets = {
            'junction_map': batch['junction_map'],
            'offset_map': batch['offset_map'],
            'offset_mask': batch['offset_mask']
        }

        # Forward pass
        outputs = model(images)
        losses = compute_losses(outputs, targets, args)

        # Accumulate losses
        for k in total_losses.keys():
            total_losses[k] += losses[k].item()
        num_batches += 1

        # Compute metrics (using threshold of 0.5 after sigmoid)
        j_pred = torch.sigmoid(outputs['junction_logits']) > 0.5
        j_gt = targets['junction_map'].to(device) > 0.5

        total_tp += (j_pred & j_gt).sum().item()
        total_fp += (j_pred & ~j_gt).sum().item()
        total_fn += (~j_pred & j_gt).sum().item()

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

    # Compute F1 score
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return avg_losses, metrics


def save_checkpoint(model, optimizer, epoch, args, metrics, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """Load model checkpoint."""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resumed from checkpoint: {filename} (epoch {epoch})")
    return epoch


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create datasets
    print("Loading datasets...")
    train_dataset = CityScale(
        args.data_root,
        split='train',
        img_size=args.img_size,
        stride=args.stride,
        aug=True
    )
    val_dataset = CityScale(
        args.data_root,
        split='valid',
        img_size=args.img_size,
        stride=args.stride,
        aug=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print(f"Creating model with {args.backbone} backbone...")
    model = SingleShotRoadGraphNet(
        backbone=args.backbone,
        nfeat=args.nfeat
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device) + 1

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }

    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, args, epoch)
        print(f"Train - Loss: {train_losses['total']:.4f}, "
              f"Junction: {train_losses['junction']:.4f}, "
              f"Offset: {train_losses['offset']:.4f}")

        # Validate
        val_losses, val_metrics = validate(model, val_loader, device, args)
        print(f"Val   - Loss: {val_losses['total']:.4f}, "
              f"Junction: {val_losses['junction']:.4f}, "
              f"Offset: {val_losses['offset']:.4f}")
        print(f"Val   - F1: {val_metrics['f1']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")

        # Update history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])

        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'junction_epoch{epoch}.pt'
            )
            save_checkpoint(model, optimizer, epoch, args, val_metrics, checkpoint_path)

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_path = os.path.join(args.checkpoint_dir, 'junction_best.pt')
            save_checkpoint(model, optimizer, epoch, args, val_metrics, best_path)
            print(f"New best F1: {best_f1:.4f}")

    # Save final checkpoint
    final_path = os.path.join(args.checkpoint_dir, 'junction_final.pt')
    save_checkpoint(model, optimizer, args.epochs, args, val_metrics, final_path)

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\nTraining completed!")
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
