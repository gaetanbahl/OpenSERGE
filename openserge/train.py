"""
Three-stage training script for OpenSERGE.

Training pipeline:
  Stage 1: Junction detection (CNN only) - trains backbone for junction/offset prediction
  Stage 2: GNN training (frozen CNN) - trains GNN for edge prediction with fixed CNN
  Stage 3: Full model fine-tuning - end-to-end training with reduced learning rate
"""
import time
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from .data.dataset import CityScale, GlobalScale
from .models.wrapper import OpenSERGE
from .models.losses import openserge_losses
from .utils.graph import collate_fn, create_edge_labels_from_model
from .utils.training import save_checkpoint, setup_logging, load_config, set_seed, save_config, EarlyStopping
from .utils.args import parse_args


def train_epoch(model, dataloader, optimizer, device, config, writer, epoch, stage_num, global_step_offset=0):
    """Train for one epoch."""
    model.train()

    epoch_losses = {'total': 0, 'junction': 0, 'offset': 0, 'edge': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Stage {stage_num} - Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Transfer all data to GPU once
        images = batch['image'].to(device)
        junction_map = batch['junction_map'].to(device)
        offset_map = batch['offset_map'].to(device)
        offset_mask = batch['offset_mask'].to(device)

        # Forward pass
        out = model(images, j_thr=config['junction_thresh'])

        # Create edge labels aligned with model's predicted graph
        edge_labels = create_edge_labels_from_model(
            out['graphs'],
            batch['edges'],
            out['cnn']['junction_logits'],
            stride=out['cnn']['stride'],
            device=device
        )

        targets = {
            'junction_map': junction_map,
            'offset_map': offset_map,
            'offset_mask': offset_mask,
            'edge_labels': edge_labels
        }

        losses = openserge_losses(out, targets)

        # Weighted loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if config['loss_weight_junction'] > 0:
            loss = loss + config['loss_weight_junction'] * losses['L_junction']
        if config['loss_weight_offset'] > 0:
            loss = loss + config['loss_weight_offset'] * losses['L_offset']
        if config['loss_weight_edge'] > 0:
            loss = loss + config['loss_weight_edge'] * losses['L_edge']

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

        # Log to tensorboard and W&B (every 10 batches)
        if batch_idx % 10 == 0:
            global_step = global_step_offset + (epoch - 1) * len(dataloader) + batch_idx

            if writer:
                writer.add_scalar(f'Stage{stage_num}/Train/Loss_Total', loss.item(), global_step)
                writer.add_scalar(f'Stage{stage_num}/Train/Loss_Junction', losses['L_junction'].item(), global_step)
                writer.add_scalar(f'Stage{stage_num}/Train/Loss_Offset', losses['L_offset'].item(), global_step)
                writer.add_scalar(f'Stage{stage_num}/Train/Loss_Edge', losses['L_edge'].item(), global_step)

            if not config.get('disable_wandb', False):
                wandb.log({
                    f'stage{stage_num}/train/loss_total': loss.item(),
                    f'stage{stage_num}/train/loss_junction': losses['L_junction'].item(),
                    f'stage{stage_num}/train/loss_offset': losses['L_offset'].item(),
                    f'stage{stage_num}/train/loss_edge': losses['L_edge'].item(),
                }, step=global_step)

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def validate_epoch(model, dataloader, device, config, writer, epoch, stage_num):
    """Validate for one epoch."""
    model.eval()

    epoch_losses = {'total': 0, 'junction': 0, 'offset': 0, 'edge': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Stage {stage_num} - Epoch {epoch} [Valid]')

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            junction_map = batch['junction_map'].to(device)
            offset_map = batch['offset_map'].to(device)
            offset_mask = batch['offset_mask'].to(device)

            out = model(images, j_thr=config['junction_thresh'])

            edge_labels = create_edge_labels_from_model(
                out['graphs'],
                batch['edges'],
                out['cnn']['junction_logits'],
                stride=out['cnn']['stride'],
                device=device
            )

            targets = {
                'junction_map': junction_map,
                'offset_map': offset_map,
                'offset_mask': offset_mask,
                'edge_labels': edge_labels
            }

            losses = openserge_losses(out, targets)

            # Weighted loss
            loss = torch.tensor(0.0, device=device, requires_grad=False)
            if config['loss_weight_junction'] > 0:
                loss = loss + config['loss_weight_junction'] * losses['L_junction']
            if config['loss_weight_offset'] > 0:
                loss = loss + config['loss_weight_offset'] * losses['L_offset']
            if config['loss_weight_edge'] > 0:
                loss = loss + config['loss_weight_edge'] * losses['L_edge']

            # Accumulate losses
            epoch_losses['total'] += loss.item()
            epoch_losses['junction'] += losses['L_junction'].item()
            epoch_losses['offset'] += losses['L_offset'].item()
            epoch_losses['edge'] += losses['L_edge'].item()
            num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    # Log to tensorboard
    if writer:
        writer.add_scalar(f'Stage{stage_num}/Valid/Loss_Total', epoch_losses['total'], epoch)
        writer.add_scalar(f'Stage{stage_num}/Valid/Loss_Junction', epoch_losses['junction'], epoch)
        writer.add_scalar(f'Stage{stage_num}/Valid/Loss_Offset', epoch_losses['offset'], epoch)
        writer.add_scalar(f'Stage{stage_num}/Valid/Loss_Edge', epoch_losses['edge'], epoch)

    # Log to W&B
    if not config.get('disable_wandb', False):
        wandb.log({
            f'stage{stage_num}/val/loss_total': epoch_losses['total'],
            f'stage{stage_num}/val/loss_junction': epoch_losses['junction'],
            f'stage{stage_num}/val/loss_offset': epoch_losses['offset'],
            f'stage{stage_num}/val/loss_edge': epoch_losses['edge'],
        })

    return epoch_losses


def train_stage(stage_num, model, train_loader, val_loader, optimizer, device, config,
                writer, logger, checkpoint_dir, max_epochs, patience, global_step_offset=0):
    """Train a single stage of the multi-stage training pipeline."""
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=config.get('min_delta', 1e-4),
        mode='min'
    )

    best_val_loss = float('inf')
    train_history = {'train': [], 'val': []}

    logger.info(f"=" * 70)
    logger.info(f"STAGE {stage_num} TRAINING")
    logger.info(f"Max epochs: {max_epochs}, Patience: {patience}")
    logger.info(f"=" * 70)

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device,
                                   config, writer, epoch, stage_num, global_step_offset)

        # Validate
        val_losses = validate_epoch(model, val_loader, device,
                                   config, writer, epoch, stage_num)

        epoch_time = time.time() - epoch_start_time

        # Log epoch summary
        logger.info(
            f"Stage {stage_num} - Epoch {epoch}/{max_epochs} ({epoch_time:.1f}s) - "
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
            logger.info(f"Stage {stage_num} - New best validation loss: {best_val_loss:.4f}")

            if not config.get('disable_wandb', False):
                wandb.log({f'stage{stage_num}/best_val_loss': best_val_loss})

        # Save checkpoint
        if epoch % config.get('save_freq', 5) == 0 or is_best:
            save_path = checkpoint_dir / f'stage{stage_num}_epoch{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, config, save_path,
                          is_best, train_losses, val_losses)
            logger.info(f"Saved checkpoint: {save_path}")

            if not config.get('disable_wandb', False) and is_best:
                best_artifact = wandb.Artifact(
                    name=f'stage{stage_num}_best_model',
                    type='model',
                    description=f"Best model for stage {stage_num} (val_loss={best_val_loss:.4f})"
                )
                best_model_path = checkpoint_dir / f'stage{stage_num}_best.pt'
                if best_model_path.exists():
                    best_artifact.add_file(str(best_model_path))
                    wandb.log_artifact(best_artifact)

        # Early stopping check
        if early_stopping(val_losses['total']):
            logger.info(f"Stage {stage_num} - Early stopping triggered at epoch {epoch}")
            break

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar(f'Stage{stage_num}/LearningRate', current_lr, epoch)
        if not config.get('disable_wandb', False):
            wandb.log({f'stage{stage_num}/learning_rate': current_lr})

    logger.info(f"Stage {stage_num} completed! Best validation loss: {best_val_loss:.4f}")

    # Save stage training history
    history_path = checkpoint_dir.parent / f'stage{stage_num}_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)

    # Calculate final global step for this stage
    final_global_step = global_step_offset + epoch * len(train_loader)

    return best_val_loss, final_global_step


def main():
    """Main training function - always uses three-stage training."""
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

    logger.info("="*70)
    logger.info("THREE-STAGE TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Stage 1 (Junction): max {config.get('stage1_epochs', 1000)} epochs, patience {config.get('stage1_patience', 100)}")
    logger.info(f"Stage 2 (GNN): max {config.get('stage2_epochs', 1000)} epochs, patience {config.get('stage2_patience', 100)}")
    logger.info(f"Stage 3 (Full): max {config.get('stage3_epochs', 2000)} epochs, patience {config.get('stage3_patience', 200)}, LR factor {config.get('stage3_lr_factor', 0.3)}")
    logger.info("="*70)

    # Setup tensorboard
    writer = SummaryWriter(exp_dir / 'tensorboard')

    # Setup Weights & Biases with safe config extraction
    if not config.get('disable_wandb', False):
        wandb_config = {
            'dataset': config.get('dataset', 'cityscale'),
            'data_root': config.get('data_root', 'N/A'),
            'img_size': config.get('img_size', 512),
            'backbone': config.get('backbone', 'resnet50'),
            'k': config.get('k'),
            'batch_size': config.get('batch_size', 8),
            'lr': config.get('lr', 0.001),
            'weight_decay': config.get('weight_decay', 0.0001),
            'junction_thresh': config.get('junction_thresh', 0.5),
            'loss_weight_junction': config.get('loss_weight_junction', 1.0),
            'loss_weight_offset': config.get('loss_weight_offset', 10.0),
            'loss_weight_edge': config.get('loss_weight_edge', 1.0),
            'min_delta': config.get('min_delta', 0.0001),
            'seed': config.get('seed', 42),
            'preload': config.get('preload', False),
            'stage1_epochs': config.get('stage1_epochs', 1000),
            'stage1_patience': config.get('stage1_patience', 100),
            'stage2_epochs': config.get('stage2_epochs', 1000),
            'stage2_patience': config.get('stage2_patience', 100),
            'stage3_epochs': config.get('stage3_epochs', 2000),
            'stage3_patience': config.get('stage3_patience', 200),
            'stage3_lr_factor': config.get('stage3_lr_factor', 0.3),
        }

        wandb.init(
            project=config.get('wandb_project', 'openserge'),
            entity=config.get('wandb_entity'),
            name=config.get('wandb_run_name', config['experiment_name']),
            tags=config.get('wandb_tags'),
            config=wandb_config,
            dir=str(exp_dir)
        )

        logger.info(f"W&B project: {config.get('wandb_project', 'openserge')}")
        logger.info(f"W&B run: {wandb.run.name if wandb.run else 'N/A'}")
    else:
        logger.info("W&B logging disabled")

    # Setup device
    if config.get('device') == "cuda":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif config.get('device') == "mps":
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Dataset selection
    dataset_type = config.get('dataset', 'cityscale')
    preload = config.get('preload', False)
    DatasetClass = GlobalScale if dataset_type == 'globalscale' else CityScale

    # Create model
    logger.info("Creating model...")
    model = OpenSERGE(backbone=config.get('backbone', 'resnet50'), k=config.get('k')).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    if not config.get('disable_wandb', False):
        wandb.watch(model, log='all', log_freq=100)

    # =========================================================================
    # STAGE 1: Junction Detection (CNN only)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: Training Junction Detection (CNN only)")
    logger.info("="*70)

    # Create datasets for stage 1 (skip edges for faster training)
    logger.info("Loading datasets (skip_edges=True)...")
    train_dataset_s1 = DatasetClass(config['data_root'], split='train',
                                    img_size=config.get('img_size', 512), aug=True,
                                    preload=preload, skip_edges=True)
    val_dataset_s1 = DatasetClass(config['data_root'], split='valid',
                                  img_size=config.get('img_size', 512), aug=False,
                                  preload=preload, skip_edges=True)

    train_loader_s1 = DataLoader(
        train_dataset_s1, batch_size=config.get('batch_size', 8), shuffle=True,
        num_workers=config.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False,
        prefetch_factor=2 if config.get('num_workers', 4) > 0 else None
    )
    val_loader_s1 = DataLoader(
        val_dataset_s1, batch_size=config.get('batch_size', 8), shuffle=False,
        num_workers=config.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False,
        prefetch_factor=2 if config.get('num_workers', 4) > 0 else None
    )

    logger.info(f"Train samples: {len(train_dataset_s1)}")
    logger.info(f"Valid samples: {len(val_dataset_s1)}")

    # Configure loss weights for stage 1 (junction + offset only)
    stage1_config = config.copy()
    stage1_config['loss_weight_junction'] = config.get('loss_weight_junction', 1.0)
    stage1_config['loss_weight_offset'] = config.get('loss_weight_offset', 10.0)
    stage1_config['loss_weight_edge'] = 0.0

    optimizer_s1 = torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0001)
    )

    best_val_loss_s1, global_step_s1 = train_stage(
        stage_num=1,
        model=model,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        optimizer=optimizer_s1,
        device=device,
        config=stage1_config,
        writer=writer,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        max_epochs=config.get('stage1_epochs', 1000),
        patience=config.get('stage1_patience', 100),
        global_step_offset=0
    )

    # =========================================================================
    # STAGE 2: GNN Training (CNN frozen)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: Training GNN (CNN frozen)")
    logger.info("="*70)

    # Freeze CNN weights
    logger.info("Freezing CNN weights...")
    for param in model.ss.parameters():
        param.requires_grad = False
    num_trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters in stage 2: {num_trainable_s2:,}")

    # Create datasets for stage 2 (with edges)
    logger.info("Loading datasets (with edges)...")
    train_dataset_s2 = DatasetClass(config['data_root'], split='train',
                                    img_size=config.get('img_size', 512), aug=True,
                                    preload=preload, skip_edges=False)
    val_dataset_s2 = DatasetClass(config['data_root'], split='valid',
                                  img_size=config.get('img_size', 512), aug=False,
                                  preload=preload, skip_edges=False)

    train_loader_s2 = DataLoader(
        train_dataset_s2, batch_size=config.get('batch_size', 8), shuffle=True,
        num_workers=config.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False,
        prefetch_factor=2 if config.get('num_workers', 4) > 0 else None
    )
    val_loader_s2 = DataLoader(
        val_dataset_s2, batch_size=config.get('batch_size', 8), shuffle=False,
        num_workers=config.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False,
        prefetch_factor=2 if config.get('num_workers', 4) > 0 else None
    )

    # Configure loss weights for stage 2 (edge only)
    stage2_config = config.copy()
    stage2_config['loss_weight_junction'] = 0.0
    stage2_config['loss_weight_offset'] = 0.0
    stage2_config['loss_weight_edge'] = config.get('loss_weight_edge', 1.0)

    trainable_params_s2 = [p for p in model.parameters() if p.requires_grad]
    optimizer_s2 = torch.optim.Adam(
        trainable_params_s2,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0001)
    )

    best_val_loss_s2, global_step_s2 = train_stage(
        stage_num=2,
        model=model,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        optimizer=optimizer_s2,
        device=device,
        config=stage2_config,
        writer=writer,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        max_epochs=config.get('stage2_epochs', 1000),
        patience=config.get('stage2_patience', 100),
        global_step_offset=global_step_s1
    )

    # =========================================================================
    # STAGE 3: Full Model Fine-tuning (reduced LR)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 3: Fine-tuning Full Model (all parameters)")
    logger.info("="*70)

    # Unfreeze all parameters
    logger.info("Unfreezing all parameters...")
    for param in model.parameters():
        param.requires_grad = True
    num_trainable_s3 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters in stage 3: {num_trainable_s3:,}")

    # Configure loss weights for stage 3 (all losses)
    stage3_config = config.copy()
    stage3_config['loss_weight_junction'] = config.get('loss_weight_junction', 1.0)
    stage3_config['loss_weight_offset'] = config.get('loss_weight_offset', 10.0)
    stage3_config['loss_weight_edge'] = config.get('loss_weight_edge', 1.0)

    # Reduced learning rate for stage 3
    stage3_lr = config.get('lr', 0.001) * config.get('stage3_lr_factor', 0.3)
    logger.info(f"Using reduced learning rate: {stage3_lr} (factor: {config.get('stage3_lr_factor', 0.3)})")

    optimizer_s3 = torch.optim.Adam(
        model.parameters(),
        lr=stage3_lr,
        weight_decay=config.get('weight_decay', 0.0001)
    )

    best_val_loss_s3, global_step_s3 = train_stage(
        stage_num=3,
        model=model,
        train_loader=train_loader_s2,  # Reuse stage 2 loaders
        val_loader=val_loader_s2,
        optimizer=optimizer_s3,
        device=device,
        config=stage3_config,
        writer=writer,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        max_epochs=config.get('stage3_epochs', 2000),
        patience=config.get('stage3_patience', 200),
        global_step_offset=global_step_s2
    )

    # =========================================================================
    # Training Complete
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("THREE-STAGE TRAINING COMPLETED!")
    logger.info("="*70)
    logger.info(f"Stage 1 (Junction): Best val loss = {best_val_loss_s1:.4f}")
    logger.info(f"Stage 2 (GNN): Best val loss = {best_val_loss_s2:.4f}")
    logger.info(f"Stage 3 (Full): Best val loss = {best_val_loss_s3:.4f}")
    logger.info("="*70)

    # Save final checkpoint (copy of stage 3 best)
    final_path = checkpoint_dir / 'final_model.pt'
    best_stage3_path = checkpoint_dir / 'stage3_best.pt'
    if best_stage3_path.exists():
        logger.info(f"Copying best stage 3 model to final_model.pt...")
        shutil.copy(best_stage3_path, final_path)

    logger.info(f"Training completed! Best validation loss: {best_val_loss_s3:.4f}")
    logger.info(f"Results saved to: {exp_dir}")

    # Log final artifacts to W&B
    if not config.get('disable_wandb', False):
        # Log final model
        if final_path.exists():
            final_artifact = wandb.Artifact(
                name='final_model',
                type='model',
                description=f"Final model (stage 3 best, val_loss={best_val_loss_s3:.4f})"
            )
            final_artifact.add_file(str(final_path))
            wandb.log_artifact(final_artifact)

        # Log final summary metrics
        wandb.summary['stage1_best_val_loss'] = best_val_loss_s1
        wandb.summary['stage2_best_val_loss'] = best_val_loss_s2
        wandb.summary['stage3_best_val_loss'] = best_val_loss_s3
        wandb.summary['final_val_loss'] = best_val_loss_s3
        wandb.summary['total_params'] = num_params
        wandb.summary['trainable_params'] = num_trainable

        wandb.finish()
        logger.info("W&B run finished successfully")

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
