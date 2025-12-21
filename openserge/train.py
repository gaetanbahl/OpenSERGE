import time
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from .data.dataset import CityScale
from .models.wrapper import OpenSERGE
from .models.losses import openserge_losses
from .utils.graph import collate_fn, create_edge_labels_from_model
from .utils.training import save_checkpoint, load_checkpoint, setup_logging, load_config, set_seed, save_config, EarlyStopping
from .utils.args import parse_args


def train_epoch(model, dataloader, optimizer, device, config, writer, epoch):
    """Train for one epoch."""
    model.train()

    epoch_losses = {'total': 0, 'junction': 0, 'offset': 0, 'edge': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config["epochs"]} [Train]')

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # Forward pass (run model first to get predicted graph structure)
        out = model(images, j_thr=config['junction_thresh'])

        # Create edge labels aligned with model's predicted graph (OPTIMIZED!)
        # This eliminates expensive alignment in loss computation
        edge_labels = create_edge_labels_from_model(
            out['graphs'],
            batch['edges'],
            out['cnn']['junction_logits'],
            stride=out['cnn']['stride'],
            device=device
        )

        targets = {
            'junction_map': batch['junction_map'].to(device),
            'offset_map': batch['offset_map'].to(device),
            'offset_mask': batch['offset_mask'].to(device),
            'edge_labels': edge_labels  # Pre-aligned labels!
        }

        losses = openserge_losses(out, targets)

        # Weighted loss - only include components with non-zero weights
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
            global_step = (epoch - 1) * len(dataloader) + batch_idx

            # TensorBoard logging
            if writer:
                writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
                writer.add_scalar('Train/Loss_Junction', losses['L_junction'].item(), global_step)
                writer.add_scalar('Train/Loss_Offset', losses['L_offset'].item(), global_step)
                writer.add_scalar('Train/Loss_Edge', losses['L_edge'].item(), global_step)

            # W&B logging
            if not config.get('disable_wandb', False):
                wandb.log({
                    'train/loss_total': loss.item(),
                    'train/loss_junction': losses['L_junction'].item(),
                    'train/loss_offset': losses['L_offset'].item(),
                    'train/loss_edge': losses['L_edge'].item(),
                    'train/epoch': epoch,
                }, step=global_step)

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

            # Forward pass (run model first to get predicted graph structure)
            out = model(images, j_thr=config['junction_thresh'])

            # Create edge labels aligned with model's predicted graph (OPTIMIZED!)
            edge_labels = create_edge_labels_from_model(
                out['graphs'],
                batch['edges'],
                out['cnn']['junction_logits'],
                stride=out['cnn']['stride'],
                device=device
            )

            targets = {
                'junction_map': batch['junction_map'].to(device),
                'offset_map': batch['offset_map'].to(device),
                'offset_mask': batch['offset_mask'].to(device),
                'edge_labels': edge_labels  # Pre-aligned labels!
            }

            losses = openserge_losses(out, targets)

            # Weighted loss - only include components with non-zero weights
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

    # Log to W&B
    if not config.get('disable_wandb', False):
        wandb.log({
            'val/loss_total': epoch_losses['total'],
            'val/loss_junction': epoch_losses['junction'],
            'val/loss_offset': epoch_losses['offset'],
            'val/loss_edge': epoch_losses['edge'],
            'epoch': epoch,
        })  # No step parameter - let wandb auto-increment

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

    # Setup Weights & Biases
    if not config.get('disable_wandb', False):
        wandb_config = {
            'data_root': config.get('data_root', 'N/A'),
            'img_size': config['img_size'],
            'backbone': config.get('backbone', 'resnet50'),
            'k': config.get('k', 'None'),
            'pretrained_cnn': config.get('pretrained_cnn', 'None'),
            'freeze_pretrained_cnn': config.get('freeze_pretrained_cnn', False),
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'lr': config['lr'],
            'weight_decay': config.get('weight_decay', 1e-4),
            'junction_thresh': config.get('junction_thresh', 0.5),
            'loss_weight_junction': config.get('loss_weight_junction', 1.0),
            'loss_weight_offset': config.get('loss_weight_offset', 1.0),
            'loss_weight_edge': config.get('loss_weight_edge', 1.0),
            'early_stop_patience': config.get('early_stop_patience', 10),
            'min_delta': config.get('min_delta', 1e-4),
            'seed': config['seed'],
            'preload': config.get('preload', False),
        }

        wandb.init(
            project=config.get('wandb_project', 'openserge'),
            entity=config.get('wandb_entity'),
            name=config.get('wandb_run_name', config['experiment_name']),
            tags=config.get('wandb_tags'),
            config=wandb_config,
            dir=str(exp_dir)
        )

        # Watch model gradients and parameters
        # wandb.watch(model) will be called after model is created

        logger.info(f"W&B project: {config.get('wandb_project', 'openserge')}")
        logger.info(f"W&B run: {wandb.run.name if wandb.run else 'N/A'}")
    else:
        logger.info("W&B logging disabled")

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

    # Load pre-trained CNN weights if specified
    if config.get('pretrained_cnn'):
        logger.info(f"Loading pre-trained CNN from: {config['pretrained_cnn']}")
        pretrained_checkpoint = torch.load(config['pretrained_cnn'], map_location=device)

        # Extract SingleShotRoadGraphNet state dict
        if 'model_state_dict' in pretrained_checkpoint:
            # It's a full training checkpoint
            full_state_dict = pretrained_checkpoint['model_state_dict']
            # Filter keys that belong to SingleShotRoadGraphNet (keys starting with 'ss.')
            cnn_state_dict = {k[3:]: v for k, v in full_state_dict.items() if k.startswith('ss.')}

            if len(cnn_state_dict) == 0:
                logger.warning("No 'ss.' prefix found in checkpoint. Attempting direct load...")
                cnn_state_dict = full_state_dict
        else:
            # It's a direct state dict
            cnn_state_dict = pretrained_checkpoint

        # Load into the CNN component
        missing_keys, unexpected_keys = model.ss.load_state_dict(cnn_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in pre-trained CNN: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in pre-trained CNN: {unexpected_keys}")

        logger.info("Pre-trained CNN weights loaded successfully!")

        # Optionally freeze CNN weights
        if config.get('freeze_pretrained_cnn', False):
            logger.info("Freezing pre-trained CNN weights...")
            for param in model.ss.parameters():
                param.requires_grad = False
            logger.info("CNN weights frozen. Only GNN will be trained.")

            # When CNN is frozen, junction and offset losses don't contribute gradients
            # Automatically set their weights to 0 to avoid gradient computation issues
            if config.get('loss_weight_junction', 1.0) > 0 or config.get('loss_weight_offset', 1.0) > 0:
                logger.warning("CNN is frozen but junction/offset loss weights are > 0.")
                logger.warning("Setting junction_weight=0 and offset_weight=0 to avoid gradient errors.")
                logger.warning("Only edge loss will be used for training the GNN.")
                config['loss_weight_junction'] = 0.0
                config['loss_weight_offset'] = 0.0

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Verify we have trainable parameters
    if num_trainable == 0:
        raise ValueError("No trainable parameters! Check your freeze settings.")

    # Watch model with W&B
    if not config.get('disable_wandb', False):
        wandb.watch(model, log='all', log_freq=100)

    # Create optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    logger.info(f"Optimizer created with {len(trainable_params)} parameter groups")

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
