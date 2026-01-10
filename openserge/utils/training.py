from datetime import datetime
import json
import logging
from pathlib import Path
import torch


def save_checkpoint(model, optimizer, epoch, config, save_path, is_best=False,
                   train_losses=None, val_losses=None, best_path_suffix='best_model.pt'):
    """Save model checkpoint.

    Args:
        best_path_suffix: Suffix for best model file (default: 'best_model.pt').
                         Can be stage-specific like 'stage1_best.pt'
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.parent / best_path_suffix
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # if optimizer is not None:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint.get('config', {}), \
           checkpoint.get('train_losses'), checkpoint.get('val_losses')

def setup_logging(log_dir):
    """Setup logging to both file and console."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config, save_path):
    """Save configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
