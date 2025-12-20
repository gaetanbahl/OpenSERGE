# Weights & Biases Integration

OpenSERGE training includes comprehensive Weights & Biases (W&B) integration for experiment tracking, visualization, and model management.

## Quick Start

### Installation

```bash
pip install wandb
```

### Login

Before running training, authenticate with W&B:

```bash
wandb login
```

You'll be prompted to enter your API key (found at https://wandb.ai/authorize).

### Basic Usage

Run training with W&B enabled (default):

```bash
python -m openserge.train --config configs/default.json
```

Run training with W&B disabled:

```bash
python -m openserge.train --config configs/default.json --disable_wandb
```

## Configuration

### Command-Line Arguments

- `--wandb_project`: W&B project name (default: "openserge")
- `--wandb_entity`: W&B entity/team name (default: None, uses your username)
- `--wandb_run_name`: Custom run name (default: uses experiment_name)
- `--wandb_tags`: Tags for the run (default: None)
- `--disable_wandb`: Disable W&B logging

### Config File

Add W&B settings to your JSON config:

```json
{
  "wandb_project": "openserge",
  "wandb_entity": "my-team",
  "wandb_run_name": "resnet50_k4_baseline",
  "wandb_tags": ["baseline", "resnet50"],
  "disable_wandb": false
}
```

### Example Commands

```bash
# Custom project and run name
python -m openserge.train \
  --config configs/default.json \
  --wandb_project my-road-extraction \
  --wandb_run_name experiment_001

# With tags
python -m openserge.train \
  --config configs/default.json \
  --wandb_tags baseline resnet50 k4

# Team project
python -m openserge.train \
  --config configs/default.json \
  --wandb_entity my-research-team \
  --wandb_project road-graphs
```

## What Gets Logged

### 1. Hyperparameters (Config)

All training hyperparameters are logged at the start:

- Dataset: `data_root`, `img_size`, `preload`
- Model: `backbone`, `k`
- Training: `epochs`, `batch_size`, `lr`, `weight_decay`
- Loss: `junction_thresh`, `loss_weight_*`
- Early stopping: `early_stop_patience`, `min_delta`
- System: `seed`

### 2. Metrics

#### Per-Batch Metrics (logged every 10 batches)

- `train/loss_total`: Total weighted loss
- `train/loss_junction`: Junction detection loss
- `train/loss_offset`: Offset regression loss
- `train/loss_edge`: Edge classification loss
- `train/epoch`: Current epoch

#### Per-Epoch Metrics

- `val/loss_total`: Validation total loss
- `val/loss_junction`: Validation junction loss
- `val/loss_offset`: Validation offset loss
- `val/loss_edge`: Validation edge loss
- `val/best_loss`: Best validation loss so far (logged when improved)
- `train/learning_rate`: Current learning rate
- `epoch`: Epoch number

### 3. Model Monitoring

The model is watched with `wandb.watch()` which logs:

- Gradients (every 100 steps)
- Model parameters (every 100 steps)
- Gradient histograms

### 4. Artifacts

#### Checkpoints

Logged when saved (every `save_freq` epochs or when best):

- Type: `model`
- Name: `checkpoint_epoch{N}`
- Contains: Full checkpoint (model + optimizer state + config)

#### Best Model

Logged whenever a new best model is found:

- Type: `model`
- Name: `best_model`
- Contains: Best checkpoint based on validation loss

#### Final Model

Logged at the end of training:

- Type: `model`
- Name: `final_model`
- Contains: Final checkpoint after all epochs

#### Training History

Logged at the end of training:

- Type: `results`
- Name: `training_history`
- Contains: JSON file with all training and validation losses

### 5. Summary Metrics

Final summary statistics logged at the end:

- `best_val_loss`: Best validation loss achieved
- `final_epoch`: Final epoch number
- `total_params`: Total model parameters
- `trainable_params`: Trainable model parameters

## Using the W&B Dashboard

### Viewing Runs

After starting training, you'll see:

```
W&B project: openserge
W&B run: openserge_baseline_20231201_143022
```

Visit https://wandb.ai/YOUR_USERNAME/openserge to view the dashboard.

### Dashboard Features

1. **Charts**: Real-time loss plots and metrics
2. **System**: GPU/CPU usage, memory, temperature
3. **Logs**: Console output from training
4. **Files**: Config files and training history
5. **Artifacts**: Models and checkpoints
6. **Overview**: Run summary and hyperparameters

### Comparing Experiments

1. Go to your project page
2. Select multiple runs using checkboxes
3. Click "Compare" to see side-by-side metrics
4. Use parallel coordinates plot to explore hyperparameter relationships

### Downloading Models

To download a saved model:

```python
import wandb

# Initialize API
api = wandb.Api()

# Get artifact
artifact = api.artifact('USERNAME/openserge/best_model:latest', type='model')

# Download
artifact_dir = artifact.download()
print(f"Model downloaded to: {artifact_dir}")
```

## Advanced Usage

### Team Collaboration

Share experiments with your team:

```bash
python -m openserge.train \
  --wandb_entity my-team \
  --wandb_project shared-experiments
```

Team members can view and compare all runs.

### Sweep (Hyperparameter Search)

Create a sweep configuration (`sweep.yaml`):

```yaml
program: -m openserge.train
method: bayes
metric:
  name: val/loss_total
  goal: minimize
parameters:
  lr:
    values: [0.0001, 0.001, 0.01]
  k:
    values: [4, 8, 16]
  backbone:
    values: ['resnet18', 'resnet50']
  data_root:
    value: '/path/to/data'
  config:
    value: null
```

Run the sweep:

```bash
# Initialize sweep
wandb sweep sweep.yaml

# Run agent (repeat on multiple machines to parallelize)
wandb agent USERNAME/openserge/SWEEP_ID
```

### Resuming Runs

If training is interrupted, resume the W&B run:

```bash
# Note: This resumes W&B tracking, use --resume for model checkpoint
python -m openserge.train \
  --resume logs/experiment_xyz/checkpoints/checkpoint_epoch20.pt \
  --wandb_run_name same_run_name_as_before
```

### Custom Logging

You can extend the training script with custom W&B logging:

```python
# Log custom metrics
wandb.log({
    'custom/my_metric': value,
    'custom/another_metric': value2
}, step=global_step)

# Log images
wandb.log({
    'predictions': wandb.Image(pred_image),
    'ground_truth': wandb.Image(gt_image)
})

# Log tables
table = wandb.Table(columns=['epoch', 'loss', 'accuracy'])
table.add_data(1, 0.5, 0.95)
wandb.log({'results': table})
```

## Integration with Other Tools

### TensorBoard

The training script logs to both W&B and TensorBoard simultaneously. You can use either or both:

```bash
# View TensorBoard
tensorboard --logdir logs/

# View W&B
# Visit https://wandb.ai
```

### Offline Mode

Run in offline mode (logs stored locally, sync later):

```bash
wandb offline
python -m openserge.train --config configs/default.json

# Sync later
wandb sync logs/EXPERIMENT_NAME/wandb/
```

## Best Practices

1. **Naming**: Use descriptive run names that indicate key hyperparameters
   ```bash
   --wandb_run_name resnet50_k8_lr0.001_bs4
   ```

2. **Tags**: Tag runs for easy filtering
   ```bash
   --wandb_tags baseline highres resnet50
   ```

3. **Projects**: Use separate projects for different research directions
   ```bash
   --wandb_project openserge-ablation
   --wandb_project openserge-production
   ```

4. **Teams**: Use team entities for collaborative work
   ```bash
   --wandb_entity research-lab
   ```

5. **Notes**: Add notes to runs via the web UI to document insights

6. **Artifacts**: Use artifact versioning for model lineage tracking

## Troubleshooting

### Authentication Issues

```bash
# Re-login
wandb login --relogin

# Check login status
wandb status
```

### Slow Logging

W&B uploads data in the background. If it's slow:

```python
# In train.py, reduce logging frequency
wandb.watch(model, log_freq=200)  # Instead of 100
```

### Disk Space

W&B stores artifacts locally before upload. Clean old runs:

```bash
wandb artifact cache cleanup 10GB
```

### API Rate Limits

If hitting rate limits with many runs:

```python
# Increase batch size for logging
wandb.log({...}, step=step, commit=False)  # Don't commit yet
# ... more logs ...
wandb.log({...}, step=step, commit=True)   # Commit batch
```

## Example Workflow

```bash
# 1. Install and login
pip install wandb
wandb login

# 2. Create a config for your experiment
cat > configs/experiment_resnet50_k4.json << EOF
{
  "experiment_name": "resnet50_k4_baseline",
  "data_root": "/path/to/data",
  "backbone": "resnet50",
  "k": 4,
  "epochs": 50,
  "batch_size": 4,
  "lr": 0.001,
  "wandb_project": "openserge",
  "wandb_tags": ["baseline", "resnet50", "k4"]
}
EOF

# 3. Run training
python -m openserge.train --config configs/experiment_resnet50_k4.json

# 4. Monitor at https://wandb.ai/USERNAME/openserge

# 5. Download best model when done
python << EOF
import wandb
api = wandb.Api()
artifact = api.artifact('USERNAME/openserge/best_model:latest')
artifact.download('./best_model')
EOF
```

## Resources

- W&B Documentation: https://docs.wandb.ai/
- W&B Python Library: https://github.com/wandb/wandb
- W&B Examples: https://github.com/wandb/examples
- Community Forum: https://community.wandb.ai/
