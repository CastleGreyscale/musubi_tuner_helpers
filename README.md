# LoRA Training System for Qwen Image

A unified, streamlined LoRA training pipeline for AI image generation with Qwen Image models. This system provides a single-configuration workflow for training aesthetic and mood-based LoRAs that capture specific cinematic styles and visual atmospheres.

## Overview

This training system simplifies LoRA development by consolidating all configuration into a single `project.toml` file. It handles the complete pipeline from dataset curation through training, with support for text encoder caching, VAE caching, and advanced training features.

### Key Features

- **Unified Configuration**: Single `project.toml` file controls everything
- **Multiple Training Modes**: Standard, Edit, and Edit Plus modes
- **Automated Pipeline**: Cache text encoders, VAE latents, and train with one command each
- **Advanced Training Options**: Style learning, sample generation, checkpoint resume
- **Memory Optimizations**: FP8 modes, block swapping, optional pinned memory
- **Project-Based Organization**: Each LoRA in its own directory with self-contained config

## Architecture

The system treats LoRA training as adding a specialized "lens" to enhance the base model with specific stylistic knowledge, rather than rebuilding fundamental capabilities. This approach excels at teaching:

- Lighting and color palettes
- Visual atmospheres and moods
- Cinematic styles and eras
- Aesthetic characteristics

The base model retains its understanding of objects, composition, and anatomy while the LoRA adds the stylistic layer.

## Installation

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU
- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) repository
- Ollama (for dataset tagging with LLaVA)

### Dependencies

```bash
pip install tomli tomli_w  # Python 3.10
# or use built-in tomllib on Python 3.11+
```

### Environment Variables

Set these in your shell configuration (e.g., `~/.config/fish/config.fish`):

```fish
set -gx COMFYUI_MODELS_ROOT "/path/to/ComfyUI/models"
set -gx LORA_PROJECTS_ROOT "/path/to/lora_projects"
set -gx MUSUBI_TUNER_ROOT "/path/to/musubi-tuner"
```

### Setup

1. Clone this repository or copy the scripts to a location in your PATH
2. Make scripts executable:
   ```bash
   chmod +x cache_text_encoder cache_vae train_lora
   ```
3. Ensure `lora_config_loader.py` is in the same directory or in your PYTHONPATH

## Project Structure

```
lora_projects/
├── noir/                          # Example project
│   ├── project.toml              # All configuration here
│   ├── dataset/                  # Your training images
│   │   ├── image001.jpg
│   │   ├── image001.txt          # Captions
│   │   └── ...
│   ├── cache_directory/          # Auto-generated caches
│   └── noir.toml                 # Auto-generated dataset config
└── christmas/                     # Another project
    ├── project.toml
    ├── dataset/
    └── cache_directory/
```

## Quick Start

### 1. Create a New Project

```bash
cd ~/lora_projects
mkdir my_lora
cd my_lora
cp /path/to/template/project.toml .
```

### 2. Configure Your Project

Edit `project.toml`:

```toml
[project]
name = "my_lora"
description = "My aesthetic LoRA"

[paths]
models_root = "${COMFYUI_MODELS_ROOT}"
projects_root = "${LORA_PROJECTS_ROOT}"
musubi_tuner_root = "${MUSUBI_TUNER_ROOT}"
dataset_dir = "./dataset"
cache_dir = "./cache_directory"

[training]
training_mode = "standard"  # or "edit", "edit_plus"
learning_rate = "5e-5"
network_dim = 32
max_epochs = 16
save_every_n_epochs = 2

[training.models]
dit_model = "qwen_image_bf16.safetensors"
vae_model = "diffusion_pytorch_model.safetensors"
text_encoder = "qwen_2.5_vl_7b.safetensors"
```

### 3. Prepare Your Dataset

Place images in `./dataset/` and caption them (manually or using the tagging script).

### 4. Run the Pipeline

```bash
# From your project directory
cache_text_encoder  # Cache text embeddings
cache_vae          # Cache VAE latents
train_lora         # Train overnight
```

## Training Modes

### Standard Mode (Default)

Regular Qwen Image training for aesthetic and mood LoRAs.

```toml
[training]
training_mode = "standard"

[training.models]
dit_model = "qwen_image_bf16.safetensors"
```

### Edit Mode

For training with Qwen Image Edit model.

```toml
[training]
training_mode = "edit"

[training.models]
dit_model_edit = "qwen_image_edit_bf16.safetensors"
```

### Edit Plus Mode

For training with Qwen Image Edit Plus model.

```toml
[training]
training_mode = "edit_plus"

[training.models]
dit_model_edit_plus = "qwen_image_edit_plus_bf16.safetensors"
```

## Configuration Reference

### Basic Training Parameters

```toml
[training]
training_mode = "standard"              # "standard", "edit", or "edit_plus"
learning_rate = "5e-5"                  # Learning rate
network_dim = 32                        # LoRA rank (16, 32, 64)
max_epochs = 16                         # Training epochs
save_every_n_epochs = 2                 # Checkpoint frequency
seed = 42                               # Random seed

# Memory settings
blocks_to_swap = 16                     # Lower = faster, higher = less VRAM
fp8_mode = "fp8_base"                   # "fp8_base" or "fp8_scaled"
use_pinned_memory_for_block_swap = false  # May improve performance (test on Windows)

# Advanced settings
optimizer = "adamw8bit"
mixed_precision = "bf16"
max_workers = 2
discrete_flow_shift = 2.2
```

### Dataset Configuration

```toml
[dataset]
resolution = 1024                       # Training resolution
batch_size = 1                          # Images per batch
num_repeats = 1                         # Dataset repetitions per epoch
shuffle_caption = false
caption_extension = ".txt"
caption_dropout_rate = 0.0

# Bucketing for mixed aspect ratios
enable_bucket = false
bucket_no_upscale = false

# Augmentation
color_aug = false
random_crop = false
flip_aug = false
```

### Advanced Training Options

```toml
[training.advanced]
# Checkpoint Resume (currently not working - see Known Issues)
resume_from_checkpoint = ""             # e.g., "my_lora-000008"
save_state = true
save_state_on_train_end = false

# Style Learning - Sample Generation
enable_sample_prompts = true
sample_every_n_epochs = 2
sample_at_first = true
sample_prompts = [
    "noir_style detective in rain",
    "noir_style city street at night"
]

# Learning Rate Scheduling
lr_scheduler = ""                       # "constant", "cosine", "cosine_with_restarts"
lr_warmup_steps = 0

# Style Enhancement
min_snr_gamma = 0                       # 5-20 helps style learning
noise_offset = 0                        # 0.0-0.1 for dark/light styles
adaptive_noise_scale = 0

# Speed Optimizations
cache_text_encoder_outputs = true       # Recommended
cache_latents = true                    # Recommended
persistent_data_loader_workers = true
gradient_accumulation_steps = 1

# Logging
logging_dir = "./logs"
log_with = "tensorboard"
log_prefix = "my_lora"
```

## Workflow

### Typical Training Session

1. **Dataset Preparation** (600-1200 images recommended)
   - Curate images with consistent aesthetic
   - Remove outliers manually
   - Caption with LLaVA or manually

2. **Caching** (one-time per dataset)
   ```bash
   cache_text_encoder  # ~5-10 minutes
   cache_vae          # ~10-20 minutes
   ```

3. **Training** (overnight)
   ```bash
   train_lora  # 6-12 hours depending on dataset size and epochs
   ```

4. **Evaluation**
   - Check sample outputs (if enabled)
   - Test at different epochs
   - Promising results often emerge around epoch 8

### Dataset Size Guidelines

Based on successful projects:

- **Minimum**: 600 images (spaghetti western project)
- **Recommended**: 800-1200 images (giallo, christmas projects)
- **Quality over quantity**: Manual curation to remove outliers is crucial

### Captioning Strategy

For aesthetic/mood LoRAs:
- Focus on atmosphere, lighting, color palette
- Avoid describing specific objects/decorations
- Let base model handle object recognition
- Use consistent terminology for the style

Example captions:
```
noir_style dramatic shadows, high contrast black and white
noir_style moody lighting, film noir atmosphere
```

## Scripts Reference

### cache_text_encoder

Caches text encoder outputs to speed up training.

```bash
cd ~/lora_projects/my_lora
cache_text_encoder

# Use different config
cache_text_encoder --config ../other_project/project.toml
```

### cache_vae

Caches VAE latents to reduce memory usage during training.

```bash
cd ~/lora_projects/my_lora
cache_vae

# Use different config
cache_vae --config ../other_project/project.toml
```

### train_lora

Runs the complete LoRA training process.

```bash
cd ~/lora_projects/my_lora
train_lora

# Use different config
train_lora --config ../other_project/project.toml
```

All scripts:
- Run from your project directory
- Auto-generate dataset configuration
- Display clear progress information
- Use unified `project.toml` configuration

## Output

Trained LoRA files are saved to `${COMFYUI_MODELS_ROOT}/loras/`:

```
loras/
├── my_lora-000002.safetensors
├── my_lora-000004.safetensors
├── my_lora-000006.safetensors
└── ...
```

If `save_state = true`, you'll also get state directories for potential checkpoint resume:

```
loras/
├── my_lora-000008.safetensors
├── my_lora-000008-state/          # Optimizer/scheduler state
└── ...
```

## Tips & Best Practices

### Dataset Curation

- **Thematic consistency** matters more than identical compositions
- Remove outliers that don't match the target aesthetic
- 600-1200 images is the sweet spot
- Modern trainers handle mixed aspect ratios well (bucketing)

### Training Strategy

- Start with default settings (learning_rate 5e-5, rank 32)
- Enable sample prompts to monitor progress
- Check results at different epochs (8, 12, 16)
- For dark/moody styles, consider `noise_offset = 0.05`
- For style-heavy training, try `min_snr_gamma = 5`

### Memory Management

- `blocks_to_swap = 16` is a good balance
- Lower blocks_to_swap = faster but more VRAM
- Higher blocks_to_swap = slower but less VRAM
- `fp8_base` is recommended over `fp8_scaled`
- Test `use_pinned_memory_for_block_swap` carefully

### When to Use Each Mode

- **Standard**: Aesthetic/mood LoRAs (most common use case)
- **Edit**: Inpainting or editing workflows
- **Edit Plus**: Advanced editing features

## Known Issues

### Checkpoint Resume Not Working

**Status**: Known issue, under investigation

**Problem**: The `resume_from_checkpoint` feature is currently not functioning correctly. While state files are saved, resuming from them does not work as expected.

**Workaround**: 
- Train to completion in a single session
- Use longer `max_epochs` if needed
- Enable `save_every_n_epochs` to have multiple checkpoints to evaluate

**Configuration** (currently non-functional):
```toml
[training.advanced]
resume_from_checkpoint = "my_lora-000008"  # Does not work currently
save_state = true                           # Still saves state files
```

### Pinned Memory on Windows

**Status**: System-dependent

**Problem**: `use_pinned_memory_for_block_swap` may cause issues on some Windows systems.

**Recommendation**:
- Test on your specific system
- Start with `false` (default)
- Only enable if you verify it works and improves performance
- Linux systems generally have fewer issues

## To-Do

### High Priority

- [ ] **Fix checkpoint resume functionality** - Currently not working, needs investigation
- [ ] **Character LoRA support** - Different approach needed vs aesthetic LoRAs
- [ ] **Dataset curator integration** - Currently exists but not documented/integrated
- [ ] **Automated tagging pipeline** - Integrate LLaVA tagging into main workflow

### Training Features

- [ ] **Dynamo optimizations** - Add support for PyTorch dynamo compilation
- [ ] **Custom timestep sampling** - More control over diffusion timesteps
- [ ] **Advanced schedulers** - Additional LR scheduler options
- [ ] **Multi-GPU training** - Distributed training support
- [ ] **LoRA merging utilities** - Tools to combine multiple LoRAs

### Workflow Improvements

- [ ] **Template system** - Pre-configured templates for common use cases
- [ ] **Progress tracking** - Better visualization of training progress
- [ ] **Automatic evaluation** - Built-in quality metrics
- [ ] **Web UI** - Optional web interface for configuration/monitoring
- [ ] **Batch processing** - Train multiple LoRAs in sequence

### Documentation

- [ ] **Video tutorials** - Step-by-step training guides
- [ ] **Example projects** - Complete noir, giallo, christmas examples
- [ ] **Troubleshooting guide** - Common issues and solutions
- [ ] **Advanced techniques** - Style learning, fine-tuning tips
- [ ] **Captioning best practices** - Detailed guide for different LoRA types

### Dataset Tools

- [ ] **Auto-curation tools** - AI-assisted outlier detection
- [ ] **Caption templates** - Style-specific captioning templates
- [ ] **Dataset analysis** - Statistics and quality checks
- [ ] **Augmentation pipeline** - Optional data augmentation
- [ ] **Deduplication** - Automatic duplicate image detection

### Quality of Life

- [ ] **Config validator** - Catch configuration errors before training
- [ ] **Dry-run mode** - Test configuration without training
- [ ] **Resource estimator** - Predict VRAM/time requirements
- [ ] **Auto-backup** - Automatic config and checkpoint backups
- [ ] **Notification system** - Alert when training completes

## Contributing

Contributions welcome! Areas of particular interest:

- Fixing the checkpoint resume functionality
- Character LoRA training methodology
- Documentation and examples
- Performance optimizations
- Dataset curation tools

## Project History

This system evolved from scattered scripts and multiple configuration formats into a unified pipeline. It represents learnings from successful LoRA projects including:

- Spaghetti western aesthetic (600 images)
- Giallo horror aesthetic (825 images)
- Film noir atmosphere (in progress)
- Christmas cozy aesthetic (1200 images planned)

## License

[Your license here]

## Acknowledgments

- Built on [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) by kohya-ss
- Uses Qwen Image models for training
- LLaVA for dataset captioning
