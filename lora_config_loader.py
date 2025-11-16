#!/usr/bin/env python3
"""
Shared configuration loader for LoRA training scripts
Loads unified project.toml and expands environment variables
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Handle Python 3.10 vs 3.11+ tomllib/tomli
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 fallback

class LoRAConfig:
    """Unified configuration loader for all LoRA scripts"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Load configuration from TOML file
        
        Args:
            config_path: Path to config file. If None, looks for project.toml in current dir
        """
        if config_path is None:
            config_path = Path.cwd() / "project.toml"
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            print(f"âŒ Error: Configuration file not found: {self.config_path}")
            print("Please create a project.toml file in your project directory.")
            sys.exit(1)
        
        # Load TOML config
        try:
            with open(self.config_path, 'rb') as f:
                self.config = tomllib.load(f)
        except Exception as e:
            print(f"âŒ Error loading config file: {e}")
            sys.exit(1)
        
        # Expand environment variables in paths
        self._expand_env_vars()
        
        # Resolve project-relative paths
        self.project_dir = self.config_path.parent
        self._resolve_paths()
    
    def _expand_env_vars(self):
        """Expand ${VAR} style environment variables in config"""
        def expand_value(value):
            if isinstance(value, str):
                # Replace ${VAR} with environment variable
                while '${' in value:
                    start = value.index('${')
                    end = value.index('}', start)
                    var_name = value[start+2:end]
                    var_value = os.environ.get(var_name, '')
                    if not var_value:
                        print(f"âš ï¸  Warning: Environment variable ${{{var_name}}} not set")
                    value = value[:start] + var_value + value[end+1:]
                return value
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(v) for v in value]
            return value
        
        self.config = expand_value(self.config)
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute paths based on project directory"""
        paths = self.config.get('paths', {})
        
        # Resolve dataset_dir and cache_dir relative to project
        for key in ['dataset_dir', 'cache_dir']:
            if key in paths and paths[key].startswith('./'):
                paths[key] = str(self.project_dir / paths[key][2:])
    
    def get(self, *keys, default=None):
        """
        Get nested config value
        
        Example:
            config.get('training', 'learning_rate')
            config.get('dataset_curator', 'filters', 'year_range')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    # ============================================
    # Convenience accessors for common values
    # ============================================
    
    @property
    def project_name(self) -> str:
        """Get project name"""
        return self.get('project', 'name', default='unnamed')
    
    @property
    def dataset_dir(self) -> Path:
        """Get dataset directory path"""
        return Path(self.get('paths', 'dataset_dir', default='./dataset'))
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory path"""
        return Path(self.get('paths', 'cache_dir', default='./cache_directory'))
    
    @property
    def models_root(self) -> Path:
        """Get ComfyUI models root path"""
        root = self.get('paths', 'models_root')
        if not root:
            print("âŒ Error: models_root not set (check COMFYUI_MODELS_ROOT env var)")
            sys.exit(1)
        return Path(root)
    
    @property
    def musubi_tuner_root(self) -> Path:
        """Get musubi-tuner repository root path"""
        root = self.get('paths', 'musubi_tuner_root')
        if not root:
            print("âŒ Error: musubi_tuner_root not set (check MUSUBI_TUNER_ROOT env var)")
            sys.exit(1)
        return Path(root)
    
    # ============================================
    # Dataset Curator helpers
    # ============================================
    
    def get_curator_config(self) -> Dict[str, Any]:
        """Get dataset curator configuration in expected format"""
        curator = self.get('dataset_curator', default={})
        
        return {
            'dataset_name': self.project_name,
            'output_dir': str(self.dataset_dir),
            'source_directories': curator.get('source_directories', []),
            'filters': curator.get('filters', {}),
            'sampling': curator.get('sampling', {}),
            'options': curator.get('options', {})
        }
    
    # ============================================
    # Training helpers
    # ============================================
    
    def get_model_path(self, model_type: str) -> Path:
        """
        Get full path to a model file
        
        Args:
            model_type: 'dit', 'vae', or 'text_encoder'
        """
        models = self.get('training', 'models', default={})
        
        # Map model types to subdirectories
        subdirs = {
            'dit': 'diffusion_models',
            'vae': 'vae',
            'text_encoder': 'text_encoders'
        }
        
        # Try both {model_type}_model and {model_type} for flexibility
        model_filename = models.get(f"{model_type}_model") or models.get(model_type)
        if not model_filename:
            print(f"âŒ Error: {model_type}_model or {model_type} not specified in config")
            print(f"   Add to [training.models] section: {model_type}_model = \"filename.safetensors\"")
            sys.exit(1)
        
        model_path = self.models_root / subdirs[model_type] / model_filename
        
        if not model_path.exists():
            print(f"âŒ Error: Model file not found: {model_path}")
            sys.exit(1)
        
        return model_path
    
    def get_dit_model_for_mode(self) -> Path:
        """
        Get the appropriate DIT model based on training_mode
        
        Returns:
            Path to the DIT model file
        """
        training_mode = self.get('training', 'training_mode', default='standard')
        models = self.get('training', 'models', default={})
        
        # Select the appropriate DIT model filename based on mode
        if training_mode == 'edit':
            model_filename = models.get('dit_model_edit')
            if not model_filename:
                print("❌ Error: training_mode is 'edit' but dit_model_edit not specified")
                print("   Add to [training.models] section: dit_model_edit = \"qwen_image_edit_bf16.safetensors\"")
                sys.exit(1)
        elif training_mode == 'edit_plus':
            model_filename = models.get('dit_model_edit_plus')
            if not model_filename:
                print("❌ Error: training_mode is 'edit_plus' but dit_model_edit_plus not specified")
                print("   Add to [training.models] section: dit_model_edit_plus = \"qwen_image_edit_plus_bf16.safetensors\"")
                sys.exit(1)
        else:  # standard mode
            model_filename = models.get('dit_model')
            if not model_filename:
                print("❌ Error: dit_model not specified in config")
                print("   Add to [training.models] section: dit_model = \"qwen_image_bf16.safetensors\"")
                sys.exit(1)
        
        model_path = self.models_root / 'diffusion_models' / model_filename
        
        if not model_path.exists():
            print(f"❌ Error: DIT model file not found: {model_path}")
            print(f"   Training mode: {training_mode}")
            sys.exit(1)
        
        return model_path
    
    def get_dataset_config_path(self) -> Path:
        """Get path to dataset TOML config (for training scripts)"""
        return self.project_dir / f"{self.project_name}.toml"
    
    def get_output_dir(self) -> Path:
        """Get LoRA output directory"""
        models_root = self.models_root
        return models_root / "loras"
    
    def generate_dataset_config(self) -> Dict[str, Any]:
        """
        Generate musubi-tuner dataset configuration from project.toml [dataset] section
        This replaces the need for a separate noir.toml file
        """
        dataset_config = self.get('dataset', default={})
        
        # Build the musubi-tuner format config (no subsets!)
        config = {
            'general': {
                'caption_extension': dataset_config.get('caption_extension', '.txt'),
            },
            'datasets': [{
                'image_directory': str(self.dataset_dir),
                'cache_directory': str(self.cache_dir),
                'resolution': dataset_config.get('resolution', [1024, 1024]),
                'batch_size': dataset_config.get('batch_size', 1),
                'num_repeats': dataset_config.get('num_repeats', 1),
            }]
        }
        
        # Add bucketing options to [general] section if enabled
        if dataset_config.get('enable_bucket'):
            config['general']['enable_bucket'] = True
            config['general']['bucket_no_upscale'] = dataset_config.get('bucket_no_upscale', True)
        
        # Add optional dataset-level fields if present
        if dataset_config.get('shuffle_caption'):
            config['datasets'][0]['shuffle_caption'] = True
        
        if dataset_config.get('caption_dropout_rate'):
            config['datasets'][0]['caption_dropout_rate'] = dataset_config['caption_dropout_rate']
        if dataset_config.get('caption_dropout_every_n_epochs'):
            config['datasets'][0]['caption_dropout_every_n_epochs'] = dataset_config['caption_dropout_every_n_epochs']
        if dataset_config.get('caption_tag_dropout_rate'):
            config['datasets'][0]['caption_tag_dropout_rate'] = dataset_config['caption_tag_dropout_rate']
        if dataset_config.get('color_aug'):
            config['datasets'][0]['color_aug'] = True
        if dataset_config.get('random_crop'):
            config['datasets'][0]['random_crop'] = True
        if dataset_config.get('flip_aug'):
            config['datasets'][0]['flip_aug'] = True
        
        return config
    
    def write_dataset_config(self, output_path: Optional[Path] = None) -> Path:
        """
        Write dataset configuration to a TOML file for musubi-tuner
        
        Args:
            output_path: Where to write the config. If None, writes to project_dir/{project_name}.toml
        
        Returns:
            Path to the written config file
        """
        if output_path is None:
            output_path = self.project_dir / f"{self.project_name}.toml"
        
        # For writing TOML, we need tomli_w
        try:
            import tomli_w
        except ImportError:
            print("âš ï¸  Warning: tomli_w not installed. Writing as text format instead.")
            print("   Install with: pip install tomli_w")
            # Fallback to manual writing
            config = self.generate_dataset_config()
            with open(output_path, 'w') as f:
                f.write('[general]\n')
                for key, value in config['general'].items():
                    if isinstance(value, bool):
                        f.write(f'{key} = {str(value).lower()}\n')
                    elif isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    elif isinstance(value, list):
                        f.write(f'{key} = {value}\n')
                    else:
                        f.write(f'{key} = {value}\n')
                
                f.write('\n[[datasets]]\n')
                dataset = config['datasets'][0]
                for key, value in dataset.items():
                    if isinstance(value, bool):
                        f.write(f'{key} = {str(value).lower()}\n')
                    elif isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    elif isinstance(value, list):
                        f.write(f'{key} = {value}\n')
                    else:
                        f.write(f'{key} = {value}\n')
            
            return output_path
        
        # Use tomli_w if available
        config = self.generate_dataset_config()
        with open(output_path, 'wb') as f:
            tomli_w.dump(config, f)
        
        return output_path
    
    def print_summary(self, script_name: str):
        """Print configuration summary for a script"""
        print("â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”")
        print(f"{script_name}")
        print("â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”")
        print(f"Project: {self.project_name}")
        print(f"Config: {self.config_path}")
        print(f"Dataset: {self.dataset_dir}")
        print("")


def load_config(config_path: Optional[Path] = None) -> LoRAConfig:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to config file. If None, looks for project.toml in current dir
    
    Returns:
        LoRAConfig instance
    """
    return LoRAConfig(config_path)


if __name__ == "__main__":
    # Test the config loader
    print("Testing configuration loader...")
    config = load_config()
    
    print(f"\nProject: {config.project_name}")
    print(f"Dataset dir: {config.dataset_dir}")
    print(f"Cache dir: {config.cache_dir}")
    print(f"Models root: {config.models_root}")
    
    print("\nDataset curator config:")
    import json
    print(json.dumps(config.get_curator_config(), indent=2))
    
    print("\nModel paths:")
    for model_type in ['dit', 'vae', 'text_encoder']:
        try:
            path = config.get_model_path(model_type)
            print(f"  {model_type}: {path}")
        except SystemExit:
            print(f"  {model_type}: (not found)")
