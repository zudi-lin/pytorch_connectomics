"""
Utility functions for Hydra configuration system.

Provides helpers for loading, saving, validating, and manipulating configs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from omegaconf import OmegaConf, DictConfig

from .hydra_config import Config


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with defaults merged
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    yaml_conf = OmegaConf.load(config_path)

    # Merge with structured config defaults
    default_conf = OmegaConf.structured(Config)
    merged = OmegaConf.merge(default_conf, yaml_conf)

    # Convert to dataclass instance
    cfg = OmegaConf.to_object(merged)

    return cfg


def save_config(cfg: Config, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        cfg: Config object to save
        save_path: Path where to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    omega_conf = OmegaConf.structured(cfg)
    OmegaConf.save(omega_conf, save_path)


def merge_configs(base_cfg: Config, *override_cfgs: Union[Config, Dict, str, Path]) -> Config:
    """
    Merge multiple configurations together.

    Args:
        base_cfg: Base configuration
        *override_cfgs: One or more override configs (Config, dict, or path to YAML)

    Returns:
        Merged Config object
    """
    result = OmegaConf.structured(base_cfg)

    for override_cfg in override_cfgs:
        if isinstance(override_cfg, (str, Path)):
            override_omega = OmegaConf.load(override_cfg)
        elif isinstance(override_cfg, Config):
            override_omega = OmegaConf.structured(override_cfg)
        elif isinstance(override_cfg, (dict, DictConfig)):
            override_omega = OmegaConf.create(override_cfg)
        else:
            raise TypeError(f"Unsupported config type: {type(override_cfg)}")

        result = OmegaConf.merge(result, override_omega)

    return OmegaConf.to_object(result)


def update_from_cli(cfg: Config, overrides: List[str]) -> Config:
    """
    Update config from command-line overrides.

    Supports dot notation: ['data.batch_size=4', 'model.architecture=unet3d']

    Args:
        cfg: Base Config object
        overrides: List of 'key=value' strings

    Returns:
        Updated Config object
    """
    cfg_omega = OmegaConf.structured(cfg)
    cli_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg_omega, cli_conf)
    return OmegaConf.to_object(merged)


def to_dict(cfg: Config, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert Config to dictionary.

    Args:
        cfg: Config object
        resolve: Whether to resolve variable interpolations

    Returns:
        Dictionary representation
    """
    omega_conf = OmegaConf.structured(cfg)
    return OmegaConf.to_container(omega_conf, resolve=resolve)


def from_dict(d: Dict[str, Any]) -> Config:
    """
    Create Config from dictionary.

    Args:
        d: Dictionary with configuration values

    Returns:
        Config object
    """
    default_conf = OmegaConf.structured(Config)
    dict_conf = OmegaConf.create(d)
    merged = OmegaConf.merge(default_conf, dict_conf)
    return OmegaConf.to_object(merged)


def print_config(cfg: Config, resolve: bool = True) -> None:
    """
    Pretty print configuration.

    Args:
        cfg: Config to print
        resolve: Whether to resolve variable interpolations
    """
    omega_conf = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(omega_conf, resolve=resolve))


def validate_config(cfg: Config) -> None:
    """
    Validate configuration values.

    Args:
        cfg: Config object to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Model validation
    if cfg.model.in_channels <= 0:
        raise ValueError("model.in_channels must be positive")
    if cfg.model.out_channels <= 0:
        raise ValueError("model.out_channels must be positive")
    if len(cfg.model.input_size) not in [2, 3]:
        raise ValueError("model.input_size must be 2D or 3D (got length {})".format(len(cfg.model.input_size)))

    # System validation
    if cfg.system.training.batch_size <= 0:
        raise ValueError("system.training.batch_size must be positive")
    if cfg.system.training.num_workers < 0:
        raise ValueError("system.training.num_workers must be non-negative")

    # Data validation
    if len(cfg.data.patch_size) not in [2, 3]:
        raise ValueError("data.patch_size must be 2D or 3D (got length {})".format(len(cfg.data.patch_size)))

    # Optimizer validation
    if cfg.optimization.optimizer.lr <= 0:
        raise ValueError("optimization.optimizer.lr must be positive")
    if cfg.optimization.optimizer.weight_decay < 0:
        raise ValueError("optimization.optimizer.weight_decay must be non-negative")

    # Training validation
    if cfg.optimization.max_epochs <= 0:
        raise ValueError("optimization.max_epochs must be positive")
    if cfg.optimization.gradient_clip_val < 0:
        raise ValueError("optimization.gradient_clip_val must be non-negative")
    if cfg.optimization.accumulate_grad_batches <= 0:
        raise ValueError("optimization.accumulate_grad_batches must be positive")

    # Loss validation
    if len(cfg.model.loss_functions) != len(cfg.model.loss_weights):
        raise ValueError("loss_functions and loss_weights must have same length")
    if any(w < 0 for w in cfg.model.loss_weights):
        raise ValueError("loss_weights must be non-negative")


def get_config_hash(cfg: Config) -> str:
    """
    Generate a hash string for the configuration.

    Useful for experiment tracking and reproducibility.

    Args:
        cfg: Config object

    Returns:
        Hash string
    """
    import hashlib

    omega_conf = OmegaConf.structured(cfg)
    yaml_str = OmegaConf.to_yaml(omega_conf, resolve=True)
    return hashlib.md5(yaml_str.encode()).hexdigest()[:8]


def create_experiment_name(cfg: Config) -> str:
    """
    Create a descriptive experiment name from config.

    Args:
        cfg: Config object

    Returns:
        Experiment name string
    """
    parts = [
        cfg.model.architecture,
        f"bs{cfg.system.training.batch_size}",
        f"lr{cfg.optimization.optimizer.lr:.0e}",
        get_config_hash(cfg),
    ]
    return "_".join(parts)


def resolve_data_paths(cfg: Config) -> Config:
    """
    Resolve data paths by combining base paths (train_path, val_path, test_path)
    with relative file paths (train_image, train_label, etc.).

    This function modifies the config in-place by:
    1. Prepending base paths to relative file paths
    2. Expanding glob patterns to actual file lists
    3. Flattening nested lists from glob expansion

    Supported paths:
    - Training: cfg.data.train_path + cfg.data.train_image/train_label/train_mask
    - Validation: cfg.data.val_path + cfg.data.val_image/val_label/val_mask
    - Testing (legacy): cfg.data.test_path + cfg.data.test_image/test_label/test_mask
    - Inference (primary): cfg.inference.data.test_path + cfg.inference.data.test_image/test_label/test_mask

    Args:
        cfg: Config object to resolve paths for

    Returns:
        Config object with resolved paths (same object, modified in-place)

    Example:
        >>> cfg.data.train_path = "/data/barcode/"
        >>> cfg.data.train_image = ["PT37/*_raw.tif", "file.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.data.train_image)
        ['/data/barcode/PT37/img1_raw.tif', '/data/barcode/PT37/img2_raw.tif', '/data/barcode/file.tif']

        >>> cfg.inference.data.test_path = "/data/test/"
        >>> cfg.inference.data.test_image = ["volume_*.tif"]
        >>> resolve_data_paths(cfg)
        >>> print(cfg.inference.data.test_image)
        ['/data/test/volume_1.tif', '/data/test/volume_2.tif']
    """
    import os
    from glob import glob

    def _combine_path(base_path: str, file_path: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
        """Helper to combine base path with file path(s) and expand globs."""
        if file_path is None:
            return file_path

        # Handle list of paths
        if isinstance(file_path, list):
            result = []
            for p in file_path:
                resolved = _combine_path(base_path, p)
                # If resolved is a list (from glob expansion), extend
                if isinstance(resolved, list):
                    result.extend(resolved)
                else:
                    result.append(resolved)
            return result

        # Handle string path
        # Combine with base path if relative
        if base_path and not os.path.isabs(file_path):
            file_path = os.path.join(base_path, file_path)

        # Expand glob patterns
        if "*" in file_path or "?" in file_path:
            expanded = sorted(glob(file_path))
            if expanded:
                return expanded
            else:
                # No matches - return original pattern (will be caught by validation)
                return file_path

        return file_path

    # Resolve training paths
    if cfg.data.train_path:
        cfg.data.train_image = _combine_path(cfg.data.train_path, cfg.data.train_image)
        cfg.data.train_label = _combine_path(cfg.data.train_path, cfg.data.train_label)
        cfg.data.train_mask = _combine_path(cfg.data.train_path, cfg.data.train_mask)
        cfg.data.train_json = _combine_path(cfg.data.train_path, cfg.data.train_json)

    # Resolve validation paths
    if cfg.data.val_path:
        cfg.data.val_image = _combine_path(cfg.data.val_path, cfg.data.val_image)
        cfg.data.val_label = _combine_path(cfg.data.val_path, cfg.data.val_label)
        cfg.data.val_mask = _combine_path(cfg.data.val_path, cfg.data.val_mask)
        cfg.data.val_json = _combine_path(cfg.data.val_path, cfg.data.val_json)

    # Resolve test paths (legacy support for cfg.data.test_path)
    if cfg.data.test_path:
        cfg.data.test_image = _combine_path(cfg.data.test_path, cfg.data.test_image)
        cfg.data.test_label = _combine_path(cfg.data.test_path, cfg.data.test_label)
        cfg.data.test_mask = _combine_path(cfg.data.test_path, cfg.data.test_mask)
        cfg.data.test_json = _combine_path(cfg.data.test_path, cfg.data.test_json)

    # Resolve inference data paths (primary location for test_path)
    if hasattr(cfg.inference.data, 'test_path') and cfg.inference.data.test_path:
        cfg.inference.data.test_image = _combine_path(cfg.inference.data.test_path, cfg.inference.data.test_image)
        cfg.inference.data.test_label = _combine_path(cfg.inference.data.test_path, cfg.inference.data.test_label)
        cfg.inference.data.test_mask = _combine_path(cfg.inference.data.test_path, cfg.inference.data.test_mask)

    return cfg


__all__ = [
    "load_config",
    "save_config",
    "merge_configs",
    "update_from_cli",
    "to_dict",
    "from_dict",
    "print_config",
    "validate_config",
    "get_config_hash",
    "create_experiment_name",
    "resolve_data_paths",
]
