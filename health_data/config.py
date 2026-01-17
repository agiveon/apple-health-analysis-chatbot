"""
Configuration management for Health Data Analysis.
Handles loading and saving configuration from JSON file.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to config.json file. If None, uses default location.
        """
        if config_file is None:
            config_file = Path(__file__).parent.parent / "config.json"
        
        self.config_file = Path(config_file)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with defaults"""
        # Default cache directory: use project directory if config file is in project, otherwise home
        if self.config_file.parent.name in ["health", ".health_data"]:
            default_cache_dir = str(self.config_file.parent / "cache")
        else:
            default_cache_dir = str(Path.home() / ".health_data_cache")
        
        default_config = {
            "paths": {
                "export_path": "",
                "cache_dir": default_cache_dir,
                "cache_filename": "all_health_data.pkl"
            },
            "claude": {
                "model": "claude-opus-4-5",
                "model_alternatives": [
                    "claude-opus-4-5",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-5-sonnet",
                    "claude-sonnet-4-20250514",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229"
                ],
                "max_tokens": 4096
            },
            "dashboard": {
                "page_title": "Health Data Analysis Dashboard",
                "page_icon": "🏥",
                "layout": "wide"
            }
        }
        
        # Load from file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config = self._deep_merge(default_config, file_config)
            except Exception as e:
                # Silently use defaults if file is invalid
                config = default_config
        else:
            config = default_config
        
        # Apply environment variable overrides
        if os.getenv("CLAUDE_MODEL"):
            config["claude"]["model"] = os.getenv("CLAUDE_MODEL")
        
        if os.getenv("EXPORT_PATH"):
            config["paths"]["export_path"] = os.getenv("EXPORT_PATH")
        
        if os.getenv("CACHE_DIR"):
            config["paths"]["cache_dir"] = os.getenv("CACHE_DIR")
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'paths.export_path')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    @property
    def export_path(self) -> str:
        """Get export path"""
        return self.get("paths.export_path", "")
    
    @export_path.setter
    def export_path(self, value: str):
        """Set export path"""
        self.set("paths.export_path", value)
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory"""
        return self.get("paths.cache_dir", str(Path.home() / ".health_data_cache"))
    
    @cache_dir.setter
    def cache_dir(self, value: str):
        """Set cache directory"""
        self.set("paths.cache_dir", value)
    
    @property
    def cache_filename(self) -> str:
        """Get cache filename"""
        return self.get("paths.cache_filename", "all_health_data.pkl")
    
    @property
    def cache_path(self) -> str:
        """Get full cache path"""
        return os.path.join(self.cache_dir, self.cache_filename)
    
    @property
    def claude_model(self) -> str:
        """Get Claude model"""
        return self.get("claude.model", "claude-opus-4-5")
    
    @property
    def claude_model_alternatives(self) -> list:
        """Get Claude model alternatives"""
        return self.get("claude.model_alternatives", [])
    
    @property
    def claude_max_tokens(self) -> int:
        """Get Claude max tokens"""
        return self.get("claude.max_tokens", 4096)
    
    @property
    def dashboard_title(self) -> str:
        """Get dashboard title"""
        return self.get("dashboard.page_title", "Health Data Analysis Dashboard")
    
    @property
    def dashboard_icon(self) -> str:
        """Get dashboard icon"""
        return self.get("dashboard.page_icon", "🏥")
    
    @property
    def dashboard_layout(self) -> str:
        """Get dashboard layout"""
        return self.get("dashboard.layout", "wide")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()

