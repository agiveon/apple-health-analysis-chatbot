"""
Checkpoint system for incremental caching and resuming processing.
"""
import os
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manages checkpoints for incremental processing"""
    
    def __init__(self, cache_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            cache_dir: Directory where checkpoints will be stored
        """
        # Convert to absolute path to avoid any path issues
        self.cache_dir = Path(cache_dir).resolve()
        self.checkpoint_dir = self.cache_dir / "checkpoints"
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.records_file = self.checkpoint_dir / "records.pkl"
        self.dataframes_dir = self.checkpoint_dir / "dataframes"
        self.dataframes_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug: Print checkpoint directory (will show in terminal)
        import sys
        print(f"Checkpoint directory: {self.checkpoint_dir}", file=sys.stderr)
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save processing metadata"""
        try:
            # Ensure directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            import sys
            print(f"Warning: Could not save metadata checkpoint: {e}", file=sys.stderr)
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load processing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_records(self, records_by_type: Dict[str, list]):
        """Save raw records by type"""
        try:
            # Ensure directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.records_file, 'wb') as f:
                pickle.dump(records_by_type, f)
        except Exception as e:
            import sys
            print(f"Warning: Could not save records checkpoint: {e}", file=sys.stderr)
    
    def load_records(self) -> Optional[Dict[str, list]]:
        """Load raw records by type"""
        if self.records_file.exists():
            with open(self.records_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_dataframe(self, record_type: str, df, is_daily: bool = False):
        """Save a single DataFrame"""
        try:
            suffix = "_daily" if is_daily else ""
            filename = f"{record_type}{suffix}.pkl"
            filepath = self.dataframes_dir / filename
            
            # Ensure directory exists
            self.dataframes_dir.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            # Log error but don't fail - checkpoint saving is optional
            import sys
            print(f"Warning: Could not save checkpoint for {record_type}: {e}", file=sys.stderr)
    
    def load_dataframe(self, record_type: str, is_daily: bool = False):
        """Load a single DataFrame"""
        suffix = "_daily" if is_daily else ""
        filename = f"{record_type}{suffix}.pkl"
        filepath = self.dataframes_dir / filename
        
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_completed_dataframes(self) -> set:
        """Get set of completed DataFrame record types"""
        completed = set()
        if self.dataframes_dir.exists():
            for file in self.dataframes_dir.glob("*.pkl"):
                name = file.stem
                if name.endswith("_daily"):
                    completed.add((name[:-6], True))
                else:
                    completed.add((name, False))
        return completed
    
    def clear(self):
        """Clear all checkpoints"""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.dataframes_dir.mkdir(exist_ok=True)

