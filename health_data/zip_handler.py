"""
Handle Apple Health zip file extraction.
"""
import zipfile
import os
import tempfile
import shutil
from pathlib import Path
import json


def extract_health_export(zip_path: str, extract_to: str = None, checkpoint_file: str = None) -> str:
    """
    Extract Apple Health export zip file and find export.xml.
    Supports checkpointing to avoid re-extraction.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to (defaults to same directory as zip)
        checkpoint_file: Optional path to checkpoint file to save extraction info
        
    Returns:
        Path to the extracted export.xml file
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    # Default extract location: same directory as zip file
    if extract_to is None:
        zip_dir = os.path.dirname(os.path.abspath(zip_path))
        extract_to = os.path.join(zip_dir, "apple_health_export")
    
    # Check checkpoint first
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            if 'extract_path' in checkpoint and os.path.exists(checkpoint['extract_path']):
                # Check if zip file hasn't changed
                zip_mtime = os.path.getmtime(zip_path)
                if checkpoint.get('zip_mtime') == zip_mtime:
                    return checkpoint['extract_path']
        except Exception:
            pass
    
    # Create extract directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Find export.xml (could be in root or in a subdirectory)
    export_xml = None
    
    # Check root first
    root_xml = os.path.join(extract_to, "export.xml")
    if os.path.exists(root_xml):
        export_xml = root_xml
    else:
        # Search in subdirectories
        for root, dirs, files in os.walk(extract_to):
            if "export.xml" in files:
                export_xml = os.path.join(root, "export.xml")
                break
    
    if export_xml is None:
        raise FileNotFoundError(
            f"Could not find export.xml in extracted zip file: {zip_path}\n"
            f"Extracted to: {extract_to}"
        )
    
    # Save checkpoint
    if checkpoint_file:
        try:
            checkpoint = {
                'extract_path': export_xml,
                'extract_dir': extract_to,
                'zip_path': zip_path,
                'zip_mtime': os.path.getmtime(zip_path)
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception:
            pass
    
    return export_xml


def get_zip_directory(zip_path: str) -> str:
    """Get the directory containing the zip file"""
    return os.path.dirname(os.path.abspath(zip_path))

