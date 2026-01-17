"""
Data caching module for Apple Health exports.
Handles parsing, caching, and loading of all health data types.
"""
try:
    from lxml import etree as ET
    LXML_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as ET
    LXML_AVAILABLE = False

import pandas as pd
from datetime import datetime
import os
import pickle
from collections import defaultdict
from pathlib import Path

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from health_data.checkpoint import CheckpointManager


def parse_health_export(export_path, progress_callback=None, checkpoint_manager=None):
    """
    Parse Apple Health export and extract all record types.
    
    Args:
        export_path: Path to the Apple Health export.xml file
        progress_callback: Optional callback function(progress_pct, message) for progress updates
        
    Returns:
        Dictionary containing:
        - raw_records: Dict of DataFrames keyed by record type
        - metadata: Dict of metadata for each record type
        - record_counts: Dict of record counts by type
        - cache_date: When data was parsed
        - total_records: Total number of records
    """
    if not os.path.exists(export_path):
        raise FileNotFoundError(f"Export file not found: {export_path}")
    
    if progress_callback:
        progress_callback(5, "Loading XML file...")
    
    # Dictionary to store all records by type
    records_by_type = defaultdict(list)
    record_metadata = defaultdict(dict)
    
    total_records = 0
    
    # Use iterparse for incremental parsing (much faster and more memory efficient)
    # This processes records as they're encountered, without loading the entire tree
    if LXML_AVAILABLE:
        if progress_callback:
            progress_callback(10, "Parsing XML records (this may take a few minutes)...")
        
        # Use iterparse for incremental parsing - much faster for large files
        context = ET.iterparse(export_path, events=("start",), tag="Record", huge_tree=True)
        
        # Count records first for progress (quick pass)
        if progress_callback:
            progress_callback(12, "Counting records...")
            # Quick count pass
            count_context = ET.iterparse(export_path, events=("start",), tag="Record", huge_tree=True)
            record_count = sum(1 for _ in count_context)
            if record_count == 0:
                raise ValueError("No records found in export file")
            del count_context  # Free memory
        else:
            record_count = None
        
        # Parse records incrementally
        for event, record in context:
            record_type = record.attrib.get("type", "")
            if not record_type:
                record.clear()  # Free memory immediately
                continue
            
            total_records += 1
            
            # Update progress less frequently for very large files (every 200k records)
            # This reduces UI overhead significantly - fewer updates = faster processing
            update_interval = 200000 if record_count and record_count > 3000000 else 100000
            if progress_callback and record_count:
                if total_records % update_interval == 0 or total_records == 1 or total_records == record_count:
                    # Progress from 15% to 70% based on records processed
                    progress_pct = min(15 + int((total_records / record_count) * 55), 70)
                    progress_callback(progress_pct, f"Processed {total_records:,} of {record_count:,} records...")
            
            # Extract all attributes (use dict() for faster conversion)
            record_data = dict(record.attrib)
            
            # Track metadata for this record type
            if record_type not in record_metadata:
                record_metadata[record_type] = {
                    "attributes": set(record_data.keys()),
                    "sample_record": record_data,
                    "has_value": "value" in record_data,
                    "has_unit": "unit" in record_data,
                    "has_startDate": "startDate" in record_data,
                    "has_endDate": "endDate" in record_data,
                }
            else:
                record_metadata[record_type]["attributes"].update(record_data.keys())
            
            # Defer expensive date/numeric conversions until DataFrame creation
            # This significantly speeds up the parsing phase
            records_by_type[record_type].append(record_data)
            
            # Free memory immediately after processing
            record.clear()
            # Clear siblings to free memory (iterparse keeps ancestors)
            while record.getprevious() is not None:
                prev = record.getprevious()
                record.getparent().remove(prev)
    else:
        # Fallback to standard parsing (slower but works without lxml)
        if progress_callback:
            progress_callback(10, "Extracting all records...")
        
        tree = ET.parse(export_path)
        root = tree.getroot()
        
        # Count records first for progress
        if progress_callback:
            progress_callback(15, "Counting records...")
            record_count = sum(1 for _ in root.iter("Record"))
            if record_count == 0:
                raise ValueError("No records found in export file")
        else:
            record_count = None
        
        for record in root.iter("Record"):
            record_type = record.attrib.get("type", "")
            if not record_type:
                continue
            
            total_records += 1
            
            # Update progress less frequently for very large files
            update_interval = 50000 if record_count and record_count > 1000000 else 10000
            if progress_callback and record_count:
                if total_records % update_interval == 0 or total_records == 1 or total_records == record_count:
                    progress_pct = min(15 + int((total_records / record_count) * 55), 70)
                    progress_callback(progress_pct, f"Processed {total_records:,} of {record_count:,} records...")
            
            # Extract all attributes
            record_data = dict(record.attrib)
            records_by_type[record_type].append(record_data)
        
        # Track metadata for this record type
        if record_type not in record_metadata:
            record_metadata[record_type] = {
                "attributes": set(record_data.keys()),
                "sample_record": record_data,
                "has_value": "value" in record_data,
                "has_unit": "unit" in record_data,
                "has_startDate": "startDate" in record_data,
                "has_endDate": "endDate" in record_data,
            }
        else:
            record_metadata[record_type]["attributes"].update(record_data.keys())
    
    if progress_callback:
        progress_callback(70, f"Processed {total_records:,} total records, found {len(records_by_type)} unique types")
    
    # Save raw records checkpoint
    if checkpoint_manager:
        checkpoint_manager.save_records(dict(records_by_type))
        checkpoint_manager.save_metadata({
            'total_records': total_records,
            'record_types': list(records_by_type.keys()),
            'record_counts': {k: len(v) for k, v in records_by_type.items()}
        })
    
    # Convert to DataFrames
    if progress_callback:
        progress_callback(75, "Converting to DataFrames...")
    dataframes = {}
    
    num_types = len(records_by_type)
    
    # Check for existing DataFrames
    completed_df_types = set()
    if checkpoint_manager:
        completed = checkpoint_manager.get_completed_dataframes()
        completed_df_types = {rt for rt, is_daily in completed if not is_daily}
    
    # Use DuckDB for faster operations if available
    if DUCKDB_AVAILABLE:
        conn = duckdb.connect()
    
    for idx, (record_type, records) in enumerate(records_by_type.items()):
        if not records:
            continue
        
        # Skip if already processed
        if record_type in completed_df_types:
            if progress_callback:
                progress_pct = 75 + int((idx / num_types) * 15)
                progress_callback(progress_pct, f"Loading cached DataFrame for {record_type} ({idx+1}/{num_types})")
            
            # Load from checkpoint
            if checkpoint_manager:
                df = checkpoint_manager.load_dataframe(record_type)
                if df is not None:
                    dataframes[record_type] = df
                    # Try to load daily aggregation too
                    daily_df = checkpoint_manager.load_dataframe(record_type, is_daily=True)
                    if daily_df is not None:
                        dataframes[f"{record_type}_daily"] = daily_df
                    continue
        
        try:
            # Create DataFrame first (pandas is fast for this)
            df = pd.DataFrame(records)
            
            # Convert dates in batch using pandas (much faster than per-record)
            if "startDate" in df.columns:
                df["startDate"] = pd.to_datetime(df["startDate"], errors='coerce')
            
            if "endDate" in df.columns:
                df["endDate"] = pd.to_datetime(df["endDate"], errors='coerce')
            
            # Convert value to numeric in batch
            if "value" in df.columns:
                df["value_numeric"] = pd.to_numeric(df["value"], errors='coerce')
            
            # Add date column if we have startDate
            if "startDate" in df.columns:
                df["date"] = df["startDate"].dt.date
            
            # Sort by date if available (DuckDB can do this faster, but pandas is fine for sorting)
            if "date" in df.columns:
                df = df.sort_values("date")
            elif "startDate" in df.columns:
                df = df.sort_values("startDate")
            
            dataframes[record_type] = df
            
            # Save DataFrame checkpoint immediately
            if checkpoint_manager:
                checkpoint_manager.save_dataframe(record_type, df)
            
            # Create daily aggregations using DuckDB (much faster than pandas groupby)
            if "HKQuantityTypeIdentifier" in record_type and "value_numeric" in df.columns and "date" in df.columns:
                if DUCKDB_AVAILABLE and len(df) > 1000:  # Use DuckDB for larger datasets
                    try:
                        # Register DataFrame with DuckDB
                        conn.register('records', df)
                        
                        # Use DuckDB for fast aggregation
                        daily_agg = conn.execute("""
                            SELECT 
                                date,
                                SUM(value_numeric) as total,
                                AVG(value_numeric) as mean,
                                MIN(value_numeric) as min,
                                MAX(value_numeric) as max,
                                COUNT(*) as count
                            FROM records
                            WHERE value_numeric IS NOT NULL
                            GROUP BY date
                            ORDER BY date
                        """).df()
                        
                        dataframes[f"{record_type}_daily"] = daily_agg
                        # Save daily aggregation checkpoint
                        if checkpoint_manager:
                            checkpoint_manager.save_dataframe(record_type, daily_agg, is_daily=True)
                    except Exception:
                        # Fallback to pandas if DuckDB fails
                        daily_agg = df.groupby("date")["value_numeric"].agg([
                            ('sum', 'sum'),
                            ('mean', 'mean'),
                            ('min', 'min'),
                            ('max', 'max'),
                            ('count', 'count')
                        ]).reset_index()
                        daily_agg.columns = ["date", "total", "mean", "min", "max", "count"]
                        dataframes[f"{record_type}_daily"] = daily_agg
                        # Save daily aggregation checkpoint
                        if checkpoint_manager:
                            checkpoint_manager.save_dataframe(record_type, daily_agg, is_daily=True)
                elif len(df) > 0:
                    # Use pandas for smaller datasets
                    daily_agg = df.groupby("date")["value_numeric"].agg([
                        ('sum', 'sum'),
                        ('mean', 'mean'),
                        ('min', 'min'),
                        ('max', 'max'),
                        ('count', 'count')
                    ]).reset_index()
                    daily_agg.columns = ["date", "total", "mean", "min", "max", "count"]
                    dataframes[f"{record_type}_daily"] = daily_agg
                    # Save daily aggregation checkpoint
                    if checkpoint_manager:
                        checkpoint_manager.save_dataframe(record_type, daily_agg, is_daily=True)
            
            # Update progress
            if progress_callback:
                progress_pct = 75 + int((idx / num_types) * 15)
                progress_callback(progress_pct, f"Created DataFrame for {record_type} ({idx+1}/{num_types})")
            
        except Exception as e:
            # Silently skip problematic record types
            continue
    
    # Close DuckDB connection if used
    if DUCKDB_AVAILABLE and 'conn' in locals():
        conn.close()
    
    return {
        "raw_records": dataframes,
        "metadata": {k: {
            "attributes": list(v["attributes"]),
            "sample_record": v["sample_record"],
            "has_value": v["has_value"],
            "has_unit": v["has_unit"],
            "has_startDate": v["has_startDate"],
            "has_endDate": v["has_endDate"],
        } for k, v in record_metadata.items()},
        "record_counts": {k: len(v) for k, v in records_by_type.items()},
        "cache_date": datetime.now(),
        "total_records": total_records,
    }


def load_all_health_data(export_path, cache_path, force_reload=False, progress_callback=None, checkpoint_manager=None):
    """
    Load ALL health data from cache or parse XML if cache doesn't exist.
    
    Args:
        export_path: Path to the Apple Health export.xml file
        cache_path: Path where cache should be stored
        force_reload: If True, ignore cache and reload from XML
        progress_callback: Optional callback function(progress_pct, message) for progress updates
        
    Returns:
        Dictionary containing all health data
    """
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if cache exists and is newer than export file
    if not force_reload and os.path.exists(cache_path):
        cache_time = os.path.getmtime(cache_path)
        if os.path.exists(export_path):
            export_time = os.path.getmtime(export_path)
            
            if cache_time > export_time:
                if progress_callback:
                    progress_callback(50, "Loading health data from cache...")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                if progress_callback:
                    progress_callback(100, f"✅ Loaded cached data with {len(data['raw_records'])} record types")
                return data
    
    # Initialize checkpoint manager if not provided
    if checkpoint_manager is None:
        cache_dir = os.path.dirname(cache_path)
        checkpoint_manager = CheckpointManager(cache_dir)
    
    # Parse from XML (with checkpoint support)
    data = parse_health_export(export_path, progress_callback=progress_callback, checkpoint_manager=checkpoint_manager)
    
    # Save to cache
    if progress_callback:
        progress_callback(95, "Saving final cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    if progress_callback:
        progress_callback(100, f"✅ Processed and cached {data['total_records']:,} records")
    
    return data


def get_data_summary(data):
    """Get a summary of the cached data"""
    if data is None:
        return "No data loaded"
    
    raw_records = data.get('raw_records', {})
    if len(raw_records) == 0:
        return "No data available"
    
    # Group by category
    categories = defaultdict(list)
    for record_type in raw_records.keys():
        if "_daily" in record_type:
            continue
        if "HKCategoryTypeIdentifier" in record_type:
            categories["Category"].append(record_type)
        elif "HKQuantityTypeIdentifier" in record_type:
            categories["Quantity"].append(record_type)
        elif "HKWorkoutTypeIdentifier" in record_type:
            categories["Workout"].append(record_type)
        else:
            categories["Other"].append(record_type)
    
    summary = {
        "total_record_types": len([k for k in raw_records.keys() if "_daily" not in k]),
        "total_records": data.get('total_records', 0),
        "cache_date": data.get('cache_date', 'Unknown'),
        "categories": dict(categories),
        "record_counts": data.get('record_counts', {}),
    }
    
    return summary


def get_available_data_types(data):
    """Get a list of available data types with descriptions"""
    if not data or 'raw_records' not in data:
        return []
    
    available = []
    metadata = data.get('metadata', {})
    
    for record_type, df in data['raw_records'].items():
        if "_daily" in record_type:
            continue
        
        info = {
            "type": record_type,
            "count": len(df),
            "columns": list(df.columns),
            "has_daily": f"{record_type}_daily" in data['raw_records'],
        }
        
        if record_type in metadata:
            info.update({
                "has_value": metadata[record_type].get("has_value", False),
                "has_unit": metadata[record_type].get("has_unit", False),
                "date_range": None,
            })
            
            if "date" in df.columns and len(df) > 0:
                info["date_range"] = f"{df['date'].min()} to {df['date'].max()}"
        
        available.append(info)
    
    return sorted(available, key=lambda x: x['count'], reverse=True)

