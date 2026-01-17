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
import multiprocessing as mp
from functools import partial
import sys
import time
import warnings

# Suppress Streamlit context warnings in worker processes
# These warnings occur because worker processes don't have Streamlit's ScriptRunContext
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit.runtime.scriptrunner_utils')

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from health_data.checkpoint import CheckpointManager


def _process_record_type_worker(record_type, records, checkpoint_dir=None, use_duckdb=False):
    """
    Worker function to process a single record type into DataFrame(s).
    Designed to be called in parallel.
    
    Args:
        record_type: Type of record (e.g., 'HKQuantityTypeIdentifierStepCount')
        records: List of record dictionaries
        checkpoint_dir: Optional checkpoint directory path for saving (recreates CheckpointManager)
        use_duckdb: Whether DuckDB is available for aggregations
        
    Returns:
        Tuple of (record_type, df, daily_df) or None if error
    """
    import sys
    import warnings
    
    # Suppress Streamlit context warnings in worker processes
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
    
    worker_start_time = time.time()
    try:
        # Recreate checkpoint manager in worker process (needed for Windows spawn)
        checkpoint_manager = None
        if checkpoint_dir:
            try:
                checkpoint_manager = CheckpointManager(checkpoint_dir)
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Worker] Warning: Could not create checkpoint manager: {e}", file=sys.__stderr__)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Worker] Processing {record_type} - {len(records):,} records", file=sys.__stderr__)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Worker] Created DataFrame: {len(df):,} rows, {len(df.columns)} columns", file=sys.__stderr__)
        
        # Convert dates in batch using pandas
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
        
        # Sort by date if available
        if "date" in df.columns:
            df = df.sort_values("date")
        elif "startDate" in df.columns:
            df = df.sort_values("startDate")
        
        # Save DataFrame checkpoint immediately (thread-safe file operations)
        if checkpoint_manager:
            try:
                checkpoint_manager.save_dataframe(record_type, df)
            except Exception as e:
                print(f"[Worker] Warning: Could not save checkpoint for {record_type}: {e}", file=sys.stderr)
        
        # Create daily aggregations
        daily_df = None
        if "HKQuantityTypeIdentifier" in record_type and "value_numeric" in df.columns and "date" in df.columns:
            # Use pandas for aggregations in workers (DuckDB can cause issues in multiprocessing)
            if len(df) > 0:
                try:
                    daily_agg = df.groupby("date")["value_numeric"].agg([
                        ('sum', 'sum'),
                        ('mean', 'mean'),
                        ('min', 'min'),
                        ('max', 'max'),
                        ('count', 'count')
                    ]).reset_index()
                    daily_agg.columns = ["date", "total", "mean", "min", "max", "count"]
                    daily_df = daily_agg
                    
                    if checkpoint_manager:
                        try:
                            checkpoint_manager.save_dataframe(record_type, daily_df, is_daily=True)
                        except Exception as e:
                            print(f"[Worker] Warning: Could not save daily checkpoint for {record_type}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"[Worker] Error creating daily aggregation for {record_type}: {e}", file=sys.stderr)
        
        worker_elapsed = time.time() - worker_start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Worker] ✓ Completed {record_type} in {worker_elapsed:.2f}s", file=sys.__stderr__)
        
        return (record_type, df, daily_df)
    
    except Exception as e:
        # Log error and return None (will be skipped)
        import traceback
        worker_elapsed = time.time() - worker_start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Worker] ✗ Error processing {record_type} after {worker_elapsed:.2f}s: {e}", file=sys.__stderr__)
        print(f"[Worker] Traceback: {traceback.format_exc()}", file=sys.__stderr__)
        return None


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
    start_time = time.time()
    print(f"\n{'='*80}", file=sys.__stderr__)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting parse_health_export", file=sys.__stderr__)
    print(f"[Cache] Export path: {export_path}", file=sys.__stderr__)
    print(f"[Cache] File size: {os.path.getsize(export_path) / (1024*1024):.2f} MB", file=sys.__stderr__)
    print(f"[Cache] Using {'lxml' if LXML_AVAILABLE else 'ElementTree'} parser", file=sys.__stderr__)
    print(f"{'='*80}\n", file=sys.__stderr__)
    
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
        count_start_time = time.time()
        if progress_callback:
            progress_callback(12, "Counting records...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Starting record counting phase...", file=sys.__stderr__)
        
        # Quick count pass with progress updates
        count_context = ET.iterparse(export_path, events=("start",), tag="Record", huge_tree=True)
        record_count = 0
        count_update_interval = 200000  # Update every 200k records during counting
        
        for _ in count_context:
            record_count += 1
            if record_count % count_update_interval == 0:
                elapsed = time.time() - count_start_time
                rate = record_count / elapsed if elapsed > 0 else 0
                if progress_callback:
                    progress_callback(12, f"Counting records... ({record_count:,} found so far)")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Counted {record_count:,} records ({rate:,.0f} records/sec)", file=sys.__stderr__)
        
        count_elapsed = time.time() - count_start_time
        if record_count == 0:
            raise ValueError("No records found in export file")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Counting complete: {record_count:,} total records in {count_elapsed:.2f}s ({record_count/count_elapsed:,.0f} records/sec)", file=sys.__stderr__)
        
        if progress_callback:
            progress_callback(12, f"Found {record_count:,} total records. Starting to parse...")
        del count_context  # Free memory
        
        # Parse records incrementally
        parse_start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Starting record parsing phase...", file=sys.__stderr__)
        
        for event, record in context:
            record_type = record.attrib.get("type", "")
            if not record_type:
                record.clear()  # Free memory immediately
                continue
            
            total_records += 1
            
            # Update progress less frequently for very large files (every 200k records)
            # This reduces UI overhead significantly - fewer updates = faster processing
            update_interval = 200000  # Always update every 200k records
            if progress_callback and record_count:
                if total_records % update_interval == 0 or total_records == 1 or total_records == record_count:
                    # Progress from 15% to 70% based on records processed
                    progress_pct = min(15 + int((total_records / record_count) * 55), 70)
                    progress_callback(progress_pct, f"Processed {total_records:,} of {record_count:,} records...")
                    
                    # Terminal logging
                    elapsed = time.time() - parse_start_time
                    rate = total_records / elapsed if elapsed > 0 else 0
                    pct = (total_records / record_count * 100) if record_count else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Parsed {total_records:,}/{record_count:,} records ({pct:.1f}%) - {rate:,.0f} records/sec", file=sys.__stderr__)
            
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Using ElementTree parser (lxml not available)", file=sys.__stderr__)
        
        if progress_callback:
            progress_callback(10, "Extracting all records...")
        
        tree = ET.parse(export_path)
        root = tree.getroot()
        
        # Count records first for progress
        count_start_time = time.time()
        if progress_callback:
            progress_callback(15, "Counting records...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Starting record counting phase...", file=sys.__stderr__)
        
        record_count = 0
        count_update_interval = 200000  # Update every 200k records during counting
        
        for _ in root.iter("Record"):
            record_count += 1
            if record_count % count_update_interval == 0:
                elapsed = time.time() - count_start_time
                rate = record_count / elapsed if elapsed > 0 else 0
                if progress_callback:
                    progress_callback(15, f"Counting records... ({record_count:,} found so far)")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Counted {record_count:,} records ({rate:,.0f} records/sec)", file=sys.__stderr__)
        
        count_elapsed = time.time() - count_start_time
        if record_count == 0:
            raise ValueError("No records found in export file")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Counting complete: {record_count:,} total records in {count_elapsed:.2f}s ({record_count/count_elapsed:,.0f} records/sec)", file=sys.__stderr__)
        
        if progress_callback:
            progress_callback(15, f"Found {record_count:,} total records. Starting to parse...")
        
        parse_start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Starting record parsing phase...", file=sys.__stderr__)
        
        for record in root.iter("Record"):
            record_type = record.attrib.get("type", "")
            if not record_type:
                continue
            
            total_records += 1
            
            # Update progress less frequently for very large files
            update_interval = 200000  # Always update every 200k records
            if progress_callback and record_count:
                if total_records % update_interval == 0 or total_records == 1 or total_records == record_count:
                    progress_pct = min(15 + int((total_records / record_count) * 55), 70)
                    progress_callback(progress_pct, f"Processed {total_records:,} of {record_count:,} records...")
                    
                    # Terminal logging
                    elapsed = time.time() - parse_start_time
                    rate = total_records / elapsed if elapsed > 0 else 0
                    pct = (total_records / record_count * 100) if record_count else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Parsed {total_records:,}/{record_count:,} records ({pct:.1f}%) - {rate:,.0f} records/sec", file=sys.__stderr__)
            
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
    
    # Calculate parsing elapsed time (parse_start_time is set in both branches)
    if 'parse_start_time' in locals():
        parse_elapsed = time.time() - parse_start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Parsing complete: {total_records:,} records in {parse_elapsed:.2f}s", file=sys.__stderr__)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Parsing complete: {total_records:,} records", file=sys.__stderr__)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Found {len(records_by_type)} unique record types", file=sys.__stderr__)
    
    # Print record type breakdown
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Record type breakdown:", file=sys.__stderr__)
    sorted_types = sorted(records_by_type.items(), key=lambda x: len(x[1]), reverse=True)
    for record_type, records in sorted_types[:10]:  # Top 10
        print(f"  - {record_type}: {len(records):,} records", file=sys.__stderr__)
    if len(sorted_types) > 10:
        print(f"  ... and {len(sorted_types) - 10} more types", file=sys.__stderr__)
    
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
    
    # Convert to DataFrames with parallel processing
    df_start_time = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [Cache] Starting DataFrame conversion phase...", file=sys.__stderr__)
    
    if progress_callback:
        progress_callback(75, "Converting to DataFrames...")
    dataframes = {}
    
    num_types = len(records_by_type)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Total record types to process: {num_types}", file=sys.__stderr__)
    
    # Check for existing DataFrames
    completed_df_types = set()
    if checkpoint_manager:
        completed = checkpoint_manager.get_completed_dataframes()
        completed_df_types = {rt for rt, is_daily in completed if not is_daily}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Found {len(completed_df_types)} cached DataFrames", file=sys.__stderr__)
    
    # Separate records into those that need processing and those already cached
    records_to_process = []
    cached_records = {}
    
    for record_type, records in records_by_type.items():
        if not records:
            continue
        
        if record_type in completed_df_types:
            # Load from checkpoint
            if checkpoint_manager:
                df = checkpoint_manager.load_dataframe(record_type)
                if df is not None:
                    cached_records[record_type] = df
                    # Try to load daily aggregation too
                    daily_df = checkpoint_manager.load_dataframe(record_type, is_daily=True)
                    if daily_df is not None:
                        cached_records[f"{record_type}_daily"] = daily_df
        else:
            records_to_process.append((record_type, records))
    
    # Process cached records first
    dataframes.update(cached_records)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Loaded {len(cached_records)} cached DataFrames", file=sys.__stderr__)
    
    # Process remaining records in parallel (with fallback to sequential)
    if records_to_process:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Need to process {len(records_to_process)} record types", file=sys.__stderr__)
        
        # Check if parallel processing is disabled via environment variable
        disable_parallel = os.environ.get('DISABLE_PARALLEL_PROCESSING', 'false').lower() == 'true'
        
        # Determine number of workers (use CPU count, but cap at reasonable number)
        # Reduce workers to avoid memory issues - use fewer workers for safety
        num_workers = min(mp.cpu_count(), len(records_to_process), 4)  # Reduced from 8 to 4
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] CPU count: {mp.cpu_count()}, Workers: {num_workers}, Parallel disabled: {disable_parallel}", file=sys.__stderr__)
        
        # Try parallel processing, but fall back to sequential if it fails or is disabled
        use_parallel = not disable_parallel and num_workers > 1 and len(records_to_process) > 1
        
        if use_parallel:
            # Parallel processing
            parallel_start_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Using PARALLEL processing with {num_workers} workers", file=sys.__stderr__)
            
            if progress_callback:
                progress_callback(76, f"Processing {len(records_to_process)} record types in parallel ({num_workers} workers)...")
            
            # Get checkpoint directory path (pass path instead of object for Windows compatibility)
            checkpoint_dir = None
            if checkpoint_manager:
                checkpoint_dir = str(checkpoint_manager.cache_dir)
            
            # Create worker function with checkpoint directory
            # Disable DuckDB in parallel workers (causes issues)
            worker_func = partial(
                _process_record_type_worker,
                checkpoint_dir=checkpoint_dir,
                use_duckdb=False  # Disable DuckDB in parallel workers
            )
            
            # Process in parallel with error handling
            try:
                # Use spawn method for Windows compatibility, fork for Mac/Linux
                try:
                    ctx = mp.get_context('spawn')  # Works on both Windows and Mac
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Using 'spawn' multiprocessing context", file=sys.__stderr__)
                except:
                    ctx = mp  # Fallback to default
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Using default multiprocessing context", file=sys.__stderr__)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Starting parallel pool...", file=sys.__stderr__)
                with ctx.Pool(processes=num_workers) as pool:
                    results = pool.starmap(worker_func, records_to_process, chunksize=1)
                
                parallel_elapsed = time.time() - parallel_start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Parallel processing completed in {parallel_elapsed:.2f}s", file=sys.__stderr__)
                
                # Collect results
                processed_count = 0
                successful_count = 0
                for result in results:
                    if result:
                        record_type, df, daily_df = result
                        dataframes[record_type] = df
                        if daily_df is not None:
                            dataframes[f"{record_type}_daily"] = daily_df
                        processed_count += 1
                        successful_count += 1
                        
                        # Update progress
                        if progress_callback:
                            progress_pct = 75 + int(((len(cached_records) + processed_count) / num_types) * 15)
                            progress_callback(progress_pct, f"Processed {len(cached_records) + processed_count}/{num_types} record types")
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Successfully processed {successful_count}/{len(records_to_process)} record types", file=sys.__stderr__)
            
            except Exception as e:
                # Fall back to sequential processing if parallel fails
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] ERROR: Parallel processing failed: {e}", file=sys.__stderr__)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Falling back to sequential processing...", file=sys.__stderr__)
                if progress_callback:
                    progress_callback(76, "Parallel processing failed, using sequential processing...")
                use_parallel = False
        
        if not use_parallel:
            # Sequential processing (fallback or default)
            sequential_start_time = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Using SEQUENTIAL processing", file=sys.__stderr__)
            
            checkpoint_dir = None
            if checkpoint_manager:
                checkpoint_dir = str(checkpoint_manager.cache_dir)
            
            for idx, (record_type, records) in enumerate(records_to_process):
                record_start_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Processing {record_type} ({idx+1}/{len(records_to_process)}) - {len(records):,} records", file=sys.__stderr__)
                
                # Use DuckDB in sequential mode (it's safe here)
                result = _process_record_type_worker(record_type, records, checkpoint_dir, DUCKDB_AVAILABLE)
                if result:
                    rt, df, daily_df = result
                    dataframes[rt] = df
                    if daily_df is not None:
                        dataframes[f"{rt}_daily"] = daily_df
                    
                    record_elapsed = time.time() - record_start_time
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] ✓ {record_type}: {len(df):,} rows in {record_elapsed:.2f}s", file=sys.__stderr__)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] ✗ Failed to process {record_type}", file=sys.__stderr__)
                
                if progress_callback:
                    progress_pct = 75 + int(((len(cached_records) + idx + 1) / num_types) * 15)
                    progress_callback(progress_pct, f"Created DataFrame for {record_type} ({len(cached_records) + idx + 1}/{num_types})")
            
            sequential_elapsed = time.time() - sequential_start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Sequential processing completed in {sequential_elapsed:.2f}s", file=sys.__stderr__)
    
    df_elapsed = time.time() - df_start_time
    total_elapsed = time.time() - start_time
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [Cache] DataFrame conversion completed in {df_elapsed:.2f}s", file=sys.__stderr__)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Total DataFrames created: {len(dataframes)}", file=sys.__stderr__)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Cache] Total processing time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)", file=sys.__stderr__)
    print(f"{'='*80}\n", file=sys.__stderr__)
    
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

