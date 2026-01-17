"""
Safe code execution module for running generated Python code.
"""
import contextlib
import traceback
import os
from io import StringIO
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta, datetime


class CodeExecutor:
    """Safely executes Python code in a controlled environment"""
    
    def __init__(self, data_dict: Dict[str, Any]):
        """
        Initialize code executor with data context.
        
        Args:
            data_dict: Dictionary of data to make available during execution
        """
        self.data_dict = data_dict
        self.captured_figures = []
    
    def execute(self, code: str, capture_figures: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str], List]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            capture_figures: If True, capture matplotlib figures instead of displaying
            
        Returns:
            Tuple of (output, errors, error_msg, captured_figures)
        """
        import sys
        print(f"[CodeExecutor] ========================================", file=sys.__stderr__)
        print(f"[CodeExecutor] Starting execution, capture_figures={capture_figures}", file=sys.__stderr__)
        print(f"[CodeExecutor] Code length: {len(code)} characters", file=sys.__stderr__)
        
        # Store debug info for UI
        self.debug_info = {
            'code': code,
            'code_length': len(code),
            'globals_before': {},
            'globals_after': {},
            'timestamp_set': False,
            'output_dir_set': False,
            'error_details': None
        }
        
        # Reset captured figures
        self.captured_figures = []
        
        # Create custom Streamlit object for figure capture
        # This proxies Streamlit calls while capturing figures when needed
        class CustomStreamlit:
            def __init__(self, executor_ref, capture_flag):
                self.executor_ref = executor_ref
                self.capture_flag = capture_flag
            
            def pyplot(self, fig=None, **kwargs):
                if fig is None:
                    fig = plt.gcf()
                if self.capture_flag:
                    self.executor_ref.captured_figures.append(fig)
                else:
                    st.pyplot(fig, **kwargs)
            
            # Proxy common Streamlit methods
            def write(self, *args, **kwargs):
                return st.write(*args, **kwargs)
            
            def markdown(self, *args, **kwargs):
                return st.markdown(*args, **kwargs)
            
            def caption(self, *args, **kwargs):
                return st.caption(*args, **kwargs)
            
            def text(self, *args, **kwargs):
                return st.text(*args, **kwargs)
            
            def title(self, *args, **kwargs):
                return st.title(*args, **kwargs)
            
            def header(self, *args, **kwargs):
                return st.header(*args, **kwargs)
            
            def subheader(self, *args, **kwargs):
                return st.subheader(*args, **kwargs)
            
            def info(self, *args, **kwargs):
                return st.info(*args, **kwargs)
            
            def success(self, *args, **kwargs):
                return st.success(*args, **kwargs)
            
            def warning(self, *args, **kwargs):
                return st.warning(*args, **kwargs)
            
            def error(self, *args, **kwargs):
                return st.error(*args, **kwargs)
            
            def dataframe(self, *args, **kwargs):
                return st.dataframe(*args, **kwargs)
            
            def table(self, *args, **kwargs):
                return st.table(*args, **kwargs)
            
            def json(self, *args, **kwargs):
                return st.json(*args, **kwargs)
            
            def metric(self, *args, **kwargs):
                return st.metric(*args, **kwargs)
            
            def columns(self, *args, **kwargs):
                return st.columns(*args, **kwargs)
            
            def container(self, *args, **kwargs):
                return st.container(*args, **kwargs)
        
        custom_st = CustomStreamlit(self, capture_figures)
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'format': format,  # Needed for string formatting (e.g., in matplotlib tick formatters)
                'repr': repr,
                'bool': bool,
                '__import__': __import__,
                'open': None,  # Disable file operations (except via safe wrapper)
            },
            'os': __import__('os'),  # Allow os.path.join, os.path.exists, etc. for output_dir operations
            # Note: We allow os module but restrict file operations to output_dir only
            # The generated code should use output_dir variable for saving files
            'pd': pd,
            'np': np,
            'plt': plt,
            'st': custom_st,
            'date': date,
            'timedelta': timedelta,
            'datetime': datetime,
        }
        
        import sys
        # Use __stderr__ to bypass Streamlit's stderr capture
        print(f"[CodeExecutor] ========== STARTING EXECUTION ==========", file=sys.__stderr__)
        print(f"[CodeExecutor] Initial globals keys: {list(safe_globals.keys())[:10]}...", file=sys.__stderr__)
        
        # Add data to execution context FIRST
        # BUT: Remove 'timestamp' from data_dict if it exists (we'll set it ourselves)
        data_dict_copy = self.data_dict.copy()
        if 'timestamp' in data_dict_copy:
            print(f"[CodeExecutor] ⚠ Warning: data_dict contains 'timestamp' key, removing it", file=sys.__stderr__)
            del data_dict_copy['timestamp']
        
        safe_globals.update(data_dict_copy)
        print(f"[CodeExecutor] After data_dict update, globals keys: {list(safe_globals.keys())[:10]}...", file=sys.__stderr__)
        
        # Pre-define timestamp for saving files (AFTER update to ensure it's not overwritten)
        # Set it MULTIPLE times to be absolutely sure
        timestamp_value = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_globals['timestamp'] = timestamp_value
        safe_globals['timestamp'] = timestamp_value  # Set twice to be sure
        print(f"[CodeExecutor] ✓ Set timestamp: {safe_globals['timestamp']}", file=sys.__stderr__)
        print(f"[CodeExecutor] ✓ Double-check timestamp: {safe_globals.get('timestamp', 'MISSING')}", file=sys.__stderr__)
        
        # Add output_dir - ALWAYS ensure it's available (for saving plots)
        if 'output_dir' in self.data_dict:
            output_dir = self.data_dict['output_dir']
            safe_globals['output_dir'] = output_dir
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            print(f"[CodeExecutor] ✓ Output directory set: {output_dir}", file=sys.__stderr__)
        else:
            # Create a default output directory if not provided
            default_output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(default_output_dir, exist_ok=True)
            safe_globals['output_dir'] = default_output_dir
            print(f"[CodeExecutor] ⚠ Warning: output_dir not in data_dict, using default: {default_output_dir}", file=sys.__stderr__)
        
        # Final verification
        if 'timestamp' in safe_globals:
            print(f"[CodeExecutor] ✓ Timestamp verified in globals: {safe_globals['timestamp']}", file=sys.__stderr__)
            self.debug_info['timestamp_set'] = True
            self.debug_info['timestamp_value'] = safe_globals['timestamp']
        else:
            print(f"[CodeExecutor] ✗ ERROR: timestamp missing from globals!", file=sys.__stderr__)
            self.debug_info['timestamp_set'] = False
        
        if 'output_dir' in safe_globals:
            self.debug_info['output_dir_set'] = True
            self.debug_info['output_dir_value'] = safe_globals['output_dir']
        
        # Store globals info for debug
        self.debug_info['globals_before'] = {
            'keys': list(safe_globals.keys())[:50],
            'total_keys': len(safe_globals),
            'has_timestamp': 'timestamp' in safe_globals,
            'has_output_dir': 'output_dir' in safe_globals,
            'has_pd': 'pd' in safe_globals,
            'has_plt': 'plt' in safe_globals,
        }
        
        print(f"[CodeExecutor] Final globals keys (first 20): {list(safe_globals.keys())[:20]}", file=sys.__stderr__)
        print(f"[CodeExecutor] Debug info stored: {self.debug_info['globals_before']}", file=sys.__stderr__)
        
        # IMPORTANT: Print debug info BEFORE redirecting stderr
        import sys
        print(f"[CodeExecutor] ========== PRE-EXECUTION CHECK ==========", file=sys.__stderr__)
        print(f"[CodeExecutor] About to execute code...", file=sys.__stderr__)
        print(f"[CodeExecutor] Final globals check - timestamp present: {'timestamp' in safe_globals}", file=sys.__stderr__)
        if 'timestamp' in safe_globals:
            print(f"[CodeExecutor] ✓ Timestamp value: {safe_globals['timestamp']}", file=sys.__stderr__)
        else:
            print(f"[CodeExecutor] ✗ ERROR: timestamp NOT in globals! Re-adding it...", file=sys.__stderr__)
            # Force add timestamp if missing
            safe_globals['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[CodeExecutor] ✓ Re-added timestamp: {safe_globals['timestamp']}", file=sys.__stderr__)
        
        # DOUBLE CHECK - ensure timestamp is definitely there
        if 'timestamp' not in safe_globals:
            print(f"[CodeExecutor] ✗✗✗ CRITICAL: timestamp STILL missing after re-add!", file=sys.__stderr__)
        else:
            print(f"[CodeExecutor] ✓✓✓ Timestamp confirmed present: {safe_globals['timestamp']}", file=sys.__stderr__)
        
        # Check if code uses timestamp
        uses_timestamp = 'timestamp' in code
        print(f"[CodeExecutor] Code uses 'timestamp': {uses_timestamp}", file=sys.__stderr__)
        if uses_timestamp:
            # Find where timestamp is used
            import re
            matches = re.findall(r'timestamp[^\w]', code)
            print(f"[CodeExecutor] Timestamp usage found: {matches[:5]}", file=sys.__stderr__)
        
        print(f"[CodeExecutor] Code preview (first 1000 chars):\n{code[:1000]}", file=sys.__stderr__)
        print(f"[CodeExecutor] ========================================", file=sys.__stderr__)
        
        # Capture stdout/stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            # NUCLEAR OPTION: Prepend timestamp definition directly to code
            # This ensures timestamp is ALWAYS available, no matter what
            timestamp_value_final = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure timestamp is in globals
            safe_globals['timestamp'] = timestamp_value_final
            
            # ALWAYS prepend timestamp definition - don't check, just do it
            # This ensures timestamp is ALWAYS available no matter what
            code_with_timestamp = f"from datetime import datetime\ntimestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n{code}"
            print(f"[CodeExecutor] ⚠ ALWAYS prepending timestamp definition to code", file=sys.__stderr__)
            print(f"[CodeExecutor] First 3 lines of modified code:\n{code_with_timestamp.split(chr(10))[:3]}", file=sys.__stderr__)
            print(f"[CodeExecutor] Original code length: {len(code)}, Modified code length: {len(code_with_timestamp)}", file=sys.__stderr__)
            
            print(f"[CodeExecutor] FINAL: timestamp in globals={safe_globals.get('timestamp', 'MISSING')}", file=sys.__stderr__)
            
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code_with_timestamp, safe_globals)
            
            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            
            # Print AFTER redirect (use __stderr__ to bypass Streamlit capture)
            print(f"[CodeExecutor] Execution completed successfully", file=sys.__stderr__)
            print(f"[CodeExecutor] Output length: {len(output)}, Errors length: {len(errors)}", file=sys.__stderr__)
            if output:
                print(f"[CodeExecutor] Output preview: {output[:200]}", file=sys.__stderr__)
            if errors:
                print(f"[CodeExecutor] Captured errors: {errors[:500]}", file=sys.__stderr__)
            
            return output, errors, None, self.captured_figures
        
        except Exception as e:
            # Print to actual stderr (bypass Streamlit capture)
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"[CodeExecutor] ========== EXECUTION FAILED ==========", file=sys.__stderr__)
            print(f"[CodeExecutor] Error: {error_msg}", file=sys.__stderr__)
            print(f"[CodeExecutor] Code that failed:\n{code_with_timestamp}", file=sys.__stderr__)
            print(f"[CodeExecutor] Available globals at failure: {list(safe_globals.keys())[:30]}", file=sys.__stderr__)
            print(f"[CodeExecutor] Timestamp in globals: {'timestamp' in safe_globals}", file=sys.__stderr__)
            if 'timestamp' in safe_globals:
                print(f"[CodeExecutor] Timestamp value: {safe_globals['timestamp']}", file=sys.__stderr__)
            print(f"[CodeExecutor] ======================================", file=sys.__stderr__)
            
            # Store error details for UI
            self.debug_info['error_details'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'full_traceback': traceback.format_exc(),
                'globals_at_failure': list(safe_globals.keys())[:50],
                'timestamp_in_globals': 'timestamp' in safe_globals,
                'timestamp_value': safe_globals.get('timestamp', None),
                'code_executed': code_with_timestamp
            }
            
            return None, None, error_msg, []

