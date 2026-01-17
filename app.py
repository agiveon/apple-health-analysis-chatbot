"""
Health Data Analysis Dashboard
Interactive Streamlit app for analyzing Apple Health data with AI-powered insights.
"""
import streamlit as st
import pandas as pd
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from health_data.config import Config
from health_data.cache import load_all_health_data, get_data_summary
from health_data.ai_client import get_ai_client
from health_data.code_executor import CodeExecutor
from health_data.prompts import build_analysis_prompt, get_data_structure_info, build_conversation_context, build_error_recovery_prompt
from health_data.zip_handler import extract_health_export, get_zip_directory
from health_data.checkpoint import CheckpointManager

# Page configuration
st.set_page_config(
    page_title="Health Data Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'zip_path' not in st.session_state:
    st.session_state.zip_path = None
if 'export_path' not in st.session_state:
    st.session_state.export_path = None
if 'cache_path' not in st.session_state:
    st.session_state.cache_path = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None


def find_or_create_cache(zip_path: str) -> tuple:
    """
    Find or create cache based on zip file location.
    
    Returns:
        (export_path, cache_path, output_dir)
    """
    zip_dir = get_zip_directory(zip_path)
    
    # Cache and output directories are in the same folder as the zip
    cache_dir = zip_dir
    output_dir = os.path.join(zip_dir, "output")
    cache_path = os.path.join(cache_dir, "all_health_data.pkl")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Checkpoint file for extraction
    checkpoint_file = os.path.join(cache_dir, "extraction_checkpoint.json")
    
    # Check if already extracted (using checkpoint)
    extract_dir = os.path.join(zip_dir, "apple_health_export")
    export_xml = os.path.join(extract_dir, "export.xml")
    
    if not os.path.exists(export_xml):
        # Extract the zip file (with checkpoint support)
        with st.spinner("📦 Extracting Apple Health export..."):
            export_xml = extract_health_export(zip_path, extract_dir, checkpoint_file=checkpoint_file)
    else:
        # Verify checkpoint is valid
        try:
            import json
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                zip_mtime = os.path.getmtime(zip_path)
                if checkpoint.get('zip_mtime') != zip_mtime:
                    # Zip file changed, re-extract
                    with st.spinner("📦 Re-extracting (zip file changed)..."):
                        export_xml = extract_health_export(zip_path, extract_dir, checkpoint_file=checkpoint_file)
        except Exception:
            pass
    
    return export_xml, cache_path, output_dir


def load_data(export_path: str, cache_path: str, force_reload: bool = False, progress_callback=None):
    """Load data, creating cache if needed"""
    return load_all_health_data(export_path, cache_path, force_reload=force_reload, progress_callback=progress_callback)


# Main title
st.title("🏥 Health Data Analysis Dashboard")
st.markdown("Chat with your health data! Ask questions or request visualizations about any data in your Apple Health export.")

# Check if we have data loaded
if st.session_state.data is None:
    # First, check config.json for existing cache
    config = Config()
    cache_path_from_config = config.get("paths.cache_path", "")
    zip_path_from_config = config.get("paths.zip_path", "")
    
    cache_found = False
    
    # If cache path exists in config and cache file exists, load it
    if cache_path_from_config and os.path.exists(cache_path_from_config):
        try:
            with st.spinner("⚡ Loading cached data..."):
                with open(cache_path_from_config, 'rb') as f:
                    import pickle
                    st.session_state.data = pickle.load(f)
                
                # Load output_dir from config to ensure plots are saved in the correct location
                output_dir_from_config = config.get("paths.output_dir", "")
                if output_dir_from_config:
                    st.session_state.output_dir = output_dir_from_config
                elif zip_path_from_config:
                    # Derive output_dir from zip_path if not in config
                    zip_dir = get_zip_directory(zip_path_from_config)
                    st.session_state.output_dir = os.path.join(zip_dir, "output")
                    os.makedirs(st.session_state.output_dir, exist_ok=True)
                
                # Also set zip_path and other paths from config
                if zip_path_from_config:
                    st.session_state.zip_path = zip_path_from_config
                st.session_state.cache_path = cache_path_from_config
                st.session_state.export_path = config.get("paths.export_path", "")
                
                cache_found = True
                st.success("✅ Loaded cached data!")
                st.rerun()
        except Exception as e:
            st.warning(f"⚠️ Could not load cache: {e}")
            cache_found = False
    
    if not cache_found:
        # Show file picker front and center
        st.markdown("---")
        st.markdown("### 📁 Select Your Apple Health Export")
        st.markdown("Please select the **zipped** Apple Health export file you downloaded from your iPhone.")
        st.markdown("")
        
        # Show current path if set
        if st.session_state.zip_path:
            st.success(f"✅ **Selected file**: `{st.session_state.zip_path}`")
            st.info(f"📁 **Files will be saved in**: `{get_zip_directory(st.session_state.zip_path)}`")
            if st.button("🔄 Change File", key="change_file"):
                # Clear config to force re-selection
                config = Config()
                config.set("paths.zip_path", "")
                config.set("paths.cache_path", "")
                config.set("paths.export_path", "")
                config.set("paths.output_dir", "")
                config.save()
                
                st.session_state.zip_path = None
                st.session_state.data = None
                st.session_state.export_path = None
                st.session_state.cache_path = None
                st.rerun()
        else:
            # Instructions for getting file path from Finder
            st.info("""
            **How to get the file path:**
            1. Open Finder and locate your zip file
            2. Right-click (or Control+click) on the file
            3. Hold the **Option** key (⌥)
            4. Click **"Copy as Pathname"** (or press Cmd+Option+C)
            5. Paste it in the field below
            """)
            
            # Text input for file path
            file_path_input = st.text_input(
                "📂 Zip file path:",
                value=st.session_state.get('pending_file_path', ''),
                key="file_path_input",
                placeholder="/path/to/your/apple_health_export.zip",
                help="Paste the full path to your Apple Health export zip file"
            )
            
            # Handle path entry
            if file_path_input and file_path_input.strip():
                file_path = file_path_input.strip()
                if os.path.exists(file_path) and file_path.endswith('.zip'):
                    st.session_state.zip_path = file_path
                    st.session_state.data = None
                    st.session_state.export_path = None
                    st.session_state.cache_path = None
                    if 'pending_file_path' in st.session_state:
                        del st.session_state.pending_file_path
                    st.success(f"✅ File selected: `{file_path}`")
                    st.info(f"📁 Files will be saved in: `{get_zip_directory(file_path)}`")
                    st.rerun()
                elif file_path and not os.path.exists(file_path):
                    st.error(f"❌ File not found: `{file_path}`")
                    st.info("💡 Make sure you copied the full path including the filename")
                elif file_path and not file_path.endswith('.zip'):
                    st.warning("⚠️ File doesn't appear to be a zip file.")
        
        # If zip path is set and data not loaded yet, extract and create cache
        if st.session_state.zip_path and st.session_state.data is None:
            zip_path = st.session_state.zip_path
            
            # Verify the path and show what will be used
            zip_dir = get_zip_directory(zip_path)
            
            # Show the zip path being used for clarity - make it VERY visible
            st.markdown("---")
            st.markdown("### 📍 Configuration")
            st.success(f"📂 **Zip file**: `{zip_path}`")
            st.success(f"📁 **Save location**: `{zip_dir}`")
            st.success(f"💾 **Checkpoints**: `{zip_dir}/checkpoints/`")
            
            # Verify the path is correct
            if not os.path.exists(zip_path):
                st.error(f"❌ Zip file not found: `{zip_path}`")
                st.stop()
            
            # Double-check the directory matches what we expect
            expected_dir = os.path.dirname(os.path.abspath(zip_path))
            if zip_dir != expected_dir:
                st.error(f"❌ **PATH MISMATCH!** Expected: `{expected_dir}`, Got: `{zip_dir}`")
                st.error("This should never happen. Please report this bug.")
                st.stop()
            
            # Show progress steps
            st.markdown("---")
            st.markdown("### 🔄 Processing Your Health Data")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Extract zip file
                status_text.text("📦 Step 1/3: Extracting zip file...")
                progress_bar.progress(10)
                
                export_path, cache_path, output_dir = find_or_create_cache(zip_path)
                st.session_state.export_path = export_path
                st.session_state.cache_path = cache_path
                st.session_state.output_dir = output_dir
                
                progress_bar.progress(30)
                status_text.text("✅ Step 1/3: Zip file extracted successfully!")
                st.info(f"📁 Extracted to: `{os.path.dirname(export_path)}`")
                
                # Step 2: Process and cache data
                status_text.text("🔄 Step 2/3: Processing health data (this may take a few minutes)...")
                progress_bar.progress(40)
                
                # Use a container to show progress messages

                progress_container = st.container()
                with progress_container:
                    st.info("⏳ Parsing XML and creating cache... This is a one-time process.")
                    st.info("💡 Tip: The cache will be reused next time, so this will be much faster!")
                
                # Create progress callback for real-time updates
                def update_progress(pct, msg):
                    """Update progress bar and status text"""
                    # Map cache progress (0-100) to step 2 progress (40-90)
                    step2_progress = 40 + int((pct / 100) * 50)
                    progress_bar.progress(step2_progress)
                    status_text.text(f"🔄 Step 2/3: {msg}")
                
                # Initialize checkpoint manager - use zip directory directly to be absolutely sure
                zip_dir = get_zip_directory(zip_path)  # Get directory from zip file path directly
                checkpoint_manager = CheckpointManager(zip_dir)
                
                # Debug: Show where checkpoints will be saved
                st.info(f"💾 Checkpoints will be saved to: `{zip_dir}/checkpoints/`")
                
                st.session_state.data = load_all_health_data(
                    export_path, 
                    cache_path, 
                    force_reload=False,
                    progress_callback=update_progress,
                    checkpoint_manager=checkpoint_manager
                )
                
                progress_bar.progress(90)
                status_text.text("✅ Step 2/3: Health data processed and cached!")
                
                # Step 3: Complete
                progress_bar.progress(100)
                status_text.text("🎉 Step 3/3: Complete! Your data is ready.")
                
                # Show summary
                summary = get_data_summary(st.session_state.data)
                if isinstance(summary, dict):
                    st.success("✅ Health data loaded successfully!")
                    st.markdown("---")
                    st.markdown("### 📊 Your Data Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Record Types", f"{summary.get('total_record_types', 0):,}")
                        st.metric("Total Records", f"{summary.get('total_records', 0):,}")
                    with col2:
                        if 'categories' in summary:
                            for cat, types in summary['categories'].items():
                                st.write(f"**{cat}:** {len(types)} types")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Save cache info to config.json for future loads
                config = Config()
                config.set("paths.zip_path", zip_path)
                config.set("paths.cache_path", cache_path)
                config.set("paths.export_path", export_path)
                config.set("paths.output_dir", output_dir)
                config.save()
                
                st.balloons()  # Celebration!
                st.success("🎉 Data loaded and cached! Refreshing to show chat interface...")
                
                # Rerun to show the chat interface (data is now loaded, so the if block will be skipped)
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing export: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                st.stop()
        
        # If we get here and data is still None, stop (shouldn't happen, but safety check)
        if st.session_state.data is None:
            st.stop()

# Data is loaded - show chat interface
data = st.session_state.data

# Prepare data for code execution
try:
    data_dict = {'raw_records': {}}
    
    raw_records = data.get('raw_records', {})
    for record_type, df in raw_records.items():
        # Create sanitized variable names
        safe_name = record_type.replace("HKQuantityTypeIdentifier", "qty_").replace(
            "HKCategoryTypeIdentifier", "cat_").replace(
            "HKWorkoutTypeIdentifier", "workout_").replace(
            "-", "_").replace(".", "_")
        data_dict[safe_name] = df.copy()
        data_dict['raw_records'][record_type] = df.copy()
    
    data_dict['metadata'] = data.get('metadata', {})
    data_dict['available_types'] = list(raw_records.keys())
    
    # Add output directory to data dict for code execution
    # Always set output_dir - use session state if available, otherwise create a default
    if st.session_state.output_dir:
        data_dict['output_dir'] = st.session_state.output_dir
    else:
        # Create a default output directory in the current working directory
        default_output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(default_output_dir, exist_ok=True)
        data_dict['output_dir'] = default_output_dir
    
except Exception as e:
    st.error(f"Error preparing data: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("💬 Conversation")
    
    # Start New Conversation button
    if st.button("🔄 Start New Conversation", use_container_width=True, type="primary"):
        # Clear all messages and start fresh
        st.session_state.messages = []
        st.rerun()

# Main content area

# Show debug panel if there was an error (OUTSIDE chat context - ALWAYS VISIBLE)
if 'last_error' in st.session_state and st.session_state['last_error']:
    error_data = st.session_state['last_error']
    st.error("❌ **ERROR DETECTED - DEBUG INFORMATION BELOW**")
    st.markdown("---")
    st.markdown("## 🐛 DEBUG INFORMATION")
    
    st.markdown("### 📋 Full Generated Code")
    st.code(error_data['code'], language="python")
    
    st.markdown("---")
    st.markdown("### 🔍 Error Details")
    st.code(error_data['error_msg'], language="python")
    
    st.markdown("---")
    st.markdown("### 📦 Data Dictionary Contents")
    st.write("**Keys in data_dict:**")
    st.write(error_data['data_dict_keys'])
    
    st.markdown("---")
    st.markdown("### 🎯 CodeExecutor Debug Info")
    if error_data['executor_debug_info']:
        st.json(error_data['executor_debug_info'])
    else:
        st.warning("No executor debug info available")
    
    st.markdown("---")
    # Clear error state after showing (but keep it for this render)
    # Don't clear immediately so it persists across reruns

# Show data summary
with st.expander("📊 Data Summary", expanded=False):
    summary = get_data_summary(data)
    if isinstance(summary, dict):
        st.write(f"**Record Types:** {summary.get('total_record_types', 0)}")
        st.write(f"**Total Records:** {summary.get('total_records', 0):,}")
        if 'categories' in summary:
            for cat, types in summary['categories'].items():
                st.write(f"**{cat}:** {len(types)} types")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            if message.get("type") == "text":
                st.write(message["content"])
            elif message.get("type") == "plot":
                if "figure" in message:
                    st.pyplot(message["figure"])
                elif "code" in message:
                    try:
                        executor = CodeExecutor(data_dict)
                        output, errors, error_msg, _ = executor.execute(
                            message["code"], 
                            capture_figures=False
                        )
                        if error_msg:
                            st.error(f"Error re-displaying plot: {error_msg}")
                            # Show debug for re-display errors too
                            with st.expander("🐛 DEBUG INFO", expanded=True):
                                st.code(message.get("code", "No code available"), language="python")
                                st.code(error_msg, language="python")
                    except Exception as e:
                        import traceback
                        st.error(f"Error re-displaying plot: {str(e)}")
                        # Show debug for exceptions too
                        with st.expander("🐛 DEBUG INFO", expanded=True):
                            st.code(traceback.format_exc(), language="python")
                            if "code" in message:
                                st.code(message["code"], language="python")
                else:
                    st.write("📊 Visualization")
            elif message.get("type") == "error":
                st.error(message["content"])
                # Show debug panel for errors - ALWAYS VISIBLE
                st.markdown("---")
                st.markdown("## 🐛 DEBUG INFORMATION")
                if "code" in message:
                    st.markdown("### 📋 Code That Failed")
                    st.code(message["code"], language="python")
                st.markdown("### 🔍 Error Message")
                st.code(message["content"], language="python")
                if "debug_info" in message and message["debug_info"]:
                    st.markdown("### 🎯 CodeExecutor Debug Info")
                    st.json(message["debug_info"])
                if "data_dict_keys" in message:
                    st.markdown("### 📦 Available Data Keys")
                    st.write(message["data_dict_keys"])

# Chat input
if prompt := st.chat_input("Ask a question about your health data..."):
    # Reset error retry count for new query
    st.session_state['error_retry_count'] = 0
    
    # If this is the first message in a new conversation, we'll create it after getting response
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get data summary and structure
                summary = get_data_summary(data)
                if isinstance(summary, dict):
                    summary_str = f"""
Total record types: {summary.get('total_record_types', 0)}
Total records: {summary.get('total_records', 0):,}
Categories: {', '.join(summary.get('categories', {}).keys())}
Top record types: {', '.join([k for k, v in sorted(summary.get('record_counts', {}).items(), key=lambda x: x[1], reverse=True)[:10]])}
"""
                else:
                    summary_str = str(summary)
                
                data_structure = get_data_structure_info(data)
                conversation_context = build_conversation_context(st.session_state.messages[:-1])
                
                # Build prompt
                full_prompt = build_analysis_prompt(
                    prompt,
                    summary_str,
                    data_structure,
                    conversation_context
                )
                
                # Initialize AI client (Claude or OpenAI based on available API key)
                config = Config()
                ai_client = get_ai_client(config)
                
                # Generate response
                response = ai_client.generate_response(full_prompt)
                
                if response["type"] == "text":
                    st.write(response["content"])
                    assistant_msg = {
                        "role": "assistant",
                        "type": "text",
                        "content": response["content"]
                    }
                    st.session_state.messages.append(assistant_msg)
                
                elif response["type"] == "code":
                    code = response["content"]
                    import sys
                    # Use __stderr__ to bypass Streamlit capture - goes directly to terminal
                    print(f"[App] ========== CODE EXECUTION START ==========", file=sys.__stderr__)
                    print(f"[App] Generated code type: code", file=sys.__stderr__)
                    print(f"[App] Code length: {len(code)} characters", file=sys.__stderr__)
                    print(f"[App] Code preview:\n{code[:500]}", file=sys.__stderr__)
                    print(f"[App] Data dict keys: {list(data_dict.keys())[:10]}", file=sys.__stderr__)
                    print(f"[App] Output dir in data_dict: {'output_dir' in data_dict}", file=sys.__stderr__)
                    if 'output_dir' in data_dict:
                        print(f"[App] Output dir value: {data_dict['output_dir']}", file=sys.__stderr__)
                    
                    executor = CodeExecutor(data_dict)
                    output, errors, error_msg, captured_figures = executor.execute(
                        code, 
                        capture_figures=True
                    )
                    
                    # Store executor debug info for UI
                    executor_debug_info = getattr(executor, 'debug_info', {})
                    
                    print(f"[App] Execution result - output: {bool(output)}, errors: {bool(errors)}, error_msg: {bool(error_msg)}", file=sys.__stderr__)
                    if error_msg:
                        print(f"[App] ERROR MESSAGE: {error_msg[:500]}", file=sys.__stderr__)
                    print(f"[App] ========== CODE EXECUTION END ==========", file=sys.__stderr__)
                    
                    if error_msg:
                        import sys
                        print(f"[App] ========== ERROR DETECTED ==========", file=sys.__stderr__)
                        print(f"[App] Error message: {error_msg}", file=sys.__stderr__)
                        print(f"[App] Full code:\n{code}", file=sys.__stderr__)
                        print(f"[App] ====================================", file=sys.__stderr__)
                        
                        # Check if we should retry (limit to 1 retry to avoid infinite loops)
                        retry_count = st.session_state.get('error_retry_count', 0)
                        max_retries = 1
                        error_handled = False  # Initialize flag
                        
                        if retry_count < max_retries:
                            # Show that we're retrying
                            st.warning(f"⚠️ Error detected. Attempting to fix and retry... (attempt {retry_count + 1}/{max_retries + 1})")
                            
                            # Build error recovery prompt
                            recovery_prompt = build_error_recovery_prompt(
                                original_query=prompt,
                                failed_code=code,
                                error_message=error_msg,
                                data_summary=summary_str,
                                data_structure_info=data_structure,
                                conversation_context=build_conversation_context(st.session_state.messages)
                            )
                            
                            # Generate fixed code
                            recovery_response = ai_client.generate_response(recovery_prompt)
                            
                            if recovery_response["type"] == "code":
                                fixed_code = recovery_response["content"]
                                
                                # Try executing the fixed code
                                st.info("🔄 Executing fixed code...")
                                executor_retry = CodeExecutor(data_dict)
                                output_retry, errors_retry, error_msg_retry, captured_figures_retry = executor_retry.execute(
                                    fixed_code,
                                    capture_figures=True
                                )
                                
                                if error_msg_retry:
                                    # Still failed after retry - show error
                                    st.error(f"❌ Error still occurred after retry:\n```\n{error_msg_retry}\n```")
                                    st.session_state['error_retry_count'] = retry_count + 1
                                    error_handled = False
                                else:
                                    # Success! Clear retry count and display results
                                    st.success("✅ Code fixed and executed successfully!")
                                    st.session_state['error_retry_count'] = 0
                                    
                                    # Display captured figures
                                    if captured_figures_retry:
                                        for fig in captured_figures_retry:
                                            st.pyplot(fig)
                                    
                                    if output_retry:
                                        st.caption(f"Output: {output_retry}")
                                    if errors_retry:
                                        st.caption(f"Warnings: {errors_retry}")
                                    
                                    # Store successful message
                                    message_data = {
                                        "role": "assistant",
                                        "type": "plot",
                                        "content": "Visualization generated (fixed after error)",
                                        "code": fixed_code
                                    }
                                    st.session_state.messages.append(message_data)
                                    
                                    # Skip the error display below - use a flag instead of continue
                                    error_handled = True
                            else:
                                st.error("❌ Could not generate fixed code. Showing original error.")
                                st.session_state['error_retry_count'] = retry_count + 1
                                error_handled = False
                        else:
                            # Max retries reached - show error
                            st.error(f"❌ Error executing code (max retries reached):\n```\n{error_msg}\n```")
                            st.session_state['error_retry_count'] = 0  # Reset for next query
                            error_handled = False
                        
                        # Only show error details if error wasn't successfully handled
                        if not error_handled:
                            # Show error FIRST - make it VERY visible
                            if retry_count >= max_retries:
                                st.error(f"❌ Error executing code:\n```\n{error_msg}\n```")
                            
                            # Store error state for debug panel OUTSIDE chat message
                            st.session_state['last_error'] = {
                                'error_msg': error_msg,
                                'code': code,
                                'executor_debug_info': executor_debug_info,
                                'data_dict_keys': list(data_dict.keys())[:30]
                            }
                            
                            # Additional details in expander
                            with st.expander("📋 Additional Debug Details", expanded=False):
                                st.markdown("### 📋 Full Generated Code")
                                st.code(code, language="python")
                                
                                st.markdown("---")
                                st.markdown("### 🔍 Error Details")
                                st.code(error_msg, language="python")
                                
                                st.markdown("---")
                                st.markdown("### 📦 Data Dictionary Contents")
                                st.write("**Keys in data_dict:**")
                                st.write(list(data_dict.keys())[:30])
                                st.write(f"**Total keys: {len(data_dict)}**")
                                
                                st.write("**output_dir in data_dict:**")
                                st.write(f"- Present: {'output_dir' in data_dict}")
                                if 'output_dir' in data_dict:
                                    st.write(f"- Value: `{data_dict['output_dir']}`")
                                    st.write(f"- Exists: {os.path.exists(data_dict['output_dir']) if data_dict['output_dir'] else False}")
                                
                                st.write("**timestamp in data_dict:**")
                                st.write(f"- Present: {'timestamp' in data_dict}")
                                if 'timestamp' in data_dict:
                                    st.write(f"- Value: `{data_dict['timestamp']}`")
                                
                                st.markdown("---")
                                st.markdown("### 🔎 Code Analysis")
                                
                                # Find all timestamp usage
                                lines = code.split('\n')
                                timestamp_lines = []
                                for i, line in enumerate(lines, 1):
                                    if 'timestamp' in line:
                                        timestamp_lines.append((i, line.strip()))
                                
                                if timestamp_lines:
                                    st.write("**Lines containing 'timestamp':**")
                                    for line_num, line_content in timestamp_lines:
                                        st.code(f"Line {line_num}: {line_content}", language="python")
                                else:
                                    st.warning("⚠️ No lines found containing 'timestamp'")
                                
                                st.markdown("---")
                                st.markdown("### 🗂️ Execution Environment")
                                
                                # Try to get what would be in safe_globals
                                st.write("**Expected variables in execution context:**")
                                expected_vars = [
                                    'timestamp', 'output_dir', 'pd', 'np', 'plt', 'st', 
                                    'date', 'timedelta', 'datetime', 'os'
                                ]
                                for var in expected_vars:
                                    if var in data_dict:
                                        st.write(f"- ✓ `{var}`: Present in data_dict")
                                    elif var == 'timestamp':
                                        st.write(f"- ✗ `{var}`: **MISSING** (should be added by CodeExecutor)")
                                    elif var == 'output_dir':
                                        st.write(f"- {'✓' if 'output_dir' in data_dict else '✗'} `{var}`: {'Present' if 'output_dir' in data_dict else 'MISSING'}")
                                    else:
                                        st.write(f"- ✓ `{var}`: Should be available (built-in/module)")
                                
                                st.markdown("---")
                                st.markdown("### 📊 Raw Records Available")
                                if 'raw_records' in data_dict:
                                    st.write(f"**Number of record types: {len(data_dict['raw_records'])}**")
                                    st.write("**First 10 record types:**")
                                    for i, (rt, df) in enumerate(list(data_dict['raw_records'].items())[:10]):
                                        st.write(f"{i+1}. `{rt}` - {len(df)} records")
                                
                                st.markdown("---")
                                st.markdown("### 🔧 System Info")
                                import platform
                                st.write(f"**Python version:** {platform.python_version()}")
                                st.write(f"**Platform:** {platform.platform()}")
                                
                                st.markdown("---")
                                st.markdown("### 🎯 CodeExecutor Debug Info")
                                if executor_debug_info:
                                    st.write("**Globals before execution:**")
                                    st.json(executor_debug_info.get('globals_before', {}))
                                    
                                    st.write("**Timestamp status:**")
                                    st.write(f"- Set: {executor_debug_info.get('timestamp_set', False)}")
                                    if executor_debug_info.get('timestamp_set'):
                                        st.write(f"- Value: `{executor_debug_info.get('timestamp_value', 'N/A')}`")
                                    
                                    st.write("**Output dir status:**")
                                    st.write(f"- Set: {executor_debug_info.get('output_dir_set', False)}")
                                    if executor_debug_info.get('output_dir_set'):
                                        st.write(f"- Value: `{executor_debug_info.get('output_dir_value', 'N/A')}`")
                                    
                                    if executor_debug_info.get('error_details'):
                                        st.write("**Error details from CodeExecutor:**")
                                        st.json(executor_debug_info['error_details'])
                                else:
                                    st.warning("⚠️ No debug info available from CodeExecutor")
                                
                                # Show the actual code that would be executed
                                st.markdown("---")
                                st.markdown("### 💻 Code Analysis")
                                st.write("**Code length:**", len(code), "characters")
                                st.write("**Code uses 'timestamp':**", 'timestamp' in code)
                                
                                if 'timestamp' in code:
                                    # Show context around timestamp usage
                                    st.write("**Context around timestamp usage:**")
                                    for i, line in enumerate(lines):
                                        if 'timestamp' in line:
                                            start = max(0, i-2)
                                            end = min(len(lines), i+3)
                                            context = '\n'.join(f"{j+1:4d}: {lines[j]}" for j in range(start, end))
                                            st.code(context, language="python")
                                            break
                                
                                st.markdown("---")
                                st.markdown("### 📝 Full Error Traceback")
                                st.code(error_msg, language="python")
                            
                            # Store error with full debug info
                            error_message_data = {
                                "role": "assistant",
                                "type": "error",
                                "content": f"Error: {error_msg}",
                                "code": code,  # Store code for reference
                                "debug_info": executor_debug_info,
                                "data_dict_keys": list(data_dict.keys())[:30]
                            }
                            st.session_state.messages.append(error_message_data)
                    else:
                        # Display captured figures
                        if captured_figures:
                            for fig in captured_figures:
                                st.pyplot(fig)
                        
                        if output:
                            st.caption(f"Output: {output}")
                        if errors:
                            st.caption(f"Warnings: {errors}")
                        
                        # Store message with figure and code
                        message_data = {
                            "role": "assistant",
                            "type": "plot",
                            "content": "Visualization generated",
                            "code": code
                        }
                        if captured_figures:
                            message_data["figure"] = captured_figures[0]
                        
                        st.session_state.messages.append(message_data)
                
            except Exception as e:
                import sys
                import traceback
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                print(f"[App] ========== TOP LEVEL EXCEPTION ==========", file=sys.__stderr__)
                print(f"[App] Error: {error_msg}", file=sys.__stderr__)
                print(f"[App] =========================================", file=sys.__stderr__)
                
                st.error(f"❌ Error: {str(e)}")
                
                # Show comprehensive debug even for top-level exceptions
                with st.expander("🐛 COMPREHENSIVE DEBUG INFO - CLICK TO EXPAND", expanded=True):
                    st.markdown("### 🔍 Exception Details")
                    st.code(error_msg, language="python")
                    
                    st.markdown("---")
                    st.markdown("### 📋 Request Info")
                    st.write(f"**User query:** {prompt}")
                    st.write(f"**Response type:** {response.get('type', 'unknown') if 'response' in locals() else 'N/A'}")
                    if 'code' in locals():
                        st.write(f"**Generated code:**")
                        st.code(code, language="python")
                    
                    st.markdown("---")
                    st.markdown("### 📦 Data Dictionary")
                    st.write(f"**Keys:** {list(data_dict.keys())[:20]}")
                    st.write(f"**output_dir:** {'output_dir' in data_dict}")
                    if 'output_dir' in data_dict:
                        st.write(f"**output_dir value:** {data_dict['output_dir']}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "error",
                    "content": error_msg
                })
