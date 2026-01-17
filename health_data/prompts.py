"""
Prompt templates and data structure information generators.
"""
from typing import Dict, Any, List
from collections import defaultdict


def build_conversation_context(conversation_history: List[Dict[str, Any]]) -> str:
    """
    Build conversation context from history.
    Includes user requests, text responses, and code for reference when answering follow-up questions.
    
    Args:
        conversation_history: List of message dictionaries
        
    Returns:
        Formatted conversation context string
    """
    if not conversation_history:
        return ""
    
    context = "\n\nPREVIOUS CONVERSATION:\n"
    for msg in conversation_history[-5:]:  # Last 5 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_type = msg.get("type", "")
        code = msg.get("code", "")
        
        if role == "user":
            context += f"User: {content}\n"
        else:
            if msg_type == "text":
                context += f"Assistant: {content}\n"
            elif msg_type == "error":
                # Include errors for context
                error_preview = content[:200] + "..." if len(content) > 200 else content
                context += f"Assistant: [Error occurred: {error_preview}]\n"
                # Include code if available (for debugging context)
                if code:
                    code_preview = code[:500] + "..." if len(code) > 500 else code
                    context += f"  Code that caused error:\n{code_preview}\n"
            elif msg_type == "plot":
                # Include code for reference when user asks follow-up questions
                context += f"Assistant: [Created a visualization based on user request]\n"
                if code:
                    context += f"  Code used:\n```python\n{code}\n```\n"
            else:
                # For other types, include what we can
                context += f"Assistant: {content}\n"
                if code:
                    context += f"  Code used:\n```python\n{code}\n```\n"
    
    return context


def get_data_structure_info(data: Dict[str, Any]) -> str:
    """
    Generate detailed information about data structure for prompts.
    
    Args:
        data: Health data dictionary
        
    Returns:
        Formatted string with data structure information
    """
    info_lines = []
    raw_records = data.get('raw_records', {})
    metadata = data.get('metadata', {})
    
    if not raw_records:
        return "No data available"
    
    info_lines.append("AVAILABLE DATA:")
    info_lines.append("")
    
    # Group by category
    categories = {
        "Quantity": [],
        "Category": [],
        "Workout": [],
        "Other": []
    }
    
    for record_type, df in raw_records.items():
        if "_daily" in record_type:
            continue
        
        if "HKQuantityTypeIdentifier" in record_type:
            categories["Quantity"].append((record_type, df))
        elif "HKCategoryTypeIdentifier" in record_type:
            categories["Category"].append((record_type, df))
        elif "HKWorkoutTypeIdentifier" in record_type:
            categories["Workout"].append((record_type, df))
        else:
            categories["Other"].append((record_type, df))
    
    # Show top data types from each category
    for category, records in categories.items():
        if not records:
            continue
        
        records_sorted = sorted(records, key=lambda x: len(x[1]), reverse=True)[:5]
        
        info_lines.append(f"{category} Data Types (showing top {len(records_sorted)}):")
        for record_type, df in records_sorted:
            info_lines.append(f"  `{record_type}`:")
            info_lines.append(f"    - Records: {len(df):,}")
            info_lines.append(f"    - Columns: {', '.join(df.columns.tolist())}")
            
            if 'date' in df.columns and len(df) > 0:
                info_lines.append(f"    - Date range: {df['date'].min()} to {df['date'].max()}")
            
            if 'value_numeric' in df.columns:
                info_lines.append(f"    - Has numeric values: Yes")
            if 'value' in df.columns:
                info_lines.append(f"    - Has value field: Yes")
            if 'unit' in df.columns:
                unique_units = df['unit'].unique()[:3]
                info_lines.append(f"    - Units: {', '.join([str(u) for u in unique_units])}")
            
            daily_type = f"{record_type}_daily"
            if daily_type in raw_records:
                info_lines.append(f"    - Daily aggregation available: `{daily_type}`")
            
            info_lines.append("")
    
    info_lines.append("DATA ACCESS:")
    info_lines.append("  - All data is in `raw_records` dictionary, keyed by record type")
    info_lines.append("  - Access data using `raw_records['RecordTypeName']`")
    info_lines.append("  - Daily aggregations (if available) have '_daily' suffix")
    info_lines.append("  - Date columns are Python date objects, NOT pandas Timestamps")
    info_lines.append("")
    
    return "\n".join(info_lines)


def build_analysis_prompt(user_query: str, data_summary: str, data_structure_info: str,
                          conversation_context: str = "") -> str:
    """
    Build the complete prompt for Claude API.
    
    Args:
        user_query: User's question or request
        data_summary: Summary of available data
        data_structure_info: Detailed data structure information
        conversation_context: Conversation history context
        
    Returns:
        Complete prompt string
    """
    prompt = f"""You are a helpful data analysis assistant for Apple Health data. The user can ask questions about any health data in their export, and you should either:
1. Provide a direct text answer if it's a simple question
2. Generate Python code to create a visualization if they want to see a plot or chart

DATA SUMMARY:
{data_summary}

DATA STRUCTURE DETAILS:
{data_structure_info}
{conversation_context}

AVAILABLE VARIABLES:
- `raw_records`: Dictionary of all DataFrames, keyed by record type
- Individual DataFrames are also available as variables with sanitized names
- `available_types`: List of all available record type names
- `metadata`: Dictionary with metadata about each record type
- `output_dir`: Directory path where output files can be saved (if available)
- `pd`: pandas module
- `np`: numpy module
- `plt`: matplotlib.pyplot module
- `st`: streamlit module (for displaying results)

COMMON DATA TYPES YOU MIGHT FIND:
- Steps: `raw_records['HKQuantityTypeIdentifierStepCount']` or `qty_StepCount`
- Heart Rate: `raw_records['HKQuantityTypeIdentifierHeartRate']` or `qty_HeartRate`
- Sleep: `raw_records['HKCategoryTypeIdentifierSleepAnalysis']` or `cat_SleepAnalysis`
- Workouts: `raw_records['HKWorkoutTypeIdentifier']` or `workout_*`
- Many others - check the DATA STRUCTURE DETAILS section above for what's available

CRITICAL DATE HANDLING RULES (MUST FOLLOW TO AVOID TypeError):
1. The `date` columns may be Python `date` objects OR pandas `datetime64` (timezone-aware or timezone-naive)
2. ALWAYS normalize dates before comparison to avoid "Invalid comparison between dtype=datetime64[ns, UTC-X] and Timestamp" errors
3. SAFE PATTERN for date filtering (ALWAYS use this pattern):
   ```python
   # Step 1: Normalize the date column to timezone-naive datetime64
   df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
   
   # Step 2: Create cutoff date as timezone-naive Timestamp
   cutoff_date = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=180)
   
   # Step 3: Filter
   df_filtered = df[df['date'] >= cutoff_date]
   ```
4. For "past X months" queries, ALWAYS use this pattern:
   ```python
   # Normalize date column first
   df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
   # Calculate cutoff
   cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=X*30)
   # Filter
   df_filtered = df[df['date'] >= cutoff]
   ```
5. To check if date column is timezone-aware: `str(df['date'].dtype)` contains 'UTC' if timezone-aware
6. NEVER compare timezone-aware datetime64 with timezone-naive Timestamp - normalize first!
7. For plotting: After filtering, convert for plotting: `pd.to_datetime(df_filtered['date'])` if needed

USER REQUEST:
{user_query}

IMPORTANT CONTEXT:
- When the user asks a NEW question, handle it independently based on what they ask for RIGHT NOW.
- When the user asks FOLLOW-UP questions about a previous visualization (e.g., "why did you use X?", "can you change Y?", "what does Z mean?"), reference the previous code from the conversation history to answer their question or make the requested modification.
- Do NOT add features the user didn't ask for unless explicitly requested.
- If the user asks to modify a previous visualization, reference the previous code and modify it based on their request.
- For NEW requests (not follow-ups), generate code from scratch based on the user's current question.
- Generate ONLY what the user explicitly requests - nothing more, nothing less.

INSTRUCTIONS:
Decide if the user wants:
- A TEXT ANSWER: If they're asking a simple question (statistics, counts, averages, etc.), provide a clear, concise text answer directly.
- A VISUALIZATION: If they want to see a plot, chart, or graph, generate Python code.

For TEXT ANSWERS:
- Provide a clear, helpful answer based on the data
- Use specific numbers and insights
- Be conversational and friendly

For VISUALIZATIONS (when generating code):
1. CRITICAL: Generate ONLY what the user explicitly asks for. Do NOT add extra features like rolling averages, trend lines, or additional visualizations unless the user specifically requests them.
2. Generate Python code that creates visualizations using matplotlib (plt)
3. IMPORTANT: Always create a figure explicitly: `fig, ax = plt.subplots(...)` or `fig = plt.figure(...)`
4. IMPORTANT: After creating the plot, use `st.pyplot(fig)` to display it in Streamlit
5. IMPORTANT: Save the figure to the output directory with a descriptive filename:
   - Use `output_dir` variable if available (it contains the path to the output folder)
   - Generate a descriptive filename based on the visualization content
   - The `timestamp` variable is already available (pre-defined), so you can use it directly: `f"{{descriptive_name}}_{{timestamp}}.png"`
   - Save with: `fig.savefig(os.path.join(output_dir, f"{{descriptive_name}}_{{timestamp}}.png"), dpi=150, bbox_inches='tight')`
6. IMPORTANT: Use `plt.close(fig)` after displaying and saving to free memory
7. The code should be complete and executable
8. Include proper labels, titles, and formatting
9. Handle edge cases (empty data, missing columns, etc.)
10. Do NOT import modules (they're already imported, but you can use `from datetime import date, timedelta, datetime`)
11. File I/O is ONLY allowed for saving plots to `output_dir` using `fig.savefig()`
12. Focus on creating clear, informative visualizations that match EXACTLY what the user requested
12. CRITICAL: Date comparison rules (MUST FOLLOW TO AVOID ERRORS):
    - The 'date' column may be datetime64 (timezone-aware or timezone-naive) or Python date objects
    - ALWAYS normalize timezone-aware columns before comparison: `df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)`
    - For date filtering, ensure both sides are timezone-naive:
      * `cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=180)`
      * `df_filtered = df[pd.to_datetime(df['date']).dt.tz_localize(None) >= cutoff]`
    - NEVER compare timezone-aware datetime64 with timezone-naive Timestamp - normalize first!
    - For "past X months" queries: `cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=X*30)`
    - Check if timezone-aware: `df['date'].dtype` shows `datetime64[ns, UTC-X]` if timezone-aware

OUTPUT FORMAT:
- If providing a text answer: Start your response with "TEXT_RESPONSE:" followed by your answer
- If generating code: Start your response with "CODE:" followed by the Python code (no markdown, no explanations)

Remember: Start with "TEXT_RESPONSE:" for text answers or "CODE:" for visualization code."""
    
    return prompt

