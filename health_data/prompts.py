"""
Prompt templates and data structure information generators.
"""
from typing import Dict, Any, List
from collections import defaultdict


def build_conversation_context(conversation_history: List[Dict[str, Any]]) -> str:
    """
    Build conversation context from history.
    IMPORTANT: Only include user requests and text responses, NOT code, to avoid hardcoding patterns.
    
    Args:
        conversation_history: List of message dictionaries
        
    Returns:
        Formatted conversation context string
    """
    if not conversation_history:
        return ""
    
    context = "\n\nPREVIOUS CONVERSATION (for context only - do NOT reuse code patterns):\n"
    for msg in conversation_history[-5:]:  # Last 5 messages
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_type = msg.get("type", "")
        
        if role == "user":
            context += f"User: {content}\n"
        else:
            # Only include text responses, NOT code - to prevent hardcoding patterns
            if msg_type == "text":
                context += f"Assistant: {content}\n"
            elif msg_type == "error":
                # Include errors for context
                error_preview = content[:200] + "..." if len(content) > 200 else content
                context += f"Assistant: [Error occurred: {error_preview}]\n"
            else:
                # For plots/code, just indicate that a visualization was created, but don't include the code
                context += f"Assistant: [Created a visualization based on user request]\n"
    
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

CRITICAL DATE HANDLING RULES:
1. The `date` columns contain Python `datetime.date` objects, NOT pandas Timestamps
2. When filtering by date, convert date strings to date objects: `pd.to_datetime('2024-01-01').date()`
3. When comparing dates, use: `df[df['date'] >= pd.to_datetime('2024-01-01').date()]`
4. To convert date column for plotting: `pd.to_datetime(df['date'])` converts dates to timestamps for matplotlib
5. NEVER directly compare Timestamp with date - always convert one to match the other
6. For date arithmetic: `from datetime import timedelta` then `df['date'] >= (datetime.now().date() - timedelta(days=30))`

USER REQUEST:
{user_query}

IMPORTANT CONTEXT:
- Each request should be handled INDEPENDENTLY based on what the user asks for RIGHT NOW.
- Do NOT reuse code patterns, window sizes, or visualization styles from previous requests unless the user explicitly asks to modify a previous visualization.
- Do NOT add features the user didn't ask for unless explicitly requested.
- If the user asks to modify a previous visualization, you may reference the previous approach, but still generate fresh code based on the current request.
- For NEW requests, generate code from scratch based on the user's current question, NOT based on previous code patterns.
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
12. CRITICAL: Date comparison rules:
    - The 'date' column in DataFrames is typically datetime64[us] (pandas Timestamp)
    - NEVER compare datetime64 columns directly with Python `date` objects
    - To filter by date, convert the comparison date to pandas Timestamp first
    - When calculating relative dates, use: `pd.Timestamp.now() - pd.Timedelta(days=X)` or convert to Timestamp
    - To convert date column for plotting, use: `pd.to_datetime(df['date'])` to convert dates to timestamps for matplotlib

OUTPUT FORMAT:
- If providing a text answer: Start your response with "TEXT_RESPONSE:" followed by your answer
- If generating code: Start your response with "CODE:" followed by the Python code (no markdown, no explanations)

Remember: Start with "TEXT_RESPONSE:" for text answers or "CODE:" for visualization code."""
    
    return prompt

