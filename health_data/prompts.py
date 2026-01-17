"""
Prompt templates and data structure information generators.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def generate_table_summary(record_type: str, df, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive summary for a single table.
    This summary is saved to cache and used for table selection.
    
    Args:
        record_type: The record type identifier
        df: The DataFrame for this record type
        metadata: Optional metadata dictionary
        
    Returns:
        Dictionary with table summary information
    """
    summary = {
        "record_type": record_type,
        "record_count": len(df),
        "columns": df.columns.tolist(),
        "column_count": len(df.columns),
    }
    
    # Add description if we can generate one
    summary["description"] = get_record_type_description(record_type)
    
    # Date information
    if 'date' in df.columns and len(df) > 0:
        try:
            date_min = df['date'].min()
            date_max = df['date'].max()
            summary["date_range"] = {
                "min": str(date_min),
                "max": str(date_max),
                "has_dates": True
            }
        except:
            summary["date_range"] = {"has_dates": False}
    else:
        summary["date_range"] = {"has_dates": False}
    
    # Value information
    if 'value_numeric' in df.columns:
        try:
            non_null = df['value_numeric'].notna().sum()
            if non_null > 0:
                summary["value_numeric"] = {
                    "has_values": True,
                    "non_null_count": int(non_null),
                    "min": float(df['value_numeric'].min()),
                    "max": float(df['value_numeric'].max()),
                    "mean": float(df['value_numeric'].mean()),
                    "median": float(df['value_numeric'].median())
                }
            else:
                summary["value_numeric"] = {"has_values": False}
        except:
            summary["value_numeric"] = {"has_values": False}
    else:
        summary["value_numeric"] = {"has_values": False}
    
    if 'value' in df.columns:
        summary["has_value_field"] = True
    else:
        summary["has_value_field"] = False
    
    # Units
    if 'unit' in df.columns:
        try:
            unique_units = df['unit'].dropna().unique().tolist()[:10]
            summary["units"] = [str(u) for u in unique_units]
            summary["unit_count"] = len(unique_units)
        except:
            summary["units"] = []
            summary["unit_count"] = 0
    else:
        summary["units"] = []
        summary["unit_count"] = 0
    
    # Add metadata if available
    if metadata:
        summary["metadata"] = {
            "has_value": metadata.get("has_value", False),
            "has_unit": metadata.get("has_unit", False),
            "has_startDate": metadata.get("has_startDate", False),
            "has_endDate": metadata.get("has_endDate", False),
            "attributes": metadata.get("attributes", [])
        }
    
    return summary


def get_record_type_description(record_type: str) -> str:
    """
    Generate a human-readable description for an Apple Health record type.
    
    Args:
        record_type: The record type identifier (e.g., HKQuantityTypeIdentifierStepCount)
        
    Returns:
        Human-readable description
    """
    # Common Apple Health record type descriptions
    descriptions = {
        # Quantity types
        "HKQuantityTypeIdentifierStepCount": "Daily step count - number of steps taken",
        "HKQuantityTypeIdentifierDistanceWalkingRunning": "Distance walked or run - total distance in meters",
        "HKQuantityTypeIdentifierFlightsClimbed": "Number of flights of stairs climbed",
        "HKQuantityTypeIdentifierHeartRate": "Heart rate measurements - beats per minute",
        "HKQuantityTypeIdentifierActiveEnergyBurned": "Active energy burned during exercise - calories",
        "HKQuantityTypeIdentifierBasalEnergyBurned": "Basal metabolic rate energy - calories at rest",
        "HKQuantityTypeIdentifierAppleExerciseTime": "Exercise time - minutes of exercise",
        "HKQuantityTypeIdentifierBodyMass": "Body weight - mass in kg",
        "HKQuantityTypeIdentifierHeight": "Height measurements - height in meters",
        "HKQuantityTypeIdentifierBodyMassIndex": "Body Mass Index (BMI) - calculated from weight and height",
        "HKQuantityTypeIdentifierBodyFatPercentage": "Body fat percentage",
        "HKQuantityTypeIdentifierLeanBodyMass": "Lean body mass - weight excluding fat",
        "HKQuantityTypeIdentifierRespiratoryRate": "Respiratory rate - breaths per minute",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "Heart rate variability - standard deviation",
        "HKQuantityTypeIdentifierDietaryWater": "Water intake - volume in liters",
        "HKQuantityTypeIdentifierDietaryEnergyConsumed": "Dietary calories consumed",
        "HKQuantityTypeIdentifierDietaryFatTotal": "Total dietary fat consumed - grams",
        "HKQuantityTypeIdentifierDietaryFatSaturated": "Saturated fat consumed - grams",
        "HKQuantityTypeIdentifierDietaryCholesterol": "Cholesterol consumed - milligrams",
        "HKQuantityTypeIdentifierDietarySodium": "Sodium consumed - milligrams",
        "HKQuantityTypeIdentifierDietaryCarbohydrates": "Carbohydrates consumed - grams",
        "HKQuantityTypeIdentifierDietaryFiber": "Dietary fiber consumed - grams",
        "HKQuantityTypeIdentifierDietarySugar": "Sugar consumed - grams",
        "HKQuantityTypeIdentifierBloodPressureSystolic": "Systolic blood pressure - mmHg",
        "HKQuantityTypeIdentifierBloodPressureDiastolic": "Diastolic blood pressure - mmHg",
        "HKQuantityTypeIdentifierBloodGlucose": "Blood glucose level - mg/dL",
        "HKQuantityTypeIdentifierOxygenSaturation": "Blood oxygen saturation - percentage",
        "HKQuantityTypeIdentifierBodyTemperature": "Body temperature - degrees Celsius",
        "HKQuantityTypeIdentifierWalkingSpeed": "Walking speed - meters per second",
        "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage": "Walking double support percentage",
        "HKQuantityTypeIdentifierWalkingAsymmetryPercentage": "Walking asymmetry percentage",
        "HKQuantityTypeIdentifierWalkingStepLength": "Walking step length - meters",
        "HKQuantityTypeIdentifierSixMinuteWalkTestDistance": "Six minute walk test distance - meters",
        "HKQuantityTypeIdentifierStairSpeedDown": "Stair descent speed - meters per second",
        "HKQuantityTypeIdentifierStairSpeedUp": "Stair ascent speed - meters per second",
        
        # Category types
        "HKCategoryTypeIdentifierSleepAnalysis": "Sleep analysis - sleep stages (asleep, awake, in bed)",
        "HKCategoryTypeIdentifierAppleStandHour": "Stand hours - whether user stood for at least 1 minute in the hour",
        "HKCategoryTypeIdentifierHighHeartRateEvent": "High heart rate events - episodes of elevated heart rate",
        "HKCategoryTypeIdentifierLowHeartRateEvent": "Low heart rate events - episodes of low heart rate",
        "HKCategoryTypeIdentifierIrregularHeartRhythmEvent": "Irregular heart rhythm events - detected arrhythmias",
        "HKCategoryTypeIdentifierMindfulSession": "Mindful session - meditation or mindfulness activities",
        "HKCategoryTypeIdentifierHeadphoneAudioExposureEvent": "Headphone audio exposure events - loud sound warnings",
        
        # Workout types
        "HKWorkoutTypeIdentifier": "Workout data - various workout types with duration, energy, distance",
    }
    
    # Check for exact match
    if record_type in descriptions:
        return descriptions[record_type]
    
    # Generate description from identifier name
    if "HKQuantityTypeIdentifier" in record_type:
        # Extract the meaningful part
        name = record_type.replace("HKQuantityTypeIdentifier", "")
        # Convert camelCase to readable text
        readable = ''.join([' ' + c if c.isupper() else c for c in name]).strip()
        return f"Quantity measurement: {readable.lower()}"
    elif "HKCategoryTypeIdentifier" in record_type:
        name = record_type.replace("HKCategoryTypeIdentifier", "")
        readable = ''.join([' ' + c if c.isupper() else c for c in name]).strip()
        return f"Category data: {readable.lower()}"
    elif "HKWorkoutTypeIdentifier" in record_type:
        return "Workout data - exercise sessions with duration, energy burned, and distance"
    else:
        return f"Health data: {record_type}"


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
            elif msg_type == "reasoning":
                # Include reasoning for context
                context += f"Assistant: [Reasoning about tables to use]\n{content}\n"
            else:
                # For other types, include what we can
                context += f"Assistant: {content}\n"
                if code:
                    context += f"  Code used:\n```python\n{code}\n```\n"
    
    return context


def get_data_structure_info(data: Dict[str, Any]) -> str:
    """
    Generate comprehensive information about ALL data tables for prompts.
    Includes description, record count, structure, and columns for every table.
    
    Args:
        data: Health data dictionary
        
    Returns:
        Formatted string with complete data structure information
    """
    info_lines = []
    raw_records = data.get('raw_records', {})
    metadata = data.get('metadata', {})
    
    if not raw_records:
        return "No data available"
    
    info_lines.append("=" * 80)
    info_lines.append("COMPLETE DATA CATALOG - ALL AVAILABLE TABLES")
    info_lines.append("=" * 80)
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
    
    # Show ALL data types from each category (not just top 5)
    for category, records in categories.items():
        if not records:
            continue
        
        records_sorted = sorted(records, key=lambda x: len(x[1]), reverse=True)
        
        info_lines.append(f"{category} Data Types ({len(records_sorted)} tables):")
        info_lines.append("-" * 80)
        
        for record_type, df in records_sorted:
            # Get description
            description = get_record_type_description(record_type)
            
            info_lines.append(f"\nTable: `{record_type}`")
            info_lines.append(f"  Description: {description}")
            info_lines.append(f"  Record Count: {len(df):,}")
            
            # Column information
            columns = df.columns.tolist()
            info_lines.append(f"  Columns ({len(columns)}): {', '.join(columns)}")
            
            # Date range
            if 'date' in df.columns and len(df) > 0:
                try:
                    date_min = df['date'].min()
                    date_max = df['date'].max()
                    info_lines.append(f"  Date Range: {date_min} to {date_max}")
                except:
                    pass
            
            # Value information
            value_info = []
            if 'value_numeric' in df.columns:
                try:
                    non_null = df['value_numeric'].notna().sum()
                    if non_null > 0:
                        value_min = df['value_numeric'].min()
                        value_max = df['value_numeric'].max()
                        value_mean = df['value_numeric'].mean()
                        value_info.append(f"Numeric values: {non_null:,} records, range [{value_min:.2f}, {value_max:.2f}], mean {value_mean:.2f}")
                except:
                    value_info.append("Has numeric values")
            if 'value' in df.columns:
                value_info.append("Has value field")
            if value_info:
                info_lines.append(f"  Values: {'; '.join(value_info)}")
            
            # Units
            if 'unit' in df.columns:
                try:
                    unique_units = df['unit'].dropna().unique()[:5]
                    if len(unique_units) > 0:
                        info_lines.append(f"  Units: {', '.join([str(u) for u in unique_units])}")
                except:
                    pass
            
            # Daily aggregation
            daily_type = f"{record_type}_daily"
            if daily_type in raw_records:
                daily_df = raw_records[daily_type]
                info_lines.append(f"  Daily Aggregation: Available (`{daily_type}`) with {len(daily_df):,} daily records")
            
            # Sample data summary
            if len(df) > 0:
                try:
                    sample_info = []
                    if 'value_numeric' in df.columns:
                        sample_info.append(f"value_numeric: {df['value_numeric'].notna().sum():,} non-null")
                    if 'date' in df.columns:
                        sample_info.append(f"date: {df['date'].notna().sum():,} non-null")
                    if sample_info:
                        info_lines.append(f"  Summary: {', '.join(sample_info)}")
                except:
                    pass
            
            info_lines.append("")
    
    info_lines.append("=" * 80)
    info_lines.append("DATA ACCESS INSTRUCTIONS:")
    info_lines.append("=" * 80)
    info_lines.append("  - All data is in `raw_records` dictionary, keyed by record type")
    info_lines.append("  - Access data using: `raw_records['RecordTypeName']`")
    info_lines.append("  - Daily aggregations (if available) have '_daily' suffix")
    info_lines.append("  - Date columns are Python date objects, NOT pandas Timestamps")
    info_lines.append("  - Example: `df = raw_records['HKQuantityTypeIdentifierStepCount']`")
    info_lines.append("")
    
    return "\n".join(info_lines)


def load_table_summaries(table_summaries_file: str) -> Dict[str, Any]:
    """
    Load all table summaries from a single JSON file.
    
    Args:
        table_summaries_file: Path to the table_summaries.json file
        
    Returns:
        Dictionary mapping record_type to summary dictionary
    """
    import sys
    summaries_file = Path(table_summaries_file)
    
    if not summaries_file.exists():
        return {}
    
    try:
        with open(summaries_file, 'r') as f:
            summaries = json.load(f)
            # Ensure it's a dictionary (should be already, but handle edge cases)
            if isinstance(summaries, dict):
                return summaries
            else:
                return {}
    except Exception as e:
        print(f"[Prompts] Warning: Could not load table summaries from {table_summaries_file}: {e}", file=sys.stderr)
        return {}


def build_reasoning_prompt(user_query: str, table_summaries: Dict[str, Any],
                          conversation_context: str = "") -> str:
    """
    Build prompt for STEP 1: Reasoning about which tables to use.
    
    Args:
        user_query: User's question or request
        table_summaries: Dictionary of all table summaries
        conversation_context: Conversation history context
        
    Returns:
        Reasoning prompt string
    """
    # Format table summaries as JSON for the LLM
    summaries_json = json.dumps(list(table_summaries.values()), indent=2, default=str)
    
    prompt = f"""You are analyzing a user's request about their Apple Health data. Your task is to identify which data tables are needed.

USER REQUEST:
{user_query}
{conversation_context}

AVAILABLE TABLES:
The following JSON array contains summaries of ALL available tables. Each table has:
- record_type: The table identifier (this is the key to access the table in raw_records)
- description: What the table contains (human-readable text description)
- record_count: Number of records in the table
- columns: List of column names
- column_types: Dictionary mapping column names to their data types (integer, float, datetime, date, string, boolean)
- example_rows: Array of 2-3 example rows showing actual data values
- date_range: Date range if available (min, max dates)
- value_numeric: Statistics about numeric values if available (min, max, mean, median)
- units: Available units if applicable
- has_value_field: Whether it has a value field

{summaries_json}

TASK:
1. FIRST: Determine if this is a TEXT QUESTION or requires CODE GENERATION
   - TEXT QUESTION: User is asking "how", "why", "what does X mean", "explain", or questions about methodology
   - CODE GENERATION: User wants a visualization, plot, chart, or data analysis that requires code
   - If user asks about previous code/visualizations, check conversation context for the code

2. If TEXT QUESTION:
   - Respond with "TEXT_RESPONSE:" followed by your answer
   - You can reference previous code from conversation context if relevant
   - Provide a clear, helpful explanation

3. If CODE GENERATION:
   - Identify which specific table(s) contain the data needed
   - Consider:
     * What type of data is requested? (steps, heart rate, sleep, workouts, etc.)
     * Are daily aggregations available that would be more efficient?
     * Do you need multiple tables for correlation or comparison?
     * What columns are needed from each table?
   - List the exact table name(s) you will use
   - Explain your reasoning

OUTPUT FORMAT:
- For TEXT QUESTIONS: Start with "TEXT_RESPONSE:" followed by your answer
- For CODE GENERATION: Start with "REASONING:" followed by:
  * List of table names you will use (exact record_type values)
  * Brief explanation of why each table is needed
  * Which columns you'll need from each table
  * Whether you'll use daily aggregations or raw data

Example TEXT RESPONSE:
TEXT_RESPONSE:
Looking at the previous code, total sleep time per day is calculated by summing all sleep duration records for each day. The code groups sleep records by date and sums the duration. If you're seeing values over 15-20 hours, it might be because:
1. Multiple sleep sessions in one day (naps + main sleep)
2. Sleep records overlapping or incorrectly timestamped
3. The calculation includes "in bed" time, not just "asleep" time

Example CODE GENERATION:
REASONING:
I will use the following tables:
1. HKQuantityTypeIdentifierStepCount - Contains daily step count data
2. HKQuantityTypeIdentifierStepCount_daily - Daily aggregation for efficiency

Reasoning: The user wants step count data over time. The _daily version is more efficient for time-series analysis.
Columns needed: date, value_numeric (or total from daily aggregation)
"""
    
    return prompt


def build_code_generation_prompt(user_query: str, selected_tables: List[str],
                                table_summaries: Dict[str, Any],
                                conversation_context: str = "") -> str:
    """
    Build prompt for STEP 2: Code generation using selected tables.
    
    Args:
        user_query: User's question or request
        selected_tables: List of table names selected in reasoning step
        table_summaries: Dictionary of all table summaries (for reference)
        conversation_context: Conversation history context
        
    Returns:
        Code generation prompt string
    """
    # Get detailed info for selected tables
    selected_table_info = []
    for table_name in selected_tables:
        if table_name in table_summaries:
            summary = table_summaries[table_name]
            selected_table_info.append(json.dumps(summary, indent=2, default=str))
    
    selected_info_str = "\n\n".join(selected_table_info)
    
    prompt = f"""Generate Python code to fulfill the user's request using the selected tables.

USER REQUEST:
{user_query}
{conversation_context}

SELECTED TABLES TO USE:
Each table summary includes:
- record_type: The exact key to use in raw_records['record_type']
- columns: List of available column names
- column_types: Data types for each column (use this to understand what operations are safe)
- example_rows: Sample data showing actual values and structure
- description: What the table contains
- date_range: Available date range
- value_numeric: Statistics if numeric values are present

{selected_info_str}

AVAILABLE VARIABLES:
- `raw_records`: Dictionary of all DataFrames, keyed by record type
- Access selected tables using: `raw_records['TableName']`
- `output_dir`: Directory path where output files can be saved
- `timestamp`: Pre-defined timestamp string for file naming
- `pd`: pandas module
- `np`: numpy module
- `plt`: matplotlib.pyplot module
- `st`: streamlit module (for displaying results)

CRITICAL DATE HANDLING RULES (MUST FOLLOW TO AVOID TypeError):
1. The `date` columns may be Python `date` objects OR pandas `datetime64` (timezone-aware or timezone-naive)
2. ALWAYS normalize dates before comparison: `df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)`
3. For date filtering, ensure both sides are timezone-naive:
   ```python
   df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
   cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=180)
   df_filtered = df[df['date'] >= cutoff]
   ```

CRITICAL: Pandas Data Type Rules (MUST FOLLOW TO AVOID AttributeError):
- NEVER call datetime accessor methods (`.dt.day_name()`, `.dt.month`, etc.) on Period columns
- Period objects don't support `.dt.day_name()` - convert to datetime first or keep as datetime
- To get unique days: `df['date'].dt.normalize().drop_duplicates()` or `df['date'].dt.date.drop_duplicates()`

INSTRUCTIONS:
- DECIDE FIRST: Is this a TEXT QUESTION or requires CODE GENERATION?
  * TEXT QUESTION: User is asking "how", "why", "what does X mean", "explain", or questions about methodology
  * CODE GENERATION: User wants a visualization, plot, chart, or data analysis

- If TEXT QUESTION:
  * Provide a clear, helpful answer based on the conversation context and previous code
  * Reference specific code from conversation context if explaining how something was calculated
  * Use "TEXT_RESPONSE:" format

- If CODE GENERATION:
  * Generate Python code that accesses the selected tables from `raw_records`
  * Create visualizations using matplotlib (plt)
  * Always create figure explicitly: `fig, ax = plt.subplots(...)`
  * Use `st.pyplot(fig)` to display in Streamlit
  * Save figure: `fig.savefig(os.path.join(output_dir, f"name_{{timestamp}}.png"), dpi=150, bbox_inches='tight')`
  * Use `plt.close(fig)` after displaying and saving
  * Handle edge cases (empty data, missing columns, etc.)
  * Generate ONLY what the user explicitly requests

OUTPUT FORMAT:
- For TEXT ANSWERS: Start with "TEXT_RESPONSE:" followed by your answer
- For CODE GENERATION: Start with "CODE:" followed by the Python code (no markdown, no explanations)
"""
    
    return prompt


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

AGENTIC WORKFLOW - FOLLOW THESE STEPS:

STEP 1: REASONING - Identify which tables to use
Before generating any code, you MUST first reason about which tables are needed:
1. Review the COMPLETE DATA CATALOG above
2. Identify which specific table(s) contain the data needed for this query
3. Consider:
   - What type of data is being requested? (steps, heart rate, sleep, workouts, etc.)
   - Are there daily aggregations available that would be more efficient?
   - Do you need multiple tables? (e.g., steps + heart rate for correlation)
   - What columns are needed from each table?
4. State your reasoning clearly before generating code

STEP 2: CODE GENERATION
After identifying the tables, generate the Python code to:
- Access the correct table(s) from `raw_records`
- Filter/process the data appropriately
- Create the requested visualization or answer

OUTPUT FORMAT:
- For TEXT ANSWERS: Start with "TEXT_RESPONSE:" followed by your answer
- For VISUALIZATIONS: Start with "REASONING:" then "CODE:" followed by Python code
  Example:
  REASONING:
  I need to use the HKQuantityTypeIdentifierStepCount table because it contains daily step counts.
  The _daily version would be more efficient, but the user asked for per-day data, so I'll use the main table.
  I'll filter for the last 6 months and create a dot plot with a 7-day rolling average.
  
  CODE:
  [Python code here]

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

13. CRITICAL: Pandas Data Type Rules (MUST FOLLOW TO AVOID AttributeError):
    - ALWAYS check what data type you're working with: `df['column'].dtype` or `type(df['column'].iloc[0])`
    - Datetime columns support: `.dt.day_name()`, `.dt.dayofweek`, `.dt.month`, `.dt.year`, `.dt.date`, `.dt.to_period()`, etc.
    - Period columns (created with `.dt.to_period()`) are DIFFERENT from Datetime:
      * Period objects have LIMITED methods - they do NOT support `.dt.day_name()`, `.dt.dayofweek`, `.dt.month`, etc.
      * Period objects support: `.dt.start_time`, `.dt.end_time`, `.dt.to_timestamp()`
      * If you need datetime methods, convert Period back: `df['period_col'].dt.to_timestamp()` or keep as datetime
    - To get unique days without converting to Period: `df['date'].dt.normalize().drop_duplicates()` or `df['date'].dt.date.drop_duplicates()`
    - NEVER call datetime accessor methods (`.dt.day_name()`, `.dt.month`, etc.) on Period columns - convert to datetime first
    - When in doubt, keep columns as datetime64 and avoid `.to_period()` unless specifically needed for period-based operations
    - Common mistake: `df['date'].dt.to_period('D').dt.day_name()` - this FAILS because Period doesn't have day_name()
    - Correct approach: `df['date'].dt.day_name()` (keep as datetime) OR `df['date'].dt.to_period('D').dt.to_timestamp().dt.day_name()` (convert back)
    - For grouping by day of week: `df.groupby(df['date'].dt.day_name())` - no need to convert to Period
    - For getting unique dates: `df['date'].drop_duplicates()` or `df['date'].dt.date.drop_duplicates()` - no need for Period

14. GENERAL PANDAS BEST PRACTICES:
    - Always verify column exists: `if 'column_name' in df.columns:`
    - Check for empty dataframes: `if len(df) == 0: return` or handle gracefully
    - Use `.copy()` when modifying filtered dataframes to avoid SettingWithCopyWarning
    - For aggregations, prefer `.groupby()` over manual loops
    - When extracting datetime components, use `.dt` accessor on datetime columns, not on Period columns
    - Test your logic step-by-step: if you convert to Period, verify what methods are available before using them
    - If you need both period operations AND datetime methods, keep the original datetime column and create a separate period column

OUTPUT FORMAT:
- If providing a text answer: Start your response with "TEXT_RESPONSE:" followed by your answer
- If generating code: Start your response with "CODE:" followed by the Python code (no markdown, no explanations)

Remember: Start with "TEXT_RESPONSE:" for text answers or "CODE:" for visualization code."""
    
    return prompt


def build_error_recovery_prompt(original_query: str, failed_code: str, error_message: str,
                                data_summary: str = "", data_structure_info: str = "",
                                selected_tables: List[str] = None,
                                table_summaries: Dict[str, Any] = None,
                                conversation_context: str = "") -> str:
    """
    Build a prompt for error recovery - asking the LLM to fix the code.
    
    Args:
        original_query: The original user query
        failed_code: The code that failed to execute
        error_message: The error message from execution
        data_summary: Summary of available data (optional, for fallback)
        data_structure_info: Detailed data structure information (optional, for fallback)
        selected_tables: List of tables that were selected (if using two-step workflow)
        table_summaries: Dictionary of table summaries (if using two-step workflow)
        conversation_context: Conversation history context
        
    Returns:
        Complete error recovery prompt string
    """
    # Use table summaries if available (two-step workflow), otherwise use old format
    if selected_tables and table_summaries:
        # Use table summaries approach
        selected_table_info = []
        for table_name in selected_tables:
            if table_name in table_summaries:
                summary = table_summaries[table_name]
                selected_table_info.append(json.dumps(summary, indent=2, default=str))
        
        selected_info_str = "\n\n".join(selected_table_info) if selected_table_info else "No table information available"
        
        prompt = f"""You previously generated code that failed to execute. Please fix the code and try again.

ORIGINAL USER REQUEST:
{original_query}

FAILED CODE:
```python
{failed_code}
```

ERROR MESSAGE:
{error_message}

SELECTED TABLES (that should be used):
{selected_info_str}
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
- `timestamp`: Pre-defined timestamp string for file naming

CRITICAL: Analyze the error carefully:
1. Check if you're accessing columns that don't exist
2. Verify data types match what you expect (check with `df['column'].dtype`)
3. Ensure you're not mixing incompatible types (e.g., dividing Series by dict, Period vs Datetime)
4. Check if dataframes are empty before operations
5. Verify variable names match what's actually available
6. Review the error message - it tells you exactly what went wrong

INSTRUCTIONS:
- Fix the code based on the error message
- Generate corrected Python code that addresses the specific error
- Keep the same overall approach unless the error indicates a fundamental issue
- Test your logic: if dividing, ensure both sides are numeric; if accessing columns, verify they exist
- Generate ONLY the corrected code - start with "CODE:" followed by the fixed Python code

OUTPUT FORMAT:
- Start your response with "CODE:" followed by the corrected Python code (no markdown, no explanations)
- The code should be complete and executable
- Address the specific error mentioned above"""
    else:
        # Fallback to old format
        prompt = f"""You previously generated code that failed to execute. Please fix the code and try again.

ORIGINAL USER REQUEST:
{original_query}

FAILED CODE:
```python
{failed_code}
```

ERROR MESSAGE:
{error_message}

DATA SUMMARY:
{data_summary}

DATA STRUCTURE DETAILS:
{data_structure_info}
{conversation_context}

AVAILABLE VARIABLES:
- `raw_records`: Dictionary of all DataFrames, keyed by record type
- Access data using `raw_records['RecordTypeName']`
- `output_dir`: Directory path where output files can be saved (if available)
- `pd`: pandas module
- `np`: numpy module
- `plt`: matplotlib.pyplot module
- `st`: streamlit module (for displaying results)
- `timestamp`: Pre-defined timestamp string for file naming

CRITICAL: Analyze the error carefully:
1. Check if you're accessing columns that don't exist
2. Verify data types match what you expect (check with `df['column'].dtype`)
3. Ensure you're not mixing incompatible types (e.g., dividing Series by dict, Period vs Datetime)
4. Check if dataframes are empty before operations
5. Verify variable names match what's actually available
6. Review the error message - it tells you exactly what went wrong

INSTRUCTIONS:
- Fix the code based on the error message
- Generate corrected Python code that addresses the specific error
- Keep the same overall approach unless the error indicates a fundamental issue
- Test your logic: if dividing, ensure both sides are numeric; if accessing columns, verify they exist
- Generate ONLY the corrected code - start with "CODE:" followed by the fixed Python code

OUTPUT FORMAT:
- Start your response with "CODE:" followed by the corrected Python code (no markdown, no explanations)
- The code should be complete and executable
- Address the specific error mentioned above"""
    
    return prompt

