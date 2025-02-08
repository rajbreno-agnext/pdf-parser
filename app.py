import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
import json
import pandas as pd
from pathlib import Path
import tempfile
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from openpyxl import Workbook
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini model configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

def initialize_gemini_model():
    """Initialize and return the Gemini model"""
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=GENERATION_CONFIG,
    )

def split_pdf_to_pages(pdf_file):
    """Split PDF into individual pages and return their paths"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)
    page_paths = []
    
    # Create temporary directory for pages
    temp_dir = tempfile.mkdtemp()
    
    for page_num in range(total_pages):
        # Create a new PDF writer for this page
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])
        
        # Save the page to a temporary file
        page_path = os.path.join(temp_dir, f'page_{page_num + 1}.pdf')
        with open(page_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        page_paths.append(page_path)
    
    return page_paths, total_pages

async def process_page(page_path, model, page_num, progress_bar, status_text):
    """Process a single PDF page"""
    try:
        # Upload to Gemini
        gemini_file = genai.upload_file(page_path, mime_type="application/pdf")
        
        # Start chat session
        chat = model.start_chat()
        response = chat.send_message([
            gemini_file,
            """Parse this PDF page into structured JSON format. Follow these guidelines:
            1. Organize data into clear tables with consistent column names
            2. Each table should have:
               - A clear table name
               - Consistent column headers
               - Rows with corresponding values
            3. Use simple data types (strings, numbers) for values
            4. Maintain data in a tabular format
            5. Ensure all related data is grouped together
            
            Return the data in this structure:
            {
                "data": [
                    {
                        "table": "Table Name",
                        "headers": ["column1", "column2", ...],
                        "rows": [
                            {"column1": "value1", "column2": "value2", ...},
                            ...
                        ]
                    },
                    ...
                ]
            }"""
        ])
        
        # Update progress
        if progress_bar and status_text:
            progress_bar.progress((page_num + 1) / progress_bar.total_pages)
            status_text.text(f"Processed page {page_num + 1}/{progress_bar.total_pages}")
        
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Error processing page {page_num + 1}: {str(e)}")
        return None

async def process_pages_parallel(page_paths, model, progress_bar, status_text):
    """Process multiple pages in parallel"""
    tasks = []
    for i, page_path in enumerate(page_paths):
        task = asyncio.create_task(process_page(page_path, model, i, progress_bar, status_text))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

async def merge_with_gemini(results, model):
    """Merge JSON results using Gemini API"""
    try:
        # Convert results to a formatted string
        json_str = json.dumps(results, indent=2)
        
        # Create prompt for merging
        prompt = f"""
        Merge these JSON results from different pages into a single coherent JSON structure.
        Follow these guidelines strictly:
        1. Create a consistent table structure
        2. Each table should have:
           - A clear table name
           - Consistent column names across rows
           - Simple data types (strings, numbers) for values
        3. Return the data in this exact format:
        {{
            "data": [
                {{
                    "table": "Table Name",
                    "rows": [
                        {{"column1": "value1", "column2": "value2", ...}},
                        {{"column1": "value3", "column2": "value4", ...}}
                    ]
                }}
            ]
        }}
        
        Input JSON array:
        {json_str}
        """
        
        # Get merged result from Gemini
        chat = model.start_chat()
        response = chat.send_message(prompt)
        
        # Parse and return the merged JSON
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Error merging results: {str(e)}")
        return None

def convert_to_csv(json_data):
    """Convert JSON data to CSV format"""
    try:
        # Initialize empty DataFrame
        df = None
        
        # Case 1: If we have a samples list directly
        if isinstance(json_data, dict) and 'samples' in json_data:
            samples = json_data['samples']
            if isinstance(samples, list):
                # Convert each sample to a dictionary if it's a string
                processed_samples = []
                for sample in samples:
                    if isinstance(sample, str):
                        # Try to safely evaluate the string to a dictionary
                        try:
                            # Remove any single quotes around dictionary keys
                            sample = sample.replace("'", '"')
                            sample_dict = json.loads(sample)
                            processed_samples.append(sample_dict)
                        except:
                            continue
                    elif isinstance(sample, dict):
                        processed_samples.append(sample)
                
                if processed_samples:
                    df = pd.DataFrame(processed_samples)
        
        # Case 2: If we have a data array with tables
        elif isinstance(json_data, dict) and 'data' in json_data:
            all_rows = []
            for table in json_data['data']:
                table_name = table.get('table', '')
                rows = table.get('rows', [])
                
                # Add table name to each row
                for row in rows:
                    row['Table'] = table_name
                    all_rows.append(row)
            
            if all_rows:
                df = pd.DataFrame(all_rows)
                # Reorder columns to put Table first
                if 'Table' in df.columns:
                    cols = ['Table'] + [col for col in df.columns if col != 'Table']
                    df = df[cols]
        
        # Case 3: If we have a direct list of samples
        elif isinstance(json_data, list):
            processed_items = []
            for item in json_data:
                if isinstance(item, str):
                    try:
                        # Try to safely evaluate the string to a dictionary
                        item = item.replace("'", '"')
                        item_dict = json.loads(item)
                        processed_items.append(item_dict)
                    except:
                        continue
                elif isinstance(item, dict):
                    processed_items.append(item)
            
            if processed_items:
                df = pd.DataFrame(processed_items)
        
        if df is not None:
            return df
        else:
            st.error("Could not convert the JSON data to a proper tabular format")
            return None
            
    except Exception as e:
        st.error(f"Error converting to CSV: {str(e)}")
        return None

def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for path in file_paths:
        try:
            os.unlink(path)
        except Exception:
            pass
    try:
        os.rmdir(os.path.dirname(file_paths[0]))
    except Exception:
        pass

# Modify the process_page function to include custom prompt
async def process_pages_with_custom_prompt(page_paths, model, progress_bar, status_text):
    """Process multiple pages in parallel with custom prompt"""
    tasks = []
    for i, page_path in enumerate(page_paths):
        try:
            gemini_file = genai.upload_file(page_path, mime_type="application/pdf")
            
            # Create base prompt
            base_prompt = "Parse this PDF page into structured JSON format. Include all relevant information."
            
            # Add custom instructions if provided
            if st.session_state.custom_prompt:
                prompt = f"{base_prompt}\n\nAdditional Instructions:\n{st.session_state.custom_prompt}"
            else:
                prompt = base_prompt
            
            # Start chat session
            chat = model.start_chat()
            response = chat.send_message([gemini_file, prompt])
            
            # Update progress
            if progress_bar and status_text:
                progress_bar.progress((i + 1) / progress_bar.total_pages)
                status_text.text(f"Processed page {i + 1}/{progress_bar.total_pages}")
            
            result = json.loads(response.text)
            tasks.append(result)
        except Exception as e:
            st.error(f"Error processing page {i + 1}: {str(e)}")
            tasks.append(None)
    
    return tasks

def convert_to_excel(json_data):
    """Convert JSON data to Excel format with multiple sheets"""
    try:
        # Create a new workbook
        workbook = Workbook()
        
        if isinstance(json_data, dict) and 'data' in json_data:
            # Remove default sheet
            default_sheet = workbook.active
            workbook.remove(default_sheet)
            
            # Process each table into its own sheet
            for table in json_data['data']:
                table_name = table.get('table', 'Sheet')
                # Create safe sheet name (Excel has 31 char limit and some invalid chars)
                safe_name = str(table_name)[:31].replace('/', '_').replace('\\', '_')
                
                # Create new sheet
                sheet = workbook.create_sheet(title=safe_name)
                
                # Get rows for this table
                rows = table.get('rows', [])
                if rows:
                    # Write headers
                    headers = list(rows[0].keys())
                    headers.remove('Table') if 'Table' in headers else None
                    for col, header in enumerate(headers, 1):
                        sheet.cell(row=1, column=col, value=header)
                    
                    # Write data
                    for row_idx, row in enumerate(rows, 2):
                        for col_idx, header in enumerate(headers, 1):
                            sheet.cell(row=row_idx, column=col_idx, value=row.get(header))
        
        # Save to bytes buffer
        excel_buffer = io.BytesIO()
        workbook.save(excel_buffer)
        excel_buffer.seek(0)
        
        return excel_buffer
    except Exception as e:
        st.error(f"Error converting to Excel: {str(e)}")
        return None

def load_saved_settings():
    """Load saved settings from local file"""
    settings_file = Path("settings.json")
    if settings_file.exists():
        try:
            with open(settings_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"api_key": "", "custom_prompt": ""}
    return {"api_key": "", "custom_prompt": ""}

def save_settings(api_key="", custom_prompt=""):
    """Save settings to local file"""
    settings_file = Path("settings.json")
    try:
        settings = {
            "api_key": api_key,
            "custom_prompt": custom_prompt
        }
        with open(settings_file, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        st.error(f"Error saving settings: {str(e)}")

def clear_settings(clear_api=False, clear_prompt=False):
    """Clear specified settings and session state"""
    try:
        settings_file = Path("settings.json")
        current_settings = {}
        
        # Load current settings if file exists
        if settings_file.exists():
            try:
                with open(settings_file, "r") as f:
                    current_settings = json.load(f)
            except:
                pass
        
        # Update settings based on what needs to be cleared
        if clear_api:
            current_settings["api_key"] = ""
            st.session_state.api_key = ""
        if clear_prompt:
            current_settings["custom_prompt"] = ""
            st.session_state.custom_prompt = ""
        
        # Save updated settings
        with open(settings_file, "w") as f:
            json.dump(current_settings, f)
        
        return True
    except Exception as e:
        st.error(f"Error clearing settings: {str(e)}")
        return False

def main():
    st.set_page_config(page_title="AgNext X Gemini Flash 2.0", layout="wide")
    
    # Load saved settings
    saved_settings = load_saved_settings()
    
    # Initialize session state for storing results, API key, and custom prompt
    if 'merged_json' not in st.session_state:
        st.session_state.merged_json = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = saved_settings.get("api_key", "") or os.getenv("GEMINI_API_KEY", "")
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = saved_settings.get("custom_prompt", "")
    if 'show_tabs' not in st.session_state:
        st.session_state.show_tabs = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "JSON"
    
    # Custom CSS for title styling
    st.markdown("""
        <style>
        .title {
            text-align: center;
            color: #2E4053;
            padding: 2rem;
            border-radius: 5px;
            margin-bottom: 3.5rem;
            background: linear-gradient(to right, #f8f9fa, #ffffff, #f8f9fa);
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .stButton > button {
            margin: 0 !important;
            padding: 0 1rem !important;
            border: none !important;
            background: transparent !important;
            color: #0066cc !important;
        }
        .stButton > button:hover {
            color: #003366 !important;
            background: #f0f0f0 !important;
        }
        .input-container {
            display: flex;
            align-items: center;
            width: 100%;
            position: relative;
        }
        .input-container .stButton {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 1;
        }
        .input-container input, .input-container textarea {
            width: 100% !important;
            padding-right: 40px !important;
        }
        .upload-text {
            margin-bottom: 2rem;
            padding: 1rem;
            text-align: center;
            color: #5D6D7E;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        .data-container {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 5px;
        }
        .api-key-container, .prompt-container {
            margin: 2rem 0;
            padding: 1rem;
            border-radius: 5px;
            background: #f8f9fa;
        }
        .hint-text {
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }
        /* Hide "press enter to apply" messages */
        small {
            display: none !important;
        }
        /* Custom input container styles */
        .custom-input-container {
            position: relative;
            width: 100%;
        }
        .custom-input-container input,
        .custom-input-container textarea {
            width: 100% !important;
            padding-right: 40px !important;
            box-sizing: border-box !important;
        }
        .custom-input-container .check-button {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            background: none;
            border: none;
            color: #0066cc;
            cursor: pointer;
            padding: 5px;
            font-size: 16px;
            z-index: 100;
        }
        .custom-input-container .check-button:hover {
            color: #003366;
        }
        /* Input field styling */
        .stTextInput > div > div > input {
            padding-right: 45px !important;
        }
        .stTextArea > div > div > textarea {
            padding-right: 45px !important;
        }
        /* Button positioning */
        .stButton > button {
            position: absolute !important;
            right: 5px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            z-index: 1 !important;
            background: transparent !important;
            border: none !important;
            color: #0066cc !important;
            min-height: 0 !important;
            padding: 0 10px !important;
            margin: 0 !important;
            line-height: normal !important;
        }
        .stButton > button:hover {
            color: #003366 !important;
            background: transparent !important;
        }
        /* Container for input + button */
        div[data-testid="column"] {
            position: relative !important;
        }
        /* Process PDF button styling */
        div[data-testid="stButton"] > button {
            background: linear-gradient(to right, #0066cc, #0099ff) !important;
            color: white !important;
            padding: 0.75rem 2rem !important;
            border-radius: 10px !important;
            border: none !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
            margin: 1.5rem 0 !important;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(to right, #005bb7, #0088e8) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            color: white !important;
        }
        div[data-testid="stButton"] > button:active {
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
        }
        .file-info {
            background: #f8f9fa !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            border: 1px solid #e9ecef !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        }
        /* Results container spacing */
        .results-spacing {
            margin-bottom: 2rem;
            padding: 1rem;
        }
        
        /* Tab container spacing */
        .stTabs {
            margin-top: 0rem;
        }
        
        /* Process button bottom margin */
        div[data-testid="stButton"] button[kind="primary"] {
            margin-bottom: 3rem !important;
        }
        </style>
        <h1 class="title">AgNext PDF Parser</h1>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="upload-text">Powered by Gemini 2.0 Flash</p>', unsafe_allow_html=True)
    
    # API Key input section with expander
    with st.expander("🔑 Add Gemini API Key"):
        container = st.container()
        with container:
            col1, col2, col3 = st.columns([18, 1, 1])
            with col1:
                api_key_input = st.text_input(
                    "Enter API Key",
                    value=st.session_state.api_key if st.session_state.api_key else "",
                    type="password",
                    placeholder="Enter your Gemini API key (optional)",
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("✓", key="api_key_check", help="Apply API Key"):
                    st.session_state.api_key = api_key_input
                    save_settings(
                        api_key=api_key_input,
                        custom_prompt=st.session_state.custom_prompt
                    )
            with col3:
                if st.button("🗑️", key="clear_api_key", help="Clear API Key"):
                    if clear_settings(clear_api=True):
                        st.success("API Key cleared!")
                        st.rerun()

    # Custom prompt input section without expander
    container = st.container()
    with container:
        col1, col2, col3 = st.columns([18, 1, 1])
        with col1:
            custom_prompt = st.text_area(
                "Additional Instructions",
                value=st.session_state.custom_prompt,
                placeholder="Enter any specific instructions for parsing (optional)",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("✓", key="apply_instructions", help="Apply Instructions"):
                st.session_state.custom_prompt = custom_prompt
                save_settings(
                    api_key=st.session_state.api_key,
                    custom_prompt=custom_prompt
                )
        with col3:
            if st.button("🗑️", key="clear_instructions", help="Clear Instructions"):
                if clear_settings(clear_prompt=True):
                    st.success("Instructions cleared!")
                    st.rerun()

    # Add checkbox for multi-sheet option before file upload
    use_multi_sheet = st.checkbox("Split different tables into separate sheets", 
                                help="When checked, each different table type will be exported to a separate sheet in the Excel file")

    # Initialize model with appropriate API key
    if st.session_state.api_key:
        genai.configure(api_key=st.session_state.api_key)
        model = initialize_gemini_model()
    else:
        env_api_key = os.getenv("GEMINI_API_KEY")
        if not env_api_key:
            st.error("⚠️ No API key found. Please either add it in the configuration above or set it in the environment variables.")
            st.stop()
        genai.configure(api_key=env_api_key)
        model = initialize_gemini_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Store file info in session state
        if 'uploaded_file_info' not in st.session_state or st.session_state.uploaded_file_info['name'] != uploaded_file.name:
            st.session_state.uploaded_file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type
            }
            # Clear results only when new file is uploaded
            st.session_state.merged_json = None
            st.session_state.df = None
        
        # Read PDF info
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        
        # Display file info in a nice format
        st.markdown("### 📁 File Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="file-info">Filename: ' + uploaded_file.name + '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="file-info">File size: {uploaded_file.size / 1024:.2f} KB</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="file-info">Total pages: {total_pages}</div>', unsafe_allow_html=True)
        
        # Process button
        if st.button("Process PDF", use_container_width=True, type="primary") or st.session_state.merged_json is not None:
            if st.session_state.merged_json is None:
                progress_bar = st.progress(0)
                progress_bar.total_pages = total_pages
                status_text = st.empty()
                
                page_paths, total_pages = split_pdf_to_pages(uploaded_file)
                
                try:
                    # Process pages with custom prompt
                    with st.spinner("⚡ Processing pages in parallel..."):
                        results = asyncio.run(process_pages_with_custom_prompt(page_paths, model, progress_bar, status_text))
                    
                    # Merge results using Gemini
                    with st.spinner("🔄 Merging results with AI..."):
                        st.session_state.merged_json = asyncio.run(merge_with_gemini(results, model))
                        if st.session_state.merged_json:
                            st.session_state.df = convert_to_csv(st.session_state.merged_json)
                
                finally:
                    # Clean up temporary files
                    cleanup_temp_files(page_paths)
                    status_text.text("✅ Processing complete!")
                    progress_bar.progress(1.0)
            
            # Display results if available
            if st.session_state.merged_json:
                # Add vertical spacing
                st.markdown("<div class='results-spacing'></div>", unsafe_allow_html=True)
                
                # Create tabs for JSON and Spreadsheet views
                results_container = st.container()
                with results_container:
                    # Add section header with spacing
                    st.markdown("### 📊 Results")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Create tabs
                    tab1, tab2 = st.tabs(["📊 JSON View", "📈 Spreadsheet View"])
                    
                    # JSON Tab
                    with tab1:
                        st.json(st.session_state.merged_json)
                        col1, col2 = st.columns([6, 4])
                        with col1:
                            st.download_button(
                                label="⬇️ Download JSON",
                                data=json.dumps(st.session_state.merged_json, indent=2),
                                file_name="parsed_data.json",
                                mime="application/json",
                                use_container_width=True,
                                key="json_download"
                            )
                    
                    # Spreadsheet Tab
                    with tab2:
                        if use_multi_sheet:
                            # Convert to Excel with multiple sheets
                            excel_buffer = convert_to_excel(st.session_state.merged_json)
                            if excel_buffer:
                                # Create separate dataframes for each table
                                table_dfs = {}
                                if isinstance(st.session_state.merged_json, dict) and 'data' in st.session_state.merged_json:
                                    for table in st.session_state.merged_json['data']:
                                        table_name = table.get('table', 'Sheet')
                                        rows = table.get('rows', [])
                                        if rows:
                                            # Create DataFrame for this table
                                            df = pd.DataFrame(rows)
                                            # Remove 'Table' column if it exists
                                            if 'Table' in df.columns:
                                                df = df.drop('Table', axis=1)
                                            table_dfs[table_name] = df
                                
                                if table_dfs:
                                    # Create a selector for different tables
                                    st.markdown("### 📑 Select Table to View")
                                    selected_table = st.selectbox(
                                        "Choose a table to view",
                                        options=list(table_dfs.keys()),
                                        label_visibility="collapsed"
                                    )
                                    
                                    # Show the selected table
                                    st.markdown(f"**{selected_table}**")
                                    st.dataframe(
                                        table_dfs[selected_table],
                                        use_container_width=True,
                                        height=400
                                    )
                                
                                col1, col2 = st.columns([6, 4])
                                with col1:
                                    st.download_button(
                                        label="⬇️ Download Excel",
                                        data=excel_buffer,
                                        file_name="parsed_data.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True,
                                        key="excel_download"
                                    )
                        else:
                            # Use existing CSV conversion
                            if st.session_state.df is not None:
                                st.dataframe(
                                    st.session_state.df,
                                    use_container_width=True,
                                    height=400
                                )
                                col1, col2 = st.columns([6, 4])
                                with col1:
                                    st.download_button(
                                        label="⬇️ Download CSV",
                                        data=st.session_state.df.to_csv(index=False),
                                        file_name="parsed_data.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        key="csv_download"
                                    )
    else:
        # Clear session state when no file is uploaded
        if 'uploaded_file_info' in st.session_state:
            del st.session_state.uploaded_file_info
        st.session_state.merged_json = None
        st.session_state.df = None
        st.session_state.show_tabs = False

if __name__ == "__main__":
    main() 