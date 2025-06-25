import streamlit as st
import os
import json
import pandas as pd
from pipeline import run_full_pipeline
from core.cleaning import load_data # We need this to get column names

# --- Page Configuration ---
st.set_page_config(
    page_title="NarratorAI",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("NarratorAI 🤖")
st.sidebar.markdown("From Raw Data to Compelling Narrative. Automatically.")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1.  **Upload** your CSV data.
2.  **Select** the column you want to understand or predict (your "target").
3.  **Our AI pipeline** analyzes the data to find the key drivers for your target.
4.  **Receive** a full data story with text, charts, and key insights.
""")

# --- Main Application ---
st.title("📊 Automated Data Storytelling Bot")
st.markdown("Upload your dataset, select a target column, and let our AI do the rest.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Initialize session state to hold information
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report' not in st.session_state:
    st.session_state.report = None

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    filepath = os.path.join(temp_dir, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load data just to get column names for the selectbox
    try:
        df_cols = load_data(filepath)
        column_options = df_cols.columns.tolist()

        # --- NEW: Target Column Selection ---
        st.info("Step 2: Select the column you want to analyze or predict.")
        target_col = st.selectbox(
            "Which column is your primary target?",
            options=column_options,
            index=len(column_options) - 1 # Default to the last column
        )
        
        if st.button(f"Analyze '{target_col}'", type="primary"):
            # Run the pipeline
            with st.spinner("Our AI is analyzing your data... This may take a moment."):
                try:
                    report_path = run_full_pipeline(filepath, target_col)
                    with open(report_path, 'r') as f:
                        st.session_state.report = json.load(f)
                    st.session_state.analysis_done = True
                except Exception as e:
                    st.error(f"An error occurred during the analysis pipeline: {e}")
                    st.exception(e) # This will print the full traceback for debugging
                    st.session_state.analysis_done = False
    
    except Exception as e:
        st.error(f"Could not read the uploaded file. Please check the format. Error: {e}")

# Display the report if analysis is done
if st.session_state.analysis_done and st.session_state.report:
    report = st.session_state.report
    st.success("Analysis Complete! Here is your data story.")
    st.markdown("---")
    st.header(report['title'])

    for i, insight in enumerate(report['insights']):
        st.subheader(f"Insight {i+1}: {insight['title']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🤖 AI Narrative:**")
            st.info(insight['narrative'])
        
        with col2:
            st.markdown("**📊 Supporting Visualization:**")
            if insight['plot_path'] and os.path.exists(insight['plot_path']):
                with open(insight['plot_path'], 'r', encoding='utf-8') as plot_file:
                    plot_html = plot_file.read()
                st.markdown.v1.html(plot_html, height=450, scrolling=True)
            else:
                st.warning("No visualization was generated for this insight.")
        st.markdown("---")