import streamlit as st
import streamlit.components.v1 as components
import os
import json
import pandas as pd
import requests
import time
from core.cleaning import load_data

# --- Page Configuration ---
st.set_page_config(
    page_title="NarratorAI",
    page_icon="📊",
    layout="wide"
)

# --- API Configuration ---
API_URL = "http://127.0.0.1:8000"

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

# Initialize session state
if 'task_id' not in st.session_state:
    st.session_state.task_id = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report' not in st.session_state:
    st.session_state.report = None

if uploaded_file is not None:
    try:
        df_cols = load_data(uploaded_file, uploaded_file.name)
        column_options = df_cols.columns.tolist()
        uploaded_file.seek(0)

        st.info("Step 2: Select the column you want to analyze or predict.")
        target_col = st.selectbox(
            "Which column is your primary target?",
            options=column_options,
            index=len(column_options) - 1
        )
        
        if st.button(f"Analyze '{target_col}'", type="primary"):
            st.session_state.analysis_done = False
            st.session_state.report = None
            st.session_state.task_id = None

            with st.spinner("Sending your request to the analysis engine..."):
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                data = {'target_col': target_col}
                try:
                    response = requests.post(f"{API_URL}/analyze", files=files, data=data)
                    if response.status_code == 200:
                        st.session_state.task_id = response.json().get('task_id')
                        st.success("Analysis started! You can see the progress below.")
                    else:
                        st.error(f"Failed to start analysis: {response.text}")
                except requests.exceptions.ConnectionError as e:
                    st.error(f"Could not connect to the analysis engine. Please make sure the API is running. Error: {e}")

    except Exception as e:
        st.error(f"Could not read the uploaded file. Please check the format. Error: {e}")

if st.session_state.task_id and not st.session_state.analysis_done:
    with st.spinner("Polling for analysis status..."):
        while True:
            try:
                status_response = requests.get(f"{API_URL}/status/{st.session_state.task_id}")
                if status_response.status_code == 200:
                    status = status_response.json().get('status')
                    if status == 'completed':
                        results_response = requests.get(f"{API_URL}/results/{st.session_state.task_id}")
                        if results_response.status_code == 200:
                            report_path = results_response.json().get('result')
                            if report_path and os.path.exists(report_path):
                                with open(report_path, 'r') as f:
                                    st.session_state.report = json.load(f)
                                st.session_state.analysis_done = True
                                st.rerun()
                            else:
                                st.error("Analysis complete, but the report file was not found.")
                                break
                        else:
                            st.error(f"Failed to get results: {results_response.text}")
                            break
                    elif status == 'failed':
                        st.error("Analysis failed. Please check the logs.")
                        break
                    else:
                        time.sleep(5) # Poll every 5 seconds
                else:
                    st.error(f"Failed to get status: {status_response.text}")
                    break
            except requests.exceptions.ConnectionError as e:
                st.error(f"Could not connect to the analysis engine. Please make sure the API is running. Error: {e}")
                break

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
                components.html(plot_html, height=450, scrolling=True)
            else:
                st.warning("No visualization was generated for this insight.")
        st.markdown("---")
