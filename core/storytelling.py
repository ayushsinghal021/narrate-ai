# <<< --- STEP 1: THE "CANARY" --- >>>
print("\n\n--- LOADING LATEST storytelling.py with OpenAI v1.x client ---\n\n")

import os
import pandas as pd
import plotly.express as px
from transformers.pipelines import pipeline
import requests
from openai import OpenAI, APIConnectionError
import torch

# --- Configuration ---
LOCAL_LLM_URL = "http://127.0.0.1:1234/v1"

# --- Global Model Initialization ---
try:
    # --- NEW: Check for GPU and set the device ---
    if torch.cuda.is_available():
        device = 0  # 0 is the ID of the first GPU
        print("GPU found! Setting device to 'cuda:0'.")
    else:
        device = -1 # -1 tells the pipeline to use the CPU
        print("No GPU found. Setting device to 'cpu'.")

    print("Initializing the fallback NLG model (distilgpt2)...")
    # --- NEW: Pass the device to the pipeline ---
    narrator = pipeline("text-generation", model="distilgpt2", device=device)
    print("Fallback NLG model initialized successfully.")
except Exception as e:
    print(f"CRITICAL WARNING: Could not load the fallback Hugging Face model. Error: {e}")
    narrator = None


def is_local_llm_available():
    print(f"Checking for local LLM server at {LOCAL_LLM_URL}...")
    try:
        response = requests.get(f"{LOCAL_LLM_URL}/models", timeout=2)
        if response.status_code == 200:
            print("...Local LLM server found.")
            return True
        return False
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("...Local LLM server not found.")
        return False

def generate_narrative_local_llm(prompt: str) -> str | None:
    try:
        client = OpenAI(base_url=LOCAL_LLM_URL, api_key="not-needed")
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst who explains insights in a clear, concise, and easy-to-understand business context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except APIConnectionError as e:
        print(f"Local LLM connection error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with the local LLM: {e}")
        return None

def generate_narrative_from_insight(insight: dict) -> str:
    details = insight['details']
    prompt = ""
    # (The prompt creation logic is unchanged)
    if insight['type'] == 'correlation':
        corr_type = "positive" if details['correlation'] > 0 else "negative"
        prompt = (f"In a business report, briefly explain the implication of a strong {corr_type} "
                  f"correlation ({details['correlation']:.2f}) between '{details['feature1']}' "
                  f"and '{details['feature2']}'.")
    elif insight['type'] == 'significant_difference':
        prompt = (f"In simple terms, what does it mean for a business if the average "
                  f"'{details['numeric_feature']}' is significantly different across various "
                  f"'{details['categorical_feature']}' groups?")
    elif insight['type'] == 'feature_importance':
        top_feature = details['features'][0]['feature']
        prompt = (f"A predictive model shows that '{top_feature}' is the most important factor "
                  f"in predicting '{details['target']}'. Briefly describe a possible business "
                  f"reason for this.")
    if not prompt:
        return "Could not generate a valid prompt for this insight."

    if is_local_llm_available():
        local_response = generate_narrative_local_llm(prompt)
        if local_response:
            print("Success! Using response from local LLM.")
            return local_response
        else:
            print("Local LLM was found but failed to generate a response. Falling back...")

    if narrator:
        print("Using fallback Hugging Face model (distilgpt2).")
        try:
            raw_output = narrator(prompt, max_length=120, num_return_sequences=1, pad_token_id=narrator.model.config.eos_token_id)
            if isinstance(raw_output, list) and raw_output:
                if isinstance(raw_output[0], dict):
                    text = raw_output[0].get('generated_text', '').replace(prompt, "").strip()
                    if text and len(text.split()) > 5:
                        return text
        except Exception as e:
            print(f"Fallback model failed: {e}")
    
    return (f"A key insight of type '{insight['type']}' was found. This indicates a "
            f"significant pattern in your data that warrants further investigation.")

# Visualization function is unchanged
def create_visualization(insight: dict, df: pd.DataFrame, output_dir: str) -> str | None:
    fig = None
    details = insight.get('details', {})
    # Only proceed if details is a dict
    if not isinstance(details, dict):
        print(f"Skipping visualization for insight '{insight.get('title', '')}' because details is not a dict.")
        return None

    feature_name = details.get('feature1', details.get('numeric_feature', 'feature_importance'))
    filename = f"{insight['type']}_{feature_name}.html"
    filepath = os.path.join(output_dir, filename)
    try:
        if insight['type'] == 'correlation':
            if details['feature1'] in df.columns and details['feature2'] in df.columns:
                fig = px.scatter(df, x=details['feature1'], y=details['feature2'], trendline="ols", title=insight['title'])
        elif insight['type'] == 'significant_difference':
            if details['numeric_feature'] in df.columns and details['categorical_feature'] in df.columns:
                fig = px.box(df, x=details['categorical_feature'], y=details['numeric_feature'], title=insight['title'], color=details['categorical_feature'])
        elif insight['type'] == 'feature_importance':
            features_df = pd.DataFrame(details['features'])
            fig = px.bar(features_df, x='importance', y='feature', orientation='h', title=insight['title'])
            fig.update_yaxes(categoryorder="total ascending")
        if fig:
            fig.write_html(filepath, full_html=False, include_plotlyjs='cdn')
            return filepath
    except Exception as e:
        print(f"Warning: Could not generate visualization for insight '{insight['title']}'. Error: {e}")
    return None