import os
import pandas as pd
import plotly.express as px
from transformers.pipelines import pipeline

# --- Global Model Initialization ---
# We initialize the model only once when the module is loaded.
# This is much more efficient than loading it for every call.
# A comprehensive try-except block handles potential download/load issues.
try:
    print("Initializing the NLG model (distilgpt2)... This may take a moment.")
    # Using a specific, well-known model for consistency.
    narrator = pipeline("text-generation", model="distilgpt2")
    print("NLG model initialized successfully.")
except Exception as e:
    print(f"CRITICAL WARNING: Could not load the Hugging Face model. "
          f"Narratives will be generated from basic templates. Error: {e}")
    narrator = None


def create_visualization(insight: dict, df: pd.DataFrame, output_dir: str) -> str | None:
    """
    Generates and saves a Plotly chart based on the insight type.

    Args:
        insight: A dictionary containing the details of the insight.
        df: The cleaned pandas DataFrame for plotting.
        output_dir: The directory to save the plot HTML file.

    Returns:
        The file path to the generated plot, or None if no plot was created.
    """
    fig = None
    # Create a more robust and unique filename.
    feature_name = insight['details'].get('feature1', insight['details'].get('numeric_feature', 'feature_importance'))
    filename = f"{insight['type']}_{feature_name}.html"
    filepath = os.path.join(output_dir, filename)

    try:
        if insight['type'] == 'correlation':
            details = insight['details']
            fig = px.scatter(df, x=details['feature1'], y=details['feature2'],
                             trendline="ols", title=insight['title'],
                             labels={details['feature1']: details['feature1'].replace('_', ' ').title(),
                                     details['feature2']: details['feature2'].replace('_', ' ').title()})

        elif insight['type'] == 'significant_difference':
            details = insight['details']
            fig = px.box(df, x=details['categorical_feature'], y=details['numeric_feature'],
                         title=insight['title'], color=details['categorical_feature'])

        elif insight['type'] == 'feature_importance':
            details = insight['details']
            # Convert the list of dicts into a DataFrame for easy plotting
            features_df = pd.DataFrame(details['features'])
            fig = px.bar(features_df, x='importance', y='feature', orientation='h',
                         title=insight['title'])
            fig.update_yaxes(categoryorder="total ascending") # Show most important at the top

        if fig:
            fig.write_html(filepath, full_html=False, include_plotlyjs='cdn')
            return filepath
            
    except Exception as e:
        print(f"Warning: Could not generate visualization for insight '{insight['title']}'. Error: {e}")

    return None


def generate_narrative_from_insight(insight: dict) -> str:
    """
    Generates human-readable text for a given insight using a robust, defensive approach.

    Args:
        insight: A dictionary containing the details of the insight.

    Returns:
        A string containing the AI-generated or template-based narrative.
    """
    # Define a consistent fallback text to use in case of any failure.
    fallback_text = (f"A key insight of type '{insight['type']}' was found. This indicates a "
                     f"significant pattern in your data that warrants further investigation.")

    # If the model failed to load at startup, immediately return the fallback.
    if not narrator:
        return fallback_text

    details = insight['details']
    prompt = ""

    # Create a specific, clear prompt for the model.
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
        return fallback_text

    # --- Bulletproof NLG Generation Block ---
    try:
        # The pipeline's output can be complex. We handle it defensively.
        raw_output = narrator(prompt, max_length=100, num_return_sequences=1, pad_token_id=narrator.model.config.eos_token_id)
        
        # 1. Check if the output is a list and not empty
        if isinstance(raw_output, list) and raw_output:
            # 2. Check if the first element is a dictionary
            if isinstance(raw_output[0], dict):
                # 3. Safely get the text using .get() to avoid KeyErrors
                generated_text = raw_output[0].get('generated_text', '')
                
                # 4. Clean the text by removing the original prompt
                text = generated_text.replace(prompt, "").strip()
                
                # 5. Validate the final text is meaningful
                if text and len(text.split()) > 5:
                    return text # SUCCESS!
                else:
                    # The model returned an empty or too-short string.
                    print(f"Warning: NLG model returned insufficient text for prompt: '{prompt}'")
                    return fallback_text
            
        # If the checks above fail, the format is unexpected.
        print(f"Warning: NLG output was not in the expected format (list of dicts). Output: {raw_output}")
        return fallback_text

    except Exception as e:
        # Catch any other unexpected errors during the generation process.
        print(f"An unexpected error occurred during narrative generation: {e}")
        return fallback_text