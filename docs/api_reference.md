Of course. Here is a detailed API reference for the NarratorAI project, generated from the provided repository information.

---

# NarratorAI API Reference

This document provides a detailed reference for the NarratorAI API, core library functions, and command-line interface. It is intended for developers who need to interact with the service programmatically or extend its functionality.

## Table of Contents

1.  [REST API](#rest-api)
    *   [POST /analyze](#post-analyze)
2.  [Core Pipeline](#core-pipeline)
    *   [`run_full_pipeline`](#function-run_full_pipeline)
3.  [Core Modules](#core-modules)
    *   [Data Cleaning (`core.cleaning`)](#module-corecleaning)
    *   [Statistical Analysis (`core.analysis`)](#module-coreanalysis)
    *   [Predictive Modeling (`core.modeling`)](#module-coremodeling)
    *   [Storytelling & Visualization (`core.storytelling`)](#module-corestorytelling)
    *   [Pipeline Tasks (`core.tasks`)](#module-coretasks)
4.  [Command-Line Interface (CLI)](#command-line-interface-cli)

---

## REST API

The NarratorAI service is exposed via a RESTful API built with FastAPI. This is the primary method for external applications to submit data and receive a data story.

### **`POST /analyze`**

Starts a new analysis job by uploading a dataset and specifying a target column. The process is asynchronous. The API will return a job ID and the final results once the pipeline is complete.

*   **Description:** This endpoint accepts a dataset file (CSV, JSON, etc.) and a target column name. It then executes the entire data storytelling pipeline, which includes data cleaning, statistical analysis, predictive modeling, and narrative generation.

*   **Request:** `multipart/form-data`
    *   **`file`** (`UploadFile`, **required**): The dataset file to be analyzed.
    *   **`target_col`** (`str`, **required**): The name of the column in the dataset that you want to analyze or predict.

*   **Response:** `200 OK`
    *   **Content-Type:** `application/json`
    *   **Body:** A JSON object containing the complete analysis, including generated narratives, statistical insights, model results, and paths to visualizations.

*   **Example: Using `requests` in Python**

    ```python
    import requests
    import json

    API_URL = "http://127.0.0.1:8000/analyze"
    FILE_PATH = "path/to/your/dataset.csv"
    TARGET_COLUMN = "customer_churn"

    with open(FILE_PATH, "rb") as f:
        files = {"file": (FILE_PATH, f, "text/csv")}
        data = {"target_col": TARGET_COLUMN}

        try:
            response = requests.post(API_URL, files=files, data=data)
            response.raise_for_status()  # Raises an exception for 4XX/5XX errors
            
            # The full result from the pipeline
            analysis_results = response.json()
            print(json.dumps(analysis_results, indent=2))

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    ```

*   **Example: Response Body Structure**

    ```json
    {
      "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
      "summary": {
        "file_name": "dataset.csv",
        "rows": 5000,
        "columns": 15,
        "target_column": "customer_churn"
      },
      "statistical_insights": [
        {
          "type": "correlation",
          "column_1": "monthly_charges",
          "column_2": "customer_churn",
          "value": 0.85,
          "narrative": "There is a strong positive correlation between monthly charges and customer churn."
        }
      ],
      "model_results": {
        "model_type": "XGBoost Classifier",
        "accuracy": 0.92,
        "key_drivers": ["contract_type", "monthly_charges", "tenure"],
        "narrative": "The primary drivers of customer churn are the contract type, monthly charges, and customer tenure."
      },
      "visualizations": [
        {
          "title": "Feature Importance for Customer Churn",
          "path": "/results/a1b2c3d4/feature_importance.png"
        },
        {
          "title": "Monthly Charges vs. Churn",
          "path": "/results/a1b2c3d4/charges_vs_churn.png"
        }
      ]
    }
    ```

*   **Error Handling:**
    *   `422 Unprocessable Entity`: If `file` or `target_col` are not provided in the request.
    *   `500 Internal Server Error`: If an unexpected error occurs during the pipeline execution. The response body may contain details about the error.

---

## Core Pipeline

This section describes the main orchestrator function that drives the entire analysis.

### **Function: `run_full_pipeline`**

Orchestrates the entire data storytelling pipeline from data loading to final narrative generation.

*   **Module:** `pipeline.py`
*   **Description:** This function serves as the central entry point for the analysis process. It calls a sequence of tasks from `core.tasks` to clean the data, perform statistical analysis, build a predictive model, and generate narratives and visualizations.
*   **Parameters:**
    *   **`fname`** (`str`, **required**): The original filename of the data (e.g., `sales_data.csv`).
    *   **`data`** (file-like object, **required**): An open file object or bytes buffer containing the raw dataset.
    *   **`target_col`** (`str`, **required**): The name of the target column for analysis.
*   **Return Value:**
    *   **`dict`**: A dictionary containing all generated insights, narratives, and metadata from the pipeline. The structure is similar to the JSON response body of the `/analyze` endpoint.
*   **Example:**

    ```python
    from pipeline import run_full_pipeline

    file_name = 'marketing_campaign.csv'
    target = 'conversion_rate'

    with open(file_name, 'rb') as f:
        # 'data' is a file-like object
        results = run_full_pipeline(fname=file_name, data=f, target_col=target)
        print(results)
    ```
*   **Error Handling:**
    *   Can raise various exceptions from underlying modules, such as `ValueError` if the `target_col` is not found or `pd.errors.ParserError` for malformed data files.

---

## Core Modules

These modules contain the building blocks of the analysis pipeline.

### Module: `core.cleaning`

Functions for loading and cleaning data.

*   **Function: `load_data(file_obj, file_name)`**
    *   **Description:** Loads data from a file object into a pandas DataFrame, automatically detecting the file type (CSV, JSON, SAS) from the filename extension.
    *   **Parameters:**
        *   `file_obj` (file-like object, **required**): The data to load.
        *   `file_name` (`str`, **required**): The original filename, used to infer the file type.
    *   **Return Value:** `pandas.DataFrame`: The loaded data.
    *   **Error Handling:** Raises `ValueError` for unsupported file formats.

*   **Function: `clean_and_preprocess_data(df)`**
    *   **Description:** Performs automated data cleaning and preprocessing steps, such as handling missing values (imputation), correcting data types, and removing irrelevant columns.
    *   **Parameters:**
        *   `df` (`pandas.DataFrame`, **required**): The raw DataFrame to be cleaned.
    *   **Return Value:** `(pandas.DataFrame, dict)`: A tuple containing the cleaned DataFrame and a metadata dictionary describing the applied transformations.

### Module: `core.analysis`

Functions for performing statistical analysis.

*   **Function: `get_statistical_insights(df, metadata)`**
    *   **Description:** Runs a series of statistical tests (e.g., correlation analysis, ANOVA) on the cleaned data to identify significant patterns and relationships.
    *   **Parameters:**
        *   `df` (`pandas.DataFrame`, **required**): The cleaned DataFrame.
        *   `metadata` (`dict`, **required**): Metadata from the cleaning process.
    *   **Return Value:** `dict`: A dictionary where keys are test names and values are the resulting insights.

### Module: `core.modeling`

Functions for training and evaluating predictive models.

*   **Function: `run_predictive_model(df, metadata, target_col)`**
    *   **Description:** Trains an XGBoost model (Classifier or Regressor, chosen automatically based on the target column type) to identify the key drivers for the specified target column.
    *   **Parameters:**
        *   `df` (`pandas.DataFrame`, **required**): The cleaned DataFrame.
        *   `metadata` (`dict`, **required**): Metadata from the cleaning process.
        *   `target_col` (`str`, **required**): The name of the target variable to predict.
    *   **Return Value:** `dict`: A dictionary containing model results, including feature importances, performance metrics (e.g., accuracy or R-squared), and a summary narrative.
    *   **Error Handling:** Raises `ValueError` if `target_col` is not present in the DataFrame.

### Module: `core.storytelling`

Functions for generating natural language narratives and visualizations.

*   **Function: `create_visualization(insight, df, output_dir)`**
    *   **Description:** Creates a visualization (e.g., a bar chart, scatter plot) using Plotly based on a specific insight.
    *   **Parameters:**
        *   `insight` (`dict`, **required**): A dictionary describing the insight to visualize.
        *   `df` (`pandas.DataFrame`, **required**): The DataFrame containing the data.
        *   `output_dir` (`str`, **required**): The directory where the chart image will be saved.
    *   **Return Value:** `str`: The file path to the generated visualization.

*   **Function: `generate_narrative_from_insight(insight)`**
    *   **Description:** Uses a Large Language Model (local or OpenAI) to convert a structured insight (e.g., a correlation value) into a human-readable sentence or paragraph.
    *   **Parameters:**
        *   `insight` (`dict`, **required**): A dictionary containing a statistical or model-based finding.
    *   **Return Value:** `str`: The generated text narrative.
    *   **Error Handling:** May raise `openai.APIConnectionError` if it fails to connect to the OpenAI API.

### Module: `core.tasks`

High-level functions that wrap and execute each major stage of the pipeline.

*   **Function: `run_cleaning_task(data)`**
    *   **Description:** Executes the data loading and cleaning steps.
    *   **Returns:** `(pandas.DataFrame, dict)`: The cleaned DataFrame and its metadata.

*   **Function: `run_statistical_analysis_task(df_clean, metadata)`**
    *   **Description:** Executes the statistical analysis step.
    *   **Returns:** `dict`: A dictionary of statistical insights.

*   **Function: `run_modeling_task(df_clean, metadata, target_col)`**
    *   **Description:** Executes the predictive modeling step.
    *   **Returns:** `dict`: A dictionary of model-based insights.

*   **Function: `run_storytelling_task(all_insights, df_clean, output_dir)`**
    *   **Description:** Generates all narratives and visualizations for the collected insights.
    *   **Returns:** `dict`: A dictionary containing lists of narratives and visualization file paths.

---

## Command-Line Interface (CLI)

The project includes a CLI for running the full analysis pipeline on a local file.

*   **File:** `main.py`
*   **Description:** The `main` function serves as the entry point for command-line execution. It parses command-line arguments to get the input file and target column, then runs the `run_full_pipeline` function.
*   **Usage:**
    ```bash
    python main.py --file <path_to_file> --target-column <column_name>
    ```
*   **Arguments:**
    *   `--file` (`str`, **required**): The path to the input data file (e.g., `data/sales.csv`).
    *   `--target-column` (`str`, **required**): The name of the target column for analysis, enclosed in quotes if it contains spaces.
*   **Example:**

    ```bash
    # Analyze the 'satisfaction_level' in an HR dataset
    python main.py --file "data/hr_data.csv" --target-column "satisfaction_level"
    ```
*   **Output:**
    The results of the analysis, including insights and narratives, will be printed to the standard output as a formatted JSON object. Visualizations will be saved to a local results directory.