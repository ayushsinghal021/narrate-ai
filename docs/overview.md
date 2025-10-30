Of course. Here is a high-level overview document for the NarratorAI project based on the provided repository information.

***

## NarratorAI: Project Overview

This document provides a high-level overview of the NarratorAI project. It is intended for new contributors, stakeholders, and anyone interested in understanding the project's purpose, architecture, and technical foundations without delving into implementation details.

### 1. Project Goals

**NarratorAI** is an automated data storytelling application designed to transform raw datasets into clear, compelling, and human-readable narratives. The primary goal is to bridge the gap between complex data and actionable insights, enabling users to understand the story hidden within their data without requiring deep statistical or machine learning expertise.

The project aims to achieve this by:

*   **Automating the Data Science Lifecycle:** It ingests raw data and automatically performs cleaning, preprocessing, statistical analysis, and predictive modeling.
*   **Identifying Key Insights:** It uses statistical methods and machine learning models to uncover significant patterns, correlations, and key drivers within the data.
*   **Generating Narratives and Visualizations:** It translates these statistical insights into plain-language text and creates relevant, dynamic visualizations to support the story.

Ultimately, NarratorAI empowers users to make data-driven decisions by presenting them with a finished "data story" rather than just raw numbers or charts.

### 2. Architecture

The system is designed with a modular, multi-layered architecture to separate concerns, promote reusability, and support multiple modes of interaction.

The architecture can be broken down into three main layers:

*   **Presentation Layer (Entry Points):** This layer provides multiple interfaces for users and systems to interact with NarratorAI.
    *   **Web Application (`app.py`):** A user-friendly interface built with **Streamlit**, allowing users to upload datasets interactively and view the generated story. This is ideal for business analysts and non-technical users.
    *   **REST API (`api.py`):** A programmatic interface built with **FastAPI**, enabling developers to integrate NarratorAI's capabilities into other applications or automated workflows.
    *   **Command-Line Interface (CLI) (`main.py`):** A scriptable interface for running the full analysis pipeline from the command line, suited for data scientists and for use in automated scripting.

*   **Orchestration Layer (`pipeline.py`):** This layer acts as the central controller. The `run_full_pipeline` function orchestrates the entire workflow by calling the various processing tasks in the correct sequence. This decouples the presentation layer from the core logic, meaning any interface can trigger the same robust process.

*   **Core Processing Layer (`core/` directory):** This layer contains the specialized modules that perform the actual data science work. Each module is responsible for a distinct stage of the analysis:
    *   **Data Cleaning (`core/cleaning.py`):** Ingests and standardizes data from various formats (CSV, JSON) and performs essential cleaning and preprocessing.
    *   **Statistical Analysis (`core/analysis.py`):** Executes statistical tests to uncover initial patterns and correlations in the data.
    *   **Predictive Modeling (`core/modeling.py`):** Trains machine learning models (specifically XGBoost) to identify the most influential features related to a user-defined target variable.
    *   **Storytelling & Visualization (`core/storytelling.py`):** Synthesizes insights from the analysis and modeling stages to generate natural language narratives and create visualizations using Plotly.




### 3. Data Flow

The data flow through NarratorAI is a sequential pipeline, where the output of one stage becomes the input for the next.

1.  **Data Ingestion:** The process begins when a user uploads a dataset (e.g., a CSV file) and specifies a target column through one of the interfaces (Web App, API, or CLI).
2.  **Cleaning and Preprocessing:** The raw data is passed to the **Cleaning** module. It is loaded into a pandas DataFrame, cleaned of inconsistencies (e.g., missing values are imputed), and preprocessed for analysis.
3.  **Insight Generation (Analysis & Modeling):** The cleaned DataFrame is then processed in parallel or sequence by two modules:
    *   The **Statistical Analysis** module generates insights based on correlations and statistical tests.
    *   The **Predictive Modeling** module trains an XGBoost model to identify the key features that drive the target column, generating further insights.
4.  **Narrative Synthesis:** All generated insights (a collection of statistical findings and feature importances) are passed to the **Storytelling** module.
5.  **Output Generation:** The Storytelling module uses the insights to:
    *   Create relevant data visualizations (e.g., charts, plots).
    *   Generate a cohesive, human-readable text narrative using a Large Language Model (LLM).
6.  **Delivery:** The final story, comprising text and visualizations, is delivered back to the user through the initial interface—displayed on the Streamlit web page, returned as a JSON object from the API, or printed to the console.

### 4. Design Decisions

Several key design decisions were made to ensure the project is flexible, maintainable, and powerful.

*   **Modular Core Logic:** The separation of the core data science functions into distinct modules (`cleaning`, `analysis`, `modeling`, `storytelling`) makes the system highly maintainable and extensible. A new analysis technique or visualization type can be added by creating a new module without affecting the rest of the system.

*   **Multiple Entry Points:** Providing a Streamlit UI, a FastAPI, and a CLI was a deliberate choice to cater to a wide range of users. This maximizes the project's applicability, from interactive analysis by a business user to automated execution in a larger data pipeline.

*   **Orchestration via a Central Pipeline:** Using a single `run_full_pipeline` function to control the workflow simplifies the logic within the presentation layer. Each interface only needs to know about this one function, ensuring a consistent execution process regardless of how the job is initiated.

*   **Flexible LLM Integration:** The storytelling module is designed to work with both the **OpenAI API** and a local LLM via the **Hugging Face Transformers** library. This is a crucial decision that provides flexibility based on user needs for cost, data privacy, or offline capability. The system can function effectively even without an internet connection or an API key.

*   **Choice of High-Performance Libraries:**
    *   **XGBoost** was chosen for predictive modeling due to its high performance, accuracy, and built-in ability to provide feature importance scores, which is essential for identifying "key drivers" for the narrative.
    *   **FastAPI** was selected for the API because of its high speed, modern features (e.g., type hints), and automatic generation of interactive API documentation.
    *   **Streamlit** was chosen for the UI because it enables rapid development of interactive data applications directly in Python, greatly simplifying the creation of a user-friendly front end.