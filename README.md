# 📊 NarratorAI: Automated Data Storytelling Bot

[![Python Version][python-shield]][python-url]
[![Streamlit Version][streamlit-shield]][streamlit-url]
[![License: MIT][license-shield]][license-url]

**NarratorAI** is a sophisticated application that transforms raw data into compelling, human-readable narratives. Using advanced AI/ML techniques and statistical analysis, this bot can automatically analyze an uploaded dataset, identify key patterns, generate dynamic visualizations, and produce an insightful data story in plain language.

This project demonstrates expertise in the full data science lifecycle, from ETL and data cleaning to predictive modeling and AI-powered natural language generation (NLG).


*(This is a placeholder image. You can replace this link after you take a screenshot of your running app!)*

---

## ✨ Core Features

*   **Intelligent Data Ingestion:** Extracts data from multiple sources like CSV and JSON.
*   **Automated Data Cleaning:** Performs type conversion, handles missing values, and cleans column names.
*   **Statistical Insight Mining:** Automatically runs correlation analysis and hypothesis testing (ANOVA) to find significant patterns.
*   **Predictive Modeling Engine:**
    *   Trains an XGBoost model (Classifier or Regressor) to find the most important features driving a user-selected target variable.
    *   Dynamically adapts the model based on the nature of the target column.
*   **AI-Powered Narrative Generation:** Uses a Hugging Face Transformer model (`distilgpt2`) to translate statistical insights into fluid, easy-to-understand text.
*   **Dynamic Visualization:** Automatically generates interactive Plotly charts (scatter plots, box plots, bar charts) best suited for each discovered insight.
*   **Interactive Dashboard:** A user-friendly Streamlit interface allows users to upload data, select a target for analysis, and explore the generated story.

---

## 🛠️ Tech Stack

The project is built with a modern data science and Python stack.

- **Backend & Data Processing:** Python 3.9+, Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **AI & Natural Language Generation:** Hugging Face Transformers, PyTorch
- **Frontend & Visualization:** Streamlit, Plotly

---

## 🚀 Getting Started

Follow these instructions to set up and run the NarratorAI bot on your local machine.

### Prerequisites

*   Python 3.9+
*   `pip` and `venv` for package management
*   Git for cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/automated-data-storytelling-bot.git
cd automated-data-storytelling-bot
```
*(Replace `YourUsername` with your actual GitHub username)*

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to keep project dependencies isolated.

*   **On macOS or Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

*   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 3. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
**Note:** This step may take several minutes, as it needs to download large libraries like `torch` and `transformers`.

### 4. Run the Streamlit App

Once the installation is complete, you can run the application.

```bash
streamlit run app.py
```

Your web browser should automatically open to `http://localhost:8501`, where you can start using the application.

---

## usage How to Use NarratorAI

1.  **Launch the application** following the steps above.
2.  **Upload a Dataset:** Click "Browse files" and select a CSV file from your computer.
3.  **Select a Target Column:** A dropdown menu will appear. Choose the column from your dataset that you are most interested in understanding or predicting.
4.  **Analyze:** Click the "Analyze" button.
5.  **Explore the Story:** The bot will perform the full analysis and generate a report containing:
    *   **AI-Generated Narratives:** Plain-text explanations of what the data means.
    *   **Supporting Visualizations:** Interactive charts that provide evidence for each narrative point.
