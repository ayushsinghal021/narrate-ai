# 📊 NarratorAI: Automated Data Storytelling Bot

**NarratorAI** is a sophisticated application that transforms raw data into compelling, human-readable narratives. Using advanced AI/ML techniques and statistical analysis, this bot can automatically analyze an uploaded dataset, identify key patterns, generate dynamic visualizations, and produce an insightful data story in plain language.

This project demonstrates expertise in the full data science lifecycle, from ETL and data cleaning to predictive modeling and AI-powered natural language generation (NLG).

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

- **Backend & Data Processing:** Python 3.9+, Pandas, NumPy, FastAPI, Uvicorn
- **Machine Learning:** Scikit-learn, XGBoost
- **AI & Natural Language Generation:** Hugging Face Transformers, PyTorch
- **Frontend & Visualization:** Streamlit, Plotly

---

## 🏛️ System Architecture

NarratorAI is designed with a decoupled architecture, separating the frontend user interface from the backend analysis engine. This design enhances scalability, maintainability, and robustness.

```
+---------------------------------------+
|               User                    |
+---------------------------------------+
                 | (Interacts with)
                 v
+---------------------------------------+
|      Frontend (Streamlit)             |
| - File Upload                         |
| - Target Selection                    |
| - Display Results                     |
+---------------------------------------+
                 | (HTTP Requests)
                 v
+---------------------------------------+
|        Backend (FastAPI)              |
| - /analyze (Start Analysis)           |
| - /status (Check Status)              |
| - /results (Get Results)              |
+---------------------------------------+
                 | (Executes)
                 v
+---------------------------------------+
|      Analysis Pipeline (Python)       |
| - Data Cleaning                       |
| - Statistical Analysis                |
| - Predictive Modeling                 |
| - Narrative Generation                |
| - Visualization                       |
+---------------------------------------+
```

*   **Frontend (Streamlit):** The frontend is a web application built with Streamlit. It provides a user-friendly interface for uploading data, selecting analysis parameters, and viewing the final data story. When a user initiates an analysis, the frontend sends a request to the backend API and then polls for the results.

*   **Backend (FastAPI):** The backend is a high-performance API built with FastAPI. It exposes endpoints for running the analysis pipeline, checking the status of analysis tasks, and retrieving the results. This allows the heavy data processing to be handled independently of the user interface. The backend is responsible for orchestrating the various stages of the analysis pipeline.

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

### 4. Run the Application

NarratorAI now runs with a separate backend and frontend. You will need to open two terminal windows to run the application.

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn api:app --reload
```

**Terminal 2: Run the Streamlit Frontend**

```bash
streamlit run app.py
```

Your web browser should automatically open to `http://localhost:8501`, where you can start using the application.

---

## 📖 How to Use NarratorAI

1.  **Launch the application** following the steps above.
2.  **Upload a Dataset:** Click "Browse files" and select a CSV file from your computer.
3.  **Select a Target Column:** A dropdown menu will appear. Choose the column from your dataset that you are most interested in understanding or predicting.
4.  **Analyze:** Click the "Analyze" button to start the analysis.
5.  **View the Story:** Once the analysis is complete, the application will display a full data story, including:
    *   **AI-Generated Narratives:** Plain-text explanations of the key insights discovered in your data.
    *   **Supporting Visualizations:** Interactive Plotly charts that provide visual evidence for each narrative point.

---

## 🔮 Future Improvements

This project provides a solid foundation for a powerful data storytelling application. The following are some potential areas for future improvement:

*   **Asynchronous Task Queue:** Replace the current `asyncio.create_task` with a more robust task queue like Celery with Redis or RabbitMQ to handle long-running analysis tasks more effectively.
*   **Scalable Data Processing:** For handling larger-than-memory datasets, replace the Pandas-based data processing with a more scalable solution like Dask or Spark.
*   **Caching:** Implement a caching layer (e.g., using Redis) to store the results of previous analyses, which would significantly improve performance for repeated requests.
*   **Plugin Architecture:** Develop a plugin system to allow for the easy addition of new analysis techniques, machine learning models, and visualization types.
*   **Enhanced User Control:** Provide more options in the user interface to allow users to customize the analysis process, such as selecting which analyses to run or tuning model parameters.
