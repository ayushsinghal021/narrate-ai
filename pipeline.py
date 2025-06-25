import os
import json
from jinja2 import Environment, FileSystemLoader
from core import cleaning, analysis, modeling, storytelling

def run_full_pipeline(filepath, target_col): # Added target_col argument
    """
    Orchestrates the entire data storytelling pipeline.
    This function simulates an Apache Airflow DAG.
    """
    session_id = os.path.basename(filepath).split('.')[0]
    output_dir = os.path.join('output', session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    report = {"title": f"Data Story for {session_id}", "insights": []}

    # Task 1: Ingest and Clean Data
    print("Pipeline Task 1: Loading and Cleaning Data...")
    df_raw = cleaning.load_data(filepath)
    df_clean, metadata = cleaning.clean_and_preprocess_data(df_raw.copy())
    df_clean.to_csv(os.path.join(output_dir, 'cleaned_data.csv'), index=False)
    
    # Task 2: Statistical Analysis
    print("Pipeline Task 2: Running Statistical Analysis...")
    stat_insights = analysis.get_statistical_insights(df_clean, metadata)
    
    # Task 3: Predictive Modeling (now requires target_col)
    print("Pipeline Task 3: Running Predictive Models...")
    ml_insights = modeling.run_predictive_model(df_clean, metadata, target_col)
    
    all_insights = stat_insights + ml_insights
    
    # Task 4: Narrative and Visualization Generation
    print("Pipeline Task 4: Generating Story and Visuals...")
    for insight in all_insights:
        narrative = storytelling.generate_narrative_from_insight(insight)
        plot_path = storytelling.create_visualization(insight, df_clean, output_dir)
        
        report["insights"].append({
            "title": insight['title'],
            "narrative": narrative,
            "plot_path": plot_path
        })
        
    # Task 5: Assemble Final Report
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Pipeline finished. Report saved to {report_path}")
    return report_path