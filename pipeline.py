import os
import json
from core import tasks

def run_full_pipeline(fname, data, target_col):
    """
    Orchestrates the entire data storytelling pipeline using modular tasks.
    """
    output_dir = os.path.join('output')
    os.makedirs(output_dir, exist_ok=True)
    
    report = {"title": f"Data Story for {fname}", "insights": []}

    # Task 1: Ingest and Clean Data
    df_clean, metadata = tasks.run_cleaning_task(data)
    
    # Task 2: Statistical Analysis
    stat_insights = tasks.run_statistical_analysis_task(df_clean, metadata)
    
    # Task 3: Predictive Modeling
    ml_insights = tasks.run_modeling_task(df_clean, metadata, target_col)
    
    all_insights = stat_insights + ml_insights
    print(f"--> Total insights to process: {len(all_insights)}")

    if not all_insights:
        print("\n\n*** No significant insights found based on current criteria. Pipeline will stop here. ***\n\n")
        return None

    # Task 4: Narrative and Visualization Generation
    report["insights"] = tasks.run_storytelling_task(all_insights, df_clean, output_dir)
        
    # Task 5: Assemble Final Report
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Pipeline finished. Report saved to {report_path}")
    return report_path
