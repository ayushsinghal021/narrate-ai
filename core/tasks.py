
import os
import json
from core import cleaning, analysis, modeling, storytelling

def run_cleaning_task(data):
    """
    Runs the data cleaning and preprocessing task.
    """
    print("Pipeline Task 1: Loading and Cleaning Data...")
    df_clean, metadata = cleaning.clean_and_preprocess_data(data)
    return df_clean, metadata

def run_statistical_analysis_task(df_clean, metadata):
    """
    Runs the statistical analysis task.
    """
    print("Pipeline Task 2: Running Statistical Analysis...")
    stat_insights = analysis.get_statistical_insights(df_clean, metadata)
    print(f"--> Found {len(stat_insights)} statistical insights.")
    return stat_insights

def run_modeling_task(df_clean, metadata, target_col):
    """
    Runs the predictive modeling task.
    """
    print("Pipeline Task 3: Running Predictive Models...")
    ml_insights = modeling.run_predictive_model(df_clean, metadata, target_col)
    print(f"--> Found {len(ml_insights)} ML insights.")
    return ml_insights

def run_storytelling_task(all_insights, df_clean, output_dir):
    """
    Runs the narrative and visualization generation task.
    """
    print("\nPipeline Task 4: Generating Story and Visuals...")
    report_insights = []
    for insight in all_insights:
        narrative = storytelling.generate_narrative_from_insight(insight)
        plot_path = storytelling.create_visualization(insight, df_clean, output_dir)
        
        report_insights.append({
            "title": insight['title'],
            "narrative": narrative,
            "plot_path": plot_path
        })
    return report_insights
