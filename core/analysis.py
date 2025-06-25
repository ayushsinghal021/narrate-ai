from scipy.stats import f_oneway
import pandas as pd

def get_statistical_insights(df, metadata):
    """Runs statistical tests to find interesting patterns."""
    insights = []
    numeric_cols = metadata['numeric_cols']
    
    # 1. Correlation Analysis
    corr_matrix = df[numeric_cols].corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.6:
                insights.append({
                    "type": "correlation",
                    "title": f"Strong Correlation between {col1} and {col2}",
                    "details": {
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": corr_value
                    }
                })

    # 2. ANOVA Test for significant differences
    # Find a good target numeric column and a categorical column with a few groups
    categorical_cols_for_anova = [col for col in metadata['categorical_cols'] if 2 < df[col].nunique() < 10]
    
    if categorical_cols_for_anova and numeric_cols:
        # Pick one of each for demonstration
        cat_col = categorical_cols_for_anova[0]
        num_col = 'MonthlyCharges' if 'MonthlyCharges' in numeric_cols else numeric_cols[0]
        
        groups = [df[num_col][df[cat_col] == category] for category in df[cat_col].unique()]
        f_stat, p_value = f_oneway(*groups)
        
        if p_value < 0.05: # Statistically significant
            insights.append({
                "type": "significant_difference",
                "title": f"Significant Difference in '{num_col}' across '{cat_col}' categories",
                "details": {
                    "numeric_feature": num_col,
                    "categorical_feature": cat_col,
                    "p_value": p_value
                }
            })
            
    return insights