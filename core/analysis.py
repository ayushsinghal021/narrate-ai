from scipy.stats import f_oneway
import pandas as pd

def get_statistical_insights(df: pd.DataFrame, metadata: dict):
    """
    Runs statistical tests to find interesting patterns. This version uses a robust,
    step-by-step method for correlation analysis to be compatible with type checkers.
    """
    insights = []
    numeric_cols = [col for col in metadata['numeric_cols'] if df[col].nunique() > 1]

    # --- NEW: Robust Correlation Analysis ---
    if len(numeric_cols) > 1:
        # 1. Calculate the correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # 2. Unstack the matrix to get a Series of all pairs
        s = corr_matrix.unstack()

        # 3. Convert the Series to a DataFrame and rename columns
        corr_df = s.reset_index()
        corr_df.columns = ['feature1', 'feature2', 'correlation']

        # 4. Remove self-correlations (e.g., HappinessScore vs. HappinessScore)
        corr_df = corr_df[corr_df['feature1'] != corr_df['feature2']]

        # 5. Create a sorted tuple of the feature pair to identify and remove duplicates
        #    (e.g., treat (A,B) and (B,A) as the same)
        corr_df['sorted_pair'] = corr_df.apply(lambda row: tuple(sorted((row['feature1'], row['feature2']))), axis=1)
        corr_df = corr_df.drop_duplicates(subset='sorted_pair')
        corr_df = corr_df.drop(columns=['sorted_pair']) # We don't need this column anymore

        # 6. Sort by the absolute value of the correlation to find the strongest relationships
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values(by='abs_correlation', ascending=False)

        # 7. Take the top 5 strongest correlations and create insights
        top_correlations = corr_df.head(5)
        for index, row in top_correlations.iterrows():
            insights.append({
                "type": "correlation",
                "title": f"Strong Correlation between {row['feature1']} and {row['feature2']}",
                "details": {
                    "feature1": row['feature1'],
                    "feature2": row['feature2'],
                    "correlation": row['correlation']
                }
            })
    # --- END of new correlation logic ---


    # --- ANOVA Test for Significant Differences ---
    categorical_cols_for_anova = [col for col in metadata['categorical_cols'] if 2 < df[col].nunique() < 15]
    
    if categorical_cols_for_anova and numeric_cols:
        cat_col = categorical_cols_for_anova[0]
        num_col = next((c for c in numeric_cols if 'score' in c.lower()), numeric_cols[0])

        try:
            groups = [df[num_col][df[cat_col] == category] for category in df[cat_col].unique()]
            f_stat, p_value = f_oneway(*groups)
            
            if p_value < 0.05: # Statistically significant
                insights.append({
                    "type": "significant_difference",
                    "title": f"Significant Difference in '{num_col}' across '{cat_col}'",
                    "details": {
                        "numeric_feature": num_col,
                        "categorical_feature": cat_col,
                        "p_value": p_value
                    }
                })
        except Exception as e:
            print(f"Could not perform ANOVA test. Reason: {e}")
            
    return insights