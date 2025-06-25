import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

def run_predictive_model(df, metadata, target_col):
    """
    Trains an XGBoost model to find key drivers for a user-selected target column.
    It automatically chooses between Classification and Regression.
    """
    if not target_col or target_col not in df.columns:
        return [] # Cannot run model without a valid target

    insights = []
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encode categorical features for better model performance
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- NEW: Automatically choose model type ---
    # If target is binary (or has few unique values), classify. Otherwise, regress.
    if y.nunique() <= 10: 
        model_type = "Classification"
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        model_type = "Regression"
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    # --- END OF NEW LOGIC ---

    model.fit(X_train, y_train)

    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(5)

    insights.append({
        "type": "feature_importance",
        "title": f"Top 5 Predictors of '{target_col}' ({model_type})",
        "details": {
            "target": target_col,
            "features": importances.to_dict('records')
        }
    })
    
    return insights