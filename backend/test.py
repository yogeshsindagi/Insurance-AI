import joblib
import xgboost as xgb

# 1. Load the old model
model = joblib.load(r"C:/Users/Yogesh/OneDrive/Documents/Insurance AI/backend/insurance_premium_model.pkl")

# 2. Extract the underlying XGBoost Booster and save it properly
# This assumes your 'model' is an XGBRegressor or XGBClassifier
if hasattr(model, "get_booster"):
    model.save_model("insurance_premium_model.json")
    print("Model converted to JSON format successfully!")
else:
    # If it's a pure joblib/sklearn pipeline, just re-saving in current version helps
    joblib.dump(model, "insurance_premium_model_updated.pkl")
    print("Model re-saved using current version.")