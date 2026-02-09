import matplotlib.pyplot as plt
import xgboost as xgb

# Load the model
model = xgb.XGBRegressor()
model.load_model("xgb_cinema_model_v5.json")

# Plot importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15, importance_type='weight')
plt.title("Top 15 Features Driving Cinepolis Predictions")
plt.show()