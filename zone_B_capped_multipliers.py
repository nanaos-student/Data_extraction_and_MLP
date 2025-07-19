# model_keras.py - Complete solution for Keras with TensorFlow Lite
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import all datasets
training_data = pd.read_csv("capped_global_training_dataset_zone_B.csv")
live_demo = pd.read_csv("C:/Users/nanao/Sumo/Live Demo/zone_B_5min.csv")

# Clean data by removing leading quote characters
training_df = training_data.map(lambda x: x.lstrip("'") if isinstance(x, str) else x)
live_df = live_demo.map(lambda x: x.lstrip("'") if isinstance(x, str) else x)

# Define columns to exclude from features
features_to_drop = ["edge_id", "type", "from_junction", "to_junction", "zone", "is_border",
                    "connected_edges_other_zone", "avg_travel_time", "max_travel_time"]

# Extract features and targets for all datasets
X_training = training_df.drop(features_to_drop, axis=1).values
y_training = np.sqrt(training_df["avg_travel_time"].values)


X_live = live_df.drop(features_to_drop, axis=1).values
y_live = live_df["avg_travel_time"].values

# Save list of feature names for reference
feature_names = [col for col in training_df.columns if col not in features_to_drop]
print(f"Using {len(feature_names)} features: {feature_names}")

# Feature Scaling
scaler_X = StandardScaler()
X_training_scaled = scaler_X.fit_transform(X_training)
X_live_scaled = scaler_X.transform(X_live)

model = keras.Sequential([
    keras.layers.Input(shape=(X_training_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='relu')  # Output layer
])

# Compile the model -
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error')

# Train the model
start_time = time.time()
history = model.fit(X_training_scaled, y_training,
batch_size=128,
epochs=50,
validation_split=0.2,
callbacks=[
keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
],verbose=1)

# Save the trained model and the fitted scaler
joblib.dump(model, "zone_B_mlp_tf_rl.pkl")
joblib.dump(scaler_X, "zone_B_scaler_tf_rl.pkl")

# Generate predictions for all density levels
print("Step 5: Making predictions on all density levels")
y_pred_live_scaled = model.predict(X_live_scaled).flatten()

# Apply inverse transform (square the predictions)
y_pred_live = y_pred_live_scaled ** 2

# Evaluation & Metrics
live_original_mean = live_df["avg_travel_time"].mean()
live_original_mean2 = np.mean(np.square(y_live))
live_original_rms = np.sqrt(np.mean(np.square(y_live)))


print(f"Mean of live demo = {live_original_mean}")


live_mae = mean_absolute_error(y_live, y_pred_live)
live_mse = mean_squared_error(y_live, y_pred_live)  # Mean Squared Error (MSE)
live_rms_e = np.sqrt(live_mse)  # Root Mean Squared Error (RMSE)
live_r2 = r2_score(y_live, y_pred_live)  # R-squared

live_error_percent = (live_mae/live_original_mean)*100
live_mse_error_percent = (live_mse/live_original_mean2)*100
live_rms_epercent = (live_rms_e/live_original_rms) * 100


if live_error_percent > 15:
    print(f"Mean Absolute Error as a % of mean for Live Demo = {live_error_percent}")
    print("Error is too high, try fixing it.")
else:
    print(f"Mean Absolute Error as a % of mean for Live Demo {live_error_percent}")
    print("Well done. Error under control.")

print(f"\nlive Mean Squared Error as a % of original mean squared = {live_mse_error_percent}%")
print(f"live RMSE as a % of original RMS = {live_rms_epercent:.2f}%")
print(f"live R-squared = {live_r2:.4f}")

# Output LPG
live_output_df = pd.DataFrame({
    'edge_id': live_df['edge_id'], 'type': live_df['type'], 'from_junction': live_df['from_junction'],
    'to_junction': live_df['to_junction'], 'zone': live_df['zone'], 'is_border': live_df['is_border'],
    'connected_edges_other_zone': live_df['connected_edges_other_zone'],
    'predicted_travel_time': y_pred_live})

# Save predictions to CSV
#live_output_df.to_csv('LPG_for_live_demo_zone_B.csv', index=False)
