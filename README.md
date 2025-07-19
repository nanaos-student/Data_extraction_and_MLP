# Data_extraction_and_MLP
Extract edge based data from a live SUMO simulation, then use a dense MLP to predict travel times across edges (roads).

extract_split_coordinates_rt.py:
Extracts data from an ongoing SUMO simulation at regular intervals.
Sorts the data by edge coordinates into zones.
Output as csv files.


zone_B_capped_multipliers.py and zone_A_capped_multipliers.py:
Uses a dense MLP to predict travel times across edges.
Standardise inputs.
Square-root transformation to targets.
Evaluate model using MAE, MSE, RMSE, R^2
Save trained model as .pkl to be use during inference.
