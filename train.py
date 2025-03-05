import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
from scraper import TEAM_NAME_MAPPING, REVERSE_TEAM_MAPPING

def preprocess_data(df):
    """
    Prepares data for training by:
    - Converting date columns to datetime and sorting chronologically.
    - Creating temporal features for each player.
    - Encoding player names and opponents as categorical features.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Encode categorical features: player name and opponent.
    player_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    opponent_encoder = LabelEncoder()
    
    df['player_id'] = player_encoder.fit_transform(df['player_name'])
    df['team_id'] = team_encoder.fit_transform(df['team_name'])
    df['opponent_id'] = opponent_encoder.fit_transform(df['opponent'])
    
    # Define feature columns. Here we include player_id so that the model learns player-specific patterns.
    feature_cols = [
        'player_id',            # Categorical feature representing player identity
        'opponent_id',          # Categorical feature representing the opponent
        'team_id',              # Categorical feature representing the team
        'opponent_win_rate_before',
        'team_win_rate_before',
        'player_avg_kills_before',
        'player_avg_deaths_before',
        'player_avg_assists_before',
        'player_std_kills_before',
        'player_std_deaths_before',
        'player_std_assists_before'
    ]
    
    features = df[feature_cols]
    targets = df[['kills', 'deaths', 'assists']]
    
    return features, targets, player_encoder, team_encoder, opponent_encoder

# Function to train a unified multi-output model using a time-series split.
def train_unified_model(features, targets):
    """
    Trains a multi-output model with LightGBM, using a time series split.
    The categorical features (player_id and opponent_id) are passed to LightGBM
    via the categorical_feature parameter.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Wrap LightGBM regressor in a MultiOutputRegressor to predict all three targets at once.
    base_model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=50,
        learning_rate=0.01,
        n_estimators=500
    )
    model = MultiOutputRegressor(base_model)
    
    rmses, maes, r2s = [], [], []
    
    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
        
        # Fit the model. Note: LightGBM can handle categorical features if passed as indices.
        # Here, player_id is the first column and opponent_id the second.
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds, multioutput='uniform_average')
        
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
    
    print(f"Average RMSE: {np.mean(rmses):.2f}")
    print(f"Average MAE: {np.mean(maes):.2f}")
    print(f"Average R2 Score: {np.mean(r2s):.2f}")
    
    return model

def tune_model(features, targets):
    """
    Tune hyperparameters of the MultiOutput LightGBM model using GridSearchCV and TimeSeriesSplit.
    """
    # Define the parameter grid. Note the prefix "estimator__" because we are tuning the base estimator.
    param_grid = {
        'estimator__num_leaves': [31, 50, 64, 80],            
        'estimator__learning_rate': [0.01, 0.05, 0.1],        
        'estimator__n_estimators': [200, 500, 1000, 1500],  
        'estimator__max_depth': [-1, 5, 10, 15],                
        'estimator__min_child_samples': [5, 10],                
        'estimator__subsample': [0.8, 1.0],                    
        'estimator__colsample_bytree': [0.8, 1.0],                               
    }
    
    # Create the base model without fixed hyperparameters.
    base_model = lgb.LGBMRegressor(objective='regression')
    multi_output_model = MultiOutputRegressor(base_model)
    
    # Set up time series cross-validation.
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        multi_output_model, 
        param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error',  # Using MSE (negative because higher is better)
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(features, targets)
    
    print("Best parameters: ", grid_search.best_params_)
    print("Best score (MSE): ", -grid_search.best_score_)
    
    return grid_search.best_estimator_

# Full pipeline function: preprocess, train, and save the model and encoders.
def train_and_save_model(df):
    features, targets, player_enc, team_enc, opponent_enc = preprocess_data(df)
    model = tune_model(features, targets)
    
    # Save the trained model and encoders for future predictions.
    joblib.dump(model, 'kda_model.pkl')
    joblib.dump(player_enc, 'player_encoder.pkl')
    joblib.dump(team_enc, 'team_encoder.pkl')
    joblib.dump(opponent_enc, 'opponent_encoder.pkl')
    
    return model

# Prediction example: predicting KDA for a future game.
def predict_kda(player_name, team, opponent_team, df_features):
    """
    Predicts kills, deaths, and assists for a given player against a specified opponent.
    Automatically extracts the latest feature values from df_features.
    """
    # Load saved artifacts.
    model = joblib.load('kda_model.pkl')
    player_enc = joblib.load('player_encoder.pkl')
    opponent_enc = joblib.load('opponent_encoder.pkl')
    team_enc = joblib.load('team_encoder.pkl')

    # Get the latest match row for the player
    player_rows = df_features[df_features["player_name"] == player_name]
    if player_rows.empty:
        raise ValueError(f"No data found for player: {player_name}")
    
    latest_match = player_rows.iloc[-1]  # Most recent match
    
    # Define the feature columns (excluding actual KDA values)
    feature_cols = [
        "opponent_win_rate_before", "team_win_rate_before",
        "player_avg_kills_before", "player_avg_deaths_before", "player_avg_assists_before",
        "player_std_kills_before", "player_std_deaths_before", "player_std_assists_before"
    ]
    
    # Extract features as a dictionary
    input_df = pd.DataFrame([latest_match[feature_cols]])
    
    # Encode categorical values
    input_df['player_id'] = player_enc.transform([player_name])[0]
    input_df['opponent_id'] = opponent_enc.transform([opponent_team])[0]
    input_df['team_id'] = team_enc.transform([team])[0]
    
    # Ensure features match training format
    feature_cols = ['player_id', 'opponent_id', 'team_id'] + feature_cols
    input_features = input_df[feature_cols]

    # Predict KDA
    prediction = model.predict(input_features)
    return prediction[0]

def predict_game_kda(team_a, team_b, df_features):
    """
    Given two team names and a processed features DataFrame,
    predict the KDA for every player in the game.
    
    For players on team_a, the opponent is team_b, and vice versa.
    Returns a dictionary with player names as keys and predicted KDA arrays as values.
    """
    if team_a not in df_features["team_name"].unique():
        if team_a in TEAM_NAME_MAPPING:
            team_a = TEAM_NAME_MAPPING[team_a]
    
    # Map team_b to full name if necessary.
    if team_b not in df_features["team_name"].unique():
        if team_b in TEAM_NAME_MAPPING:
            team_b = TEAM_NAME_MAPPING[team_b]

    predictions = {}
    
    # Get the list of unique players for team_a and team_b
    team_a_players = df_features[df_features["team_name"] == team_a]["player_name"].unique()
    team_b_players = df_features[df_features["team_name"] == team_b]["player_name"].unique()
    
    # For each player in team_a, predict their KDA against team_b
    for player in team_a_players:
        try:
            if team_b in REVERSE_TEAM_MAPPING:
                reverse_team_b = REVERSE_TEAM_MAPPING[team_b]
            pred = predict_kda(player, team_a, reverse_team_b, df_features)
            predictions[player] = pred
        except Exception as e:
            predictions[player] = f"Error: {e}"
    
    # For each player in team_b, predict their KDA against team_a
    for player in team_b_players:
        try:
            if team_a in REVERSE_TEAM_MAPPING:
                reverse_team_a = REVERSE_TEAM_MAPPING[team_a]
            pred = predict_kda(player, team_b, reverse_team_a, df_features)
            predictions[player] = pred
        except Exception as e:
            predictions[player] = f"Error: {e}"

    for player, pred in predictions.items():
        print(f"{player}: {pred}")

    return predictions


