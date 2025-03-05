from feature_engineering import process_match_string, build_features
from scraper import fetch_team_statistics

def predict_match(match_string):
    # 1. parse "OK BRION vs Dplus KIA"
    team_a, team_b, team_a_map, team_b_map = process_match_string(match_string)

    # 2. get stats
    team_a_winrate, team_a_stats = fetch_team_statistics(team_a)
    team_b_winrate, team_b_stats = fetch_team_statistics(team_b)

    # 3. combine
    all_player_dfs = {**team_a_stats, **team_b_stats}
    combined_map = {**team_a_map, **team_b_map}

    # example of your existing approach to store team_name -> winrate
    team_winrates = {
        team_a: team_a_winrate,
        team_b: team_b_winrate,
        # or any other teams you might need
    }

    # 4. build combined feature DF
    df_features = build_features(all_player_dfs, combined_map, team_winrates)

    # 5. run XGBoost prediction
    xgb_model = load_xgboost_model("my_kills_model.json")
    feature_cols = [
        "player_avg_kills_before",
        "team_win_rate_before",
        "opp_allowed_kills_before",
        "player_avg_kills_vs_opponent",
        "player_avg_deaths_vs_opponent",
        "player_avg_assists_vs_opponent",
        # etc...
    ]
    preds = xgb_model.predict(df_features[feature_cols])

    # attach predictions
    df_features["predicted_kills"] = preds

    return df_features

# Then you can do:
if __name__ == "__main__":
    match_str = "OK BRION vs Dplus KIA"
    df_result = predict_match(match_str)
    print(df_result[["player_name", "team_name", "predicted_kills"]])