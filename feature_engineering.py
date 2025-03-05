import pandas as pd
from scraper import fetch_team_statistics, fetch_teams_from_league
import sys
import bisect

TEAM_CACHE = {}  # Format: {team_name: {"dates": [...], "win_rates": [...]}

########################################
# Utility Functions
########################################

def parse_game_str(game_str):
    """
    Given a game string in the format "TeamA vs TeamB",
    return a tuple (TeamA, TeamB).
    """
    teams = game_str.split(" vs ")
    if len(teams) != 2:
        print("Invalid match string format. Expected 'TEAM_A vs TEAM_B'.")
        return None
    team_a, team_b = teams
    return team_a, team_b

def get_opponent(game_str, player_team):
    """
    Given a game string (e.g. "OK BRION vs Dplus KIA") and the player's team,
    return the opponent team.
    """
    teams = parse_game_str(game_str)
    if not teams:
        return None
    if player_team == teams[0]:
        return teams[1]
    else:
        return teams[0]

def split_score(score_str):
    """
    If Score is like "3/1/9", return a pd.Series with (3, 1, 9).
    Otherwise, return (None, None, None).
    """
    if not isinstance(score_str, str):
        return pd.Series([None, None, None])
    parts = score_str.split("/")
    if len(parts) == 3:
        try:
            return pd.Series([int(parts[0]), int(parts[1]), int(parts[2])])
        except ValueError:
            pass
    return pd.Series([None, None, None])

def find_similar_strength_teams(opponent_name, team_winrates, threshold=0.1):
    """
    Find teams with a similar win rate to the given opponent.
    """
    if opponent_name not in team_winrates:
        return []
    opponent_winrate = team_winrates[opponent_name]
    similar_teams = [
        t for t, w in team_winrates.items()
        if abs(w - opponent_winrate) <= threshold and t != opponent_name
    ]
    return similar_teams

def compute_avg_kda_vs_similar_opponents(opponent_name, player_df, team_winrates):
    """
    Compute a player's KDA (Kills, Deaths, Assists) against similar-strength opponents
    if the player hasn't faced the exact opponent.
    """
    similar_teams = find_similar_strength_teams(opponent_name, team_winrates)
    df_vs_opponent = player_df[player_df["Opponent"] == opponent_name]
    if not df_vs_opponent.empty:
        return (df_vs_opponent["Kills"].mean(),
                df_vs_opponent["Deaths"].mean(),
                df_vs_opponent["Assists"].mean())
    
    # Fallback: use matches against teams with similar win rate
    df_vs_similar = player_df[player_df["Opponent"].isin(similar_teams)]
    if df_vs_similar.empty:
        return (0, 0, 0)
    
    return (df_vs_similar["Kills"].mean(),
            df_vs_similar["Deaths"].mean(),
            df_vs_similar["Assists"].mean())

def get_historical_win_rate(team_history, target_date):
    """Get win rate as of target_date using binary search"""
    if not team_history or "dates" not in team_history:
        return 0.5  # Default for unknown teams
    
    dates = team_history["dates"]
    win_rates = team_history["win_rates"]
    
    # Find last index where date < target_date
    idx = bisect.bisect_left(dates, target_date) - 1
    return win_rates[idx] if idx >= 0 else 0.5

########################################
# Core Feature Engineering
########################################

def build_features(player_dfs, team_name_map, teams_to_include=None):
    """
    Build features for player match data.
    If teams_to_include is provided, only process players from those teams.
    """
    feature_rows = []
    
    for player_name, df in player_dfs.items():
        team = team_name_map[player_name]
        # Only filter if teams_to_include is provided
        if teams_to_include is not None and team not in teams_to_include:
            continue
            
        df = df.sort_values("Date").reset_index(drop=True)
        df["Opponent"] = df["Game"].apply(lambda game: get_opponent(game, team))
        df["Date"] = pd.to_datetime(df["Date"])
        
        if "Score" in df.columns:
            df[["Kills", "Deaths", "Assists"]] = df["Score"].apply(split_score)
        
        # Rolling player stats calculations
        for stat in ["Kills", "Deaths", "Assists"]:
            df[f"player_avg_{stat.lower()}_before"] = df[stat].expanding().mean().shift(1).fillna(0)
            df[f"player_std_{stat.lower()}_before"] = df[stat].expanding().std().shift(1).fillna(0)
        
        # Process each match and add feature rows
        for idx, row in df.iterrows():
            current_date = row["Date"]
            opponent = row["Opponent"]
            
            # Determine team win rate before the match using TEAM_CACHE
            team_match_idx = -1
            for i, m in enumerate(TEAM_CACHE.get(team, {}).get("match_history", [])):
                if m["date"] == current_date and m["opponent"] == opponent:
                    team_match_idx = i
                    break
                    
            team_win_rate = (
                TEAM_CACHE[team]["win_rates"][team_match_idx - 1] 
                if team_match_idx > 0 
                else 0.5
            )
            
            # Determine opponent win rate before the match
            opponent_team = opponent
            opponent_match_idx = -1
            for i, m in enumerate(TEAM_CACHE.get(opponent_team, {}).get("match_history", [])):
                if m["date"] == current_date and m["opponent"] == team:
                    opponent_match_idx = i
                    break

            opponent_win_rate = (
                TEAM_CACHE[opponent_team]["win_rates"][opponent_match_idx - 1] 
                if opponent_match_idx > 0 
                else 0.5
            )
            
            feature_rows.append({
                "player_name": player_name,
                "team_name": team,
                "opponent": opponent,
                "date": current_date,
                "opponent_win_rate_before": opponent_win_rate,
                "kills": row["Kills"],
                "deaths": row["Deaths"],
                "assists": row["Assists"],
                "player_avg_kills_before": row["player_avg_kills_before"],
                "player_avg_deaths_before": row["player_avg_deaths_before"],
                "player_avg_assists_before": row["player_avg_assists_before"],
                "player_std_kills_before": row["player_std_kills_before"],
                "player_std_deaths_before": row["player_std_deaths_before"],
                "player_std_assists_before": row["player_std_assists_before"],
                "team_win_rate_before": team_win_rate,
            })
    
    return pd.DataFrame(feature_rows)


########################################
# Process Match String
########################################

def process_match_string(match_string, url="https://gol.gg/teams/list/season-S15/split-Winter/tournament-ALL/"):
    """
    Splits a match string of the form 'TEAM_A vs TEAM_B'
    and returns a DataFrame of features for all players in the match.
    """
    team_a, team_b = parse_game_str(match_string)

    # Fetch team statistics for both teams
    fetch_and_cache_team(team_a, url)
    fetch_and_cache_team(team_b, url)
    _fetch_opponents_for_team(team_a, url)  
    _fetch_opponents_for_team(team_b, url)

    player_dfs = {}
    team_name_map = {}
    
    # Collect data for all relevant teams
    all_teams = [team_a, team_b]
    for team in [team_a, team_b]:
        all_teams.extend([
            match["opponent"]
            for match in TEAM_CACHE.get(team, {}).get("match_history", [])
        ])
    
    # Deduplicate teams
    for team in set(all_teams):
        if team in TEAM_CACHE:
            team_stats = fetch_team_statistics(team, url)
            player_dfs.update(team_stats)
            team_name_map.update({p: team for p in team_stats.keys()})
    
    return build_features(player_dfs, team_name_map, team_a, team_b)

def process_league(league_name, url="https://gol.gg/teams/list/season-S15/split-Winter/tournament-ALL/"):
    """
    Processes the entire league data to build features.
    """
    # Get all teams in the league
    teams = fetch_teams_from_league(league_name)
        
    # Populate the TEAM_CACHE for each team and their opponents
    for team in teams:
        fetch_and_cache_team(team, url)
        _fetch_opponents_for_team(team, url)
    
    player_dfs = {}
    team_name_map = {}
    
    # Fetch player statistics for each team
    for team in teams:
        team_stats = fetch_team_statistics(team, url)
        if team_stats:
            player_dfs.update(team_stats)
            team_name_map.update({p: team for p in team_stats.keys()})
    
    # Build features for all teams (pass all teams to include)
    return build_features(player_dfs, team_name_map, teams_to_include=teams)

def fetch_and_cache_team(team, url):
    if team in TEAM_CACHE:
        return
    
    # Scrape team data
    team_stats = fetch_team_statistics(team, url)
    if not team_stats:
        return

    seen_games = set()
    all_matches = []
    
    # Extract deduplicated matches from all players
    for player_df in team_stats.values():
        for _, row in player_df.iterrows():
            # Extract key components
            try:
                # Convert MM:SS duration to total seconds
                mins, secs = map(int, row["Duration"].split(':'))
                duration = mins * 60 + secs
            except (ValueError, AttributeError):
                duration = None  # Handle missing/invalid duration
            date = pd.to_datetime(row["Date"])
            opponent = get_opponent(row["Game"], team)
            
            # Create composite key
            composite_key = (date, opponent, duration)
            
            if composite_key not in seen_games:
                all_matches.append({
                    "date": date,
                    "opponent": opponent,
                    "result": row["Result"],
                    "duration": duration
                })
                seen_games.add(composite_key)
    
    # Sort matches chronologically
    all_matches.sort(key=lambda x: x["date"])
    
    # Compute expanding win rates
    cumulative_wins = 0
    win_rates = []
    for i, match in enumerate(all_matches):
        cumulative_wins += 1 if match["result"] == "Victory" else 0
        win_rate = cumulative_wins / (i + 1) if (i + 1) > 0 else 0.5
        win_rates.append(win_rate)
    
    # Update cache
    TEAM_CACHE[team] = {
        "match_history": all_matches,
        "win_rates": win_rates
    }


def _fetch_opponents_for_team(team, url):
    if team not in TEAM_CACHE:
        return
    
    # Extract opponents from team's match history
    opponents = set()
    for match in TEAM_CACHE[team]["match_history"]:
        opponent = match["opponent"]
        opponents.add(opponent)
    
    # Fetch and cache opponents
    for opponent in opponents:
        if opponent not in TEAM_CACHE:
            fetch_and_cache_team(opponent, url)

########################################
# Main
########################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_engineering.py <MatchString>")
        sys.exit(1)
    match_string = " ".join(sys.argv[1:])
    df_features = process_match_string(match_string)
    if df_features is not None:
        print("Feature columns:", df_features.columns.tolist())
        print(df_features.head())
