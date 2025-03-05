import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys

TEAM_NAME_MAPPING = {
    # LPL (China)
    "RNG": "Royal Never Give Up",
    "IG": "Invictus Gaming",
    "FPX": "Funplus Phoenix",
    "EDG": "Edward Gaming",
    "WE": "Team WE",
    "NIP": "Ninjas in Pyjamas",
    
    # LEC (Europe)
    "G2": "G2 Esports",
    "FNC": "Fnatic",
    "MAD": "MAD Lions",
    "GX": "GIANTX",
    "KC": "Karmine Corp",
    "BDS": "Team BDS",
    "MKOI": "Movistar KOI",
    "VIT": "Team Vitality",
    "TH": "Team Heretics",
    
    # LCS (North America)
    "C9": "Cloud9",
    "TL": "Team Liquid",
    "TSM": "TSM",
    
    # LCK (Korea)
    "T1": "T1",
    "DK": "Dplus KIA",
    "GEN": "Gen.G eSports",
    "DRX": "DRX",
    "HLE": "Hanwha Life eSports",
    "NS": "Nongshim RedForce",
    "KT": "KT Rolster",
    "BRO": "OK BRION",
    "DNF": "DN Freecs",
    "BFX": "BNK FearX",

    # LCP (Taiwan/Japan)
    "GAM": "GAM Esports",
    "PSG": "PSG Talon",

    # TCL (Turkey)
    "EF": "Eternal Fire",
    "SUP": "Papara SuperMassive",
    "BJK": "Besiktas Esports",
    "BGT": "BoostGate Esports",
    "MISA": "Misa Esports",
    "ULF": "ULF Esports",
    "BLDP": "BBL Dark Passage",
    "BW": "Bushido Wildcats",

}

REVERSE_TEAM_MAPPING = {v: k for k, v in TEAM_NAME_MAPPING.items()}


def fetch_page(url):
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/58.0.3029.110 Safari/537.3')
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def parse_table(table):
    """Extract headers and rows from an HTML table."""
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = []
    for row in table.find_all('tr')[1:]:  # skip header row
        cells = row.find_all('td')
        if len(cells) != len(headers):
            continue
        row_data = [cell.get_text(strip=True) for cell in cells]
        rows.append(dict(zip(headers, row_data)))
    return headers, rows


def parse_statistics(url):
    """
    Fetch a player's matchlist page, extract statistics, and return a DataFrame
    with only the 'Game', 'Result', and 'Score' columns.
    """
    # Adjust URL to get matchlist
    matchlist_url = url.replace("player-stats", "player-matchlist").replace("champion-ALL/", "")
    page_content = fetch_page(matchlist_url)
    if not page_content:
        return pd.DataFrame()
    
    soup = BeautifulSoup(page_content, 'html.parser')
    
    # Get player name from <h1> if available
    h1_tag = soup.find('h1')
    player_name = h1_tag.get_text(" ", strip=True) if h1_tag else "Unknown"
    
    table = soup.find('table', class_='table_list')
    if not table:
        print("No table found on player's matchlist page.")
        return pd.DataFrame()
    
    headers, rows = parse_table(table)

    # Filter to only the columns of interest
    filtered = [{
        "Game": row.get("Game", ""),
        "Result": row.get("Result", ""),
        "Score": row.get("Score", ""),
        "Date": row.get("Date", ""),
        "Duration": row.get("Duration", "")
    } for row in rows]
    
    return pd.DataFrame(filtered)


def fetch_team_urls(team_name=None, url="https://gol.gg/teams/list/season-S15/split-Winter/tournament-ALL/"):
    """
    Fetch team URLs from the listing page.
    If team_name is provided, return its URL (if found). Otherwise, return a dict of all teams.
    """
    possible_names = [
        team_name,
        TEAM_NAME_MAPPING.get(team_name, team_name),
        REVERSE_TEAM_MAPPING.get(team_name, team_name),
        TEAM_NAME_MAPPING.get(team_name.upper(), team_name)
    ]

    page_content = fetch_page(url)
    if not page_content:
        return None

    soup = BeautifulSoup(page_content, 'html.parser')
    team_table = soup.find('table', class_='table_list playerslist tablesaw trhover')
    teams = {}
    for row in team_table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if not cells:
            continue
        listed_name = cells[0].get_text(strip=True)
        
        # Check all possible name variations
        for name in possible_names:
            if name.lower() == listed_name.lower():
                relative_url = cells[0].find('a')['href'][1:]
                full_url = f"https://gol.gg/teams{relative_url}"
                teams[listed_name] = full_url
                return full_url  # Return first match
    
    print(f"Team '{team_name}' not found. Tried variations: {possible_names}")
    return None


def fetch_player_data(team_url):
    """
    Given a team URL, fetch and return a dict mapping player names to their profile URLs.
    """
    page_content = fetch_page(team_url)
    if not page_content:
        print(f"Failed to fetch page for team at {team_url}.")
        return {}

    soup = BeautifulSoup(page_content, 'html.parser')
    player_table = None

    # Locate the table with the player's stats using its caption text
    for table in soup.find_all("table", class_="table_list"):
        caption_tag = table.find("caption")
        if caption_tag and "player's stats" in caption_tag.get_text(strip=True):
            player_table = table
            break

    if not player_table:
        print("No player table found on team page.")
        return {}
    
    tbody = player_table.find("tbody")
    if not tbody:
        print("No <tbody> found in player table.")
        return {}

    player_urls = {}
    for row in tbody.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
        a_tag = tds[1].find("a")
        if a_tag:
            player_name = a_tag.get_text(strip=True)
            relative_url = a_tag.get("href", "")
            if relative_url.startswith(".."):
                relative_url = relative_url[2:]
            elif relative_url.startswith("."):
                relative_url = relative_url[1:]
            full_url = f"https://gol.gg{relative_url}"
            player_urls[player_name] = full_url

    return player_urls

def fetch_teams_from_league(league_name):
    league_map = {
        "LPL": "https://gol.gg/tournament/tournament-ranking/LPL%202025%20Split%201%20Playoffs/",
        "LEC": "https://gol.gg/tournament/tournament-ranking/LEC%202025%20Winter%20Playoffs/",
        "LCK": "https://gol.gg/tournament/tournament-ranking/LCK%20Cup%202025/",
        "LTAN": "https://gol.gg/tournament/tournament-ranking/LTA%20North%202025%20Split%201/",
        "LTAS": "https://gol.gg/tournament/tournament-ranking/LTA%20South%202025%20Split%201/",
        "LCP": "https://gol.gg/tournament/tournament-ranking/LCP%202025%20Season%20Kickoff/",
        "TCL": "https://gol.gg/tournament/tournament-ranking/TCL%202025%20Winter/",
    }

    team_names = set()

    page_content = fetch_page(league_map.get(league_name, league_name))
    soup = BeautifulSoup(page_content, 'html.parser')
    table = soup.find("table", class_="table_list")
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if not cells:
            continue
        listed_name = cells[0].get_text(strip=True)
        if listed_name:
            team_names.add(listed_name)
    
    return team_names


def fetch_team_statistics(team_name, url="https://gol.gg/teams/list/season-S15/split-Winter/tournament-ALL/"):
    """
    Fetch all player statistics for a team. Returns a dictionary mapping
    each player name to their statistics DataFrame.
    """
    team_url = fetch_team_urls(team_name, url)
    if not team_url:
        print(f"Team '{team_name}' not found.")
        return None

    players = fetch_player_data(team_url)
    if not players:
        print("No players found for the team.")
        return None

    team_stats = {}
    for player_name, url in players.items():
        stats_df = parse_statistics(url)
        team_stats[player_name] = stats_df
    return team_stats

def fetch_league_statistics(league_name):
    teams = fetch_teams_from_league(league_name)
    print(teams)
    league_stats = {}
    for team in teams:
        league_stats[team] = fetch_team_statistics(team)
    
    return league_stats    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <TeamName>")
        sys.exit(1)

    # Support multi-word team names
    team_name = ' '.join(sys.argv[1:])

    # Example: fetch and print team player statistics
    stats_dict = fetch_league_statistics(team_name)
    if stats_dict is not None:
        for player, df in stats_dict.items():
            print(f"Statistics for {player}:")
            print(df)
            print("-" * 40)

