import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import impectPy as ip
from src.apostador.config import APIFOOTBALL_RAW_DIR, IMPECT_RAW_DIR, ODDS_RAW_DIR
from src.apostador.duckdb_store import write_json

# Load environment variables
load_dotenv()

USERNAME_IMPECT = os.getenv("USERNAME_IMPECT")
PASSWORD_IMPECT = os.getenv("PASSWORD_IMPECT")
APIFOOTBALL_KEY = os.getenv("APIFOOTBALL")
ODDSAPI_KEY = os.getenv("ODDSAPI")

# Setup directories

def test_api_football(target_date):
    print(f"\n--- Testing API-Football for date: {target_date} ---")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": APIFOOTBALL_KEY}
    params = {"date": target_date}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        print(f"API-Football Status Code: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            fixtures = data.get("response", [])
            print(f"Total fixtures retrieved: {len(fixtures)}")
            
            # Save raw json
            raw_path = APIFOOTBALL_RAW_DIR / f"fixtures_{target_date}.json"
            write_json(raw_path, data)
            print(f"Saved raw fixtures to {raw_path}")
            
            if fixtures:
                # Display sample fixture
                sample = fixtures[0]
                fixture_id = sample.get("fixture", {}).get("id")
                league_name = sample.get("league", {}).get("name")
                country = sample.get("league", {}).get("country")
                home = sample.get("teams", {}).get("home", {}).get("name")
                away = sample.get("teams", {}).get("away", {}).get("name")
                print(f"Sample Fixture ID: {fixture_id} | {league_name} ({country}): {home} vs {away}")
            return data
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")
    return None

def test_odds_api():
    print("\n--- Testing The Odds API ---")
    # Get active sports first to see which ones are available
    url_sports = "https://api.the-odds-api.com/v4/sports/"
    params_sports = {"apiKey": ODDSAPI_KEY}
    try:
        resp = requests.get(url_sports, params=params_sports)
        print(f"Odds API Sports Status Code: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            return None
            
        sports_data = resp.json()
        raw_path_sports = ODDS_RAW_DIR / "sports.json"
        write_json(raw_path_sports, sports_data)
            
        soccer_sports = [s["key"] for s in sports_data if "soccer" in s["key"]]
        print(f"Total active soccer sports: {len(soccer_sports)}")
        print(f"Sample soccer sports: {soccer_sports[:5]}")
        
        target_sport = "soccer_brazil_campeonato"
        if soccer_sports:
            if "soccer_epl" in soccer_sports:
                target_sport = "soccer_epl"
            else:
                target_sport = soccer_sports[0]
                
        print(f"Fetching odds for: {target_sport}")
        url_odds = f"https://api.the-odds-api.com/v4/sports/{target_sport}/odds/"
        params_odds = {
            "apiKey": ODDSAPI_KEY,
            "regions": "eu",
            "markets": "h2h"
        }
        
        resp_odds = requests.get(url_odds, params=params_odds)
        print(f"Odds API Odds Status Code: {resp_odds.status_code}")
        if resp_odds.status_code == 200:
            odds_data = resp_odds.json()
            print(f"Total matches with odds retrieved: {len(odds_data)}")
            
            raw_path_odds = ODDS_RAW_DIR / f"odds_{target_sport}.json"
            write_json(raw_path_odds, odds_data)
            print(f"Saved raw odds to {raw_path_odds}")
            
            if odds_data:
                sample = odds_data[0]
                home = sample.get("home_team")
                away = sample.get("away_team")
                print(f"Sample Odds Match: {home} vs {away}")
                bookmakers = sample.get("bookmakers", [])
                if bookmakers:
                    print(f"Bookmaker: {bookmakers[0].get('title')}")
                    markets = bookmakers[0].get("markets", [])
                    if markets:
                        print(f"Market Outcomes: {markets[0].get('outcomes')}")
            return odds_data
        else:
            print(f"Error: {resp_odds.text}")
    except Exception as e:
        print(f"Exception: {e}")
    return None

def test_impect_api():
    print("\n--- Testing Impect API via impectPy ---")
    try:
        token = ip.getAccessToken(username=USERNAME_IMPECT, password=PASSWORD_IMPECT)
        print("Authenticated successfully.")
        
        # Iterations
        iterations = ip.getIterations(token=token)
        print(f"Iterations retrieved. Count: {len(iterations)}")
        write_json(IMPECT_RAW_DIR / "iterations.json", json.loads(iterations.to_json(orient="records")))
        
        # Test getting squads ratings for iteration 1207 (Liga Dimayor I 2025)
        # Check if 1207 is present
        if 1207 in iterations["id"].values:
            target_iter = 1207
        else:
            target_iter = int(iterations.iloc[0]["id"])
            
        print(f"Fetching ratings for iteration {target_iter}...")
        ratings = ip.getSquadRatings(iteration=target_iter, token=token)
        print(f"Ratings retrieved. Shape: {ratings.shape}")
        write_json(IMPECT_RAW_DIR / f"ratings_{target_iter}.json", json.loads(ratings.to_json(orient="records")))
        
        print(f"Fetching averages for iteration {target_iter}...")
        averages = ip.getSquadIterationAverages(iteration=target_iter, token=token)
        print(f"Averages retrieved. Shape: {averages.shape}")
        write_json(IMPECT_RAW_DIR / f"averages_{target_iter}.json", json.loads(averages.to_json(orient="records")))
        
        return True
    except Exception as e:
        print(f"Impect API Error: {e}")
    return False

if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    test_api_football(today_str)
    test_api_football(tomorrow_str)
    test_odds_api()
    test_impect_api()
