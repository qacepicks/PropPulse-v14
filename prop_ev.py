#!/usr/bin/env python3
# prop_ev.py — PropPulse+ v2025.3 (FIXED VERSION)
# L20-weighted projection + FantasyPros DvP + Auto position + Manual odds entry

# ===============================
# IMPORTS
# ===============================
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
import os, json, time, math
from datetime import datetime, timezone
from dvp_updater import load_dvp_data
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams, players
import pytz
import math
import glob
import platform
import subprocess
import sys
print("DEBUG: stdin.isatty() =", sys.stdin.isatty())

# ===============================
# ENHANCED DISPLAY SYSTEM
# ===============================
from display_results import (
    display_top_props,
    show_by_probability,
    show_high_confidence,
    show_by_stat_type,
    display_summary_stats,
    export_to_csv,
    export_to_markdown,
    interactive_display
)

# ===============================
# AUTO FETCH PLAYER LOGS
# ===============================
def load_player_logs(player):
    """Loads local logs; if missing, auto-fetches from BallDontLie API."""
    safe_name = player.lower().replace('.', '').replace("'", "").replace(' ', '_')
    path = f"data/{safe_name}.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    # --- Auto-fetch fallback ---
    print(f"[Data] 🆕 Fetching logs for missing player: {player}")
    try:
        resp = requests.get(
            f"https://www.balldontlie.io/api/v1/players?search={player}"
        ).json()
        if not resp["data"]:
            raise ValueError("Player not found on BallDontLie")

        player_id = resp["data"][0]["id"]
        games = requests.get(
            f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page=100"
        ).json()["data"]

        df = pd.DataFrame([
            {
                "PTS": g.get("pts", 0),
                "REB": g.get("reb", 0),
                "AST": g.get("ast", 0),
                "FG3M": g.get("fg3m", 0),
                "game_date": g.get("game", {}).get("date", None)
            }
            for g in games
        ])

        if "game_date" not in df.columns:
            df["game_date"] = None

        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[Data] ✅ Cached new log → {path}")
        return df

    except Exception as e:
        print(f"[Data] ❌ Failed to fetch {player}: {e}")
        return pd.DataFrame(columns=["PTS","REB","AST","FG3M"])

# ===============================
# GLOBAL DEFAULTS
# ===============================
pace_mult = 1.0
is_home = None

# --- Import model calibration constants ---
from calibration import (
    TEMP_Z, BALANCE_BIAS, CLIP_MIN, CLIP_MAX,
    MULT_CENTER, MULT_MAX_DEV, EMP_PRIOR_K, W_EMP_MAX,
    INCLUDE_LAST_SEASON, SHRINK_TO_LEAGUE
)

# ===============================
# LOAD DVP DATA
# ===============================
dvp_data = load_dvp_data()

# ==========================================
# 🔧 Balanced Scoring & Probability System
# ==========================================
TEMP_Z = 1.45
BALANCE_BIAS = 0.20
CLIP_MIN, CLIP_MAX = 0.08, 0.92

MULT_CENTER = 1.00
MULT_MAX_DEV = 0.15

EMP_PRIOR_K = 20
W_EMP_MAX = 0.30

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _cap(x, lo, hi):
    return max(lo, min(hi, x))

def normalize_multiplier(raw: float,
                         center: float = MULT_CENTER,
                         max_dev: float = MULT_MAX_DEV) -> float:
    """Normalize DvP / pace / homeaway multipliers to stay near 1.0"""
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    log_m = math.log(raw / center)
    log_m *= 0.5
    m = math.exp(log_m)
    return _cap(m, 1.0 - max_dev, 1.0 + max_dev)

def adjusted_mean(mu_base: float,
                  multipliers: dict,
                  league_mu: float | None = None,
                  shrink_to_league: float = 0.10) -> float:
    """Apply normalized multipliers & optional shrink toward league average."""
    prod = 1.0
    for k, v in (multipliers or {}).items():
        prod *= normalize_multiplier(float(v))
    mu = mu_base * prod
    if league_mu is not None and math.isfinite(league_mu):
        mu = (1.0 - shrink_to_league) * mu + shrink_to_league * league_mu
    return mu

def prob_over_from_normal(mu: float, sigma: float, line: float,
                          temp_z: float = TEMP_Z) -> float:
    """Flattened z-score probability (smaller slope -> less extreme results)."""
    eps = 1e-9
    sigma_eff = max(sigma, eps)
    z = (line - mu) / sigma_eff
    z /= max(1.0, temp_z)
    return float(1.0 - norm.cdf(z))

def smooth_empirical_prob(vals: np.ndarray, line: float, k: int = EMP_PRIOR_K) -> float:
    """Empirical hit rate vs the line with Beta prior smoothing toward 50%."""
    n = int(vals.size)
    hits = int((vals > line).sum())
    alpha = hits + 0.5 * k
    beta = (n - hits) + 0.5 * k
    return alpha / (alpha + beta)

def finalize_prob(p_raw: float,
                  balance_bias: float = BALANCE_BIAS,
                  clip_min: float = CLIP_MIN,
                  clip_max: float = CLIP_MAX) -> float:
    """Pull probabilities gently toward 50% and clip to safe range."""
    p = (1.0 - balance_bias) * p_raw + balance_bias * 0.5
    return _cap(p, clip_min, clip_max)

# ==========================================
# CORE WRAPPER — Balanced Probability
# ==========================================
def calibrated_prob_over(
    mu_base: float,
    sigma_base: float,
    line: float,
    multipliers: dict,
    recent_vals: np.ndarray,
    league_mu: float | None = None
) -> float:
    """
    Returns a balanced P(over) after:
    1) Normalizing multipliers
    2) Flattening z-score (temperature)
    3) Blending in empirical hit-rate (sample-size aware)
    4) Applying a small balance bias toward 50%
    5) Clipping to sane bounds
    """
    mu = adjusted_mean(mu_base, multipliers, league_mu=league_mu, shrink_to_league=0.10)
    p_model = prob_over_from_normal(mu, sigma_base, line, temp_z=TEMP_Z)

    p_emp = smooth_empirical_prob(np.array(recent_vals, dtype=float), line)
    n = int(len(recent_vals))
    w_emp = min(W_EMP_MAX, n / (n + EMP_PRIOR_K))
    p_blend = (1 - w_emp) * p_model + w_emp * p_emp

    return finalize_prob(p_blend, balance_bias=BALANCE_BIAS,
                         clip_min=CLIP_MIN, clip_max=CLIP_MAX)

# ==========================================
# EV / ODDS
# ==========================================
def american_to_prob(odds: int) -> float:
    return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)

def net_payout(odds: int) -> float:
    return 100/abs(odds) if odds < 0 else odds/100

def ev_per_dollar(p: float, odds: int) -> float:
    """Expected value per $1 wager"""
    return p * net_payout(odds) - (1 - p)

def ev_sportsbook(p, odds):
    return p * net_payout(odds) - (1 - p)

# ===============================
# CONFIG
# ===============================
def load_settings():
    default = {
        "default_sportsbook": "Fliff",
        "default_region": "us",
        "data_path": "data/",
        "injury_api_key": "YOUR_SPORTSDATAIO_KEY",
        "balldontlie_api_key": "YOUR_BALLDONTLIE_KEY",
        "cache_hours": 24
    }
    path = "settings.json"

    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f, indent=4)
        print("[Config] Created new settings.json.")
        return default

    with open(path, "r") as f:
        settings = json.load(f)

    for k, v in default.items():
        if k not in settings:
            settings[k] = v

    os.makedirs(settings["data_path"], exist_ok=True)
    return settings

def get_bdl(endpoint, params=None, settings=None, timeout=10):
    """Universal BallDon'tLie API caller - FIXED VERSION"""
    base_url = "https://api.balldontlie.io/v1"
    
    # ✅ FIX: Handle None settings
    if settings is None:
        settings = {"balldontlie_api_key": "free"}
    
    api_key = settings.get("balldontlie_api_key", "free")
    
    headers = {}
    if api_key and api_key.lower() != "free" and api_key != "YOUR_BALLDONTLIE_KEY":
        headers["Authorization"] = f"Bearer {api_key}"
    
    url = f"{base_url}{endpoint}"
    params = params or {}
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        
        print(f"[BDL] GET {endpoint} with params={params}")
        print(f"[BDL] Status: {r.status_code}")
        
        if r.status_code == 401:
            print(f"[BDL] ❌ 401 Unauthorized")
            if api_key and api_key != "YOUR_BALLDONTLIE_KEY":
                print(f"[BDL] Using key: {api_key[:10]}...{api_key[-4:]}")
            else:
                print("[BDL] ❌ No valid API key configured in settings.json")
            return None
            
        elif r.status_code == 429:
            print("[BDL] ⚠️ Rate limited - waiting 2s...")
            time.sleep(2)
            return None
            
        elif r.status_code == 404:
            print(f"[BDL] ⚠️ 404 Not Found: {url}")
            return None
            
        elif r.status_code == 403:
            print(f"[BDL] ❌ 403 Forbidden")
            return None
            
        elif r.status_code == 200:
            data = r.json()
            result_count = len(data.get('data', []))
            print(f"[BDL] ✅ Success - returned {result_count} records")
            return data
        else:
            print(f"[BDL] ⚠️ Unexpected status code: {r.status_code}")
            return None
        
    except requests.exceptions.Timeout:
        print(f"[BDL] ⚠️ Request timeout after {timeout}s")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[BDL] ⚠️ Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[BDL] ❌ Failed to parse JSON response: {e}")
        return None

# ===============================
# ✅ FIXED: POSITION DETECTION
# ===============================
def get_player_position_auto(player_name, df_logs=None, settings=None):
    """Automatically fetches player position using BallDontLie V1 API."""
    try:
        print(f"[Position] 🔍 Searching BallDontLie for '{player_name}'...")
        
        last_name = player_name.split()[-1]
        print(f"[Position] Searching by last name: '{last_name}'")
        
        data = get_bdl("/players", {"search": last_name}, settings)
        
        if data and "data" in data and len(data["data"]) > 0:
            print(f"[Position] Found {len(data['data'])} matching player(s)")
            
            for player in data["data"]:
                full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                print(f"[Position] Checking: {full_name}")
                if full_name.lower() == player_name.lower():
                    pos = player.get("position", "").strip().upper()
                    if pos:
                        pos = normalize_position(pos)
                        print(f"[Position] ✅ BallDontLie V1 (exact match) → {pos}")
                        return pos
            
            pos = data["data"][0].get("position", "").strip().upper()
            if pos:
                pos = normalize_position(pos)
                first_match = f"{data['data'][0].get('first_name', '')} {data['data'][0].get('last_name', '')}"
                print(f"[Position] ✅ BallDontLie V1 (using first result: {first_match}) → {pos}")
                return pos
        else:
            print(f"[Position] ⚠️ BallDontLie returned no results for '{last_name}'")
        
    except Exception as e:
        print(f"[Position] ⚠️ BallDontLie error: {e}")
    
    print(f"[Position] 🔍 Using enhanced stat-based inference...")
    if df_logs is not None and len(df_logs) > 0:
        return infer_position_from_stats(df_logs, player_name)
    
    print("[Position] ⚠️ No data available, defaulting to SF")
    return "SF"

def normalize_position(pos):
    """Normalize position abbreviations to standard 5 positions"""
    pos = pos.upper().strip()
    
    position_map = {
        "G": "SG",
        "G-F": "SF",
        "F": "SF", 
        "F-G": "SF",
        "F-C": "PF",
        "C-F": "C"
    }
    
    return position_map.get(pos, pos) if pos in position_map else pos

def infer_position_from_stats(df_logs, player_name=""):
    """Improved position inference using multiple statistical indicators."""
    def avg(col):
        if col not in df_logs.columns:
            return 0.0
        return pd.to_numeric(df_logs[col], errors="coerce").fillna(0).mean()
    
    a_pts = avg("PTS")
    a_reb = avg("REB")
    a_ast = avg("AST")
    a_fg3 = avg("FG3M")
    
    print(f"[Position] Stats: PTS={a_pts:.1f} REB={a_reb:.1f} AST={a_ast:.1f} 3PM={a_fg3:.1f}")
    
    if a_ast >= 6.5:
        print(f"[Position] 🔍 Inferred PG (high AST: {a_ast:.1f})")
        return "PG"
    
    if a_reb >= 9 and a_fg3 < 1.0:
        print(f"[Position] 🔍 Inferred C (high REB: {a_reb:.1f}, low 3PM)")
        return "C"
    
    if a_reb >= 7.5 and a_ast < 5.5:
        print(f"[Position] 🔍 Inferred PF (REB: {a_reb:.1f}, AST: {a_ast:.1f})")
        return "PF"
    
    if 4.5 <= a_reb <= 8 and 3 <= a_ast <= 6 and a_pts >= 12:
        print(f"[Position] 🔍 Inferred SF (balanced: PTS={a_pts:.1f}, REB={a_reb:.1f}, AST={a_ast:.1f})")
        return "SF"
    
    if a_ast >= 3 and a_reb < 5.5 and (a_fg3 >= 1.5 or a_pts >= 15):
        print(f"[Position] 🔍 Inferred SG (AST: {a_ast:.1f}, REB: {a_reb:.1f})")
        return "SG"
    
    if a_reb >= 7:
        print(f"[Position] 🔍 Default to PF (high REB: {a_reb:.1f})")
        return "PF"
    elif a_reb >= 5:
        print(f"[Position] 🔍 Default to SF (moderate REB: {a_reb:.1f})")
        return "SF"
    else:
        print(f"[Position] 🔍 Default to SG")
        return "SG"

# ===============================
# ✅ FIXED: OPPONENT DETECTION
# ===============================
def get_upcoming_opponent_abbr(player_name, settings=None):
    """Uses BallDontLie V1 to pull the player's next opponent - FIXED VERSION"""
    
    # ✅ FIX: Ensure settings exists
    if settings is None:
        settings = {"balldontlie_api_key": "free"}
    
    try:
        print(f"[Schedule] 🔍 Looking up next game for {player_name}...")
        
        last_name = player_name.split()[-1]
        print(f"[Schedule] Searching by last name: '{last_name}'")
        
        player_data = get_bdl("/players", {"search": last_name}, settings)
        
        if not player_data or "data" not in player_data or not player_data["data"]:
            print(f"[Schedule] ⚠️ Player not found via API")
            return None, None
        
        player = None
        for p in player_data["data"]:
            full_name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            if full_name.lower() == player_name.lower():
                player = p
                print(f"[Schedule] ✅ Found exact match: {full_name}")
                break
        
        if not player:
            player = player_data["data"][0]
            fallback_name = f"{player.get('first_name', '')} {player.get('last_name', '')}"
            print(f"[Schedule] Using first result: {fallback_name}")
        
        team = player.get("team", {})
        team_id = team.get("id")
        team_abbr = team.get("abbreviation", "UNK")
        
        if not team_id:
            print("[Schedule] ⚠️ No team ID found")
            return None, None
        
        print(f"[Schedule] Player team: {team_abbr} (ID: {team_id})")
        
        today = datetime.now(timezone.utc)
        season = today.year if today.month >= 10 else today.year - 1
        
        games_data = get_bdl("/games", {
            "seasons[]": season,
            "team_ids[]": team_id,
            "per_page": 100
        }, settings)
        
        if not games_data or "data" not in games_data:
            print("[Schedule] ⚠️ No games data returned")
            return None, None
        
        games = games_data["data"]
        print(f"[Schedule] Found {len(games)} games for season {season}")
        
        today_date = today.date()
        future_games = []
        
        for game in games:
            try:
                game_date_str = game.get("date", "")
                game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00")).date()
                
                if game_date >= today_date:
                    future_games.append((game_date, game))
            except Exception:
                continue
        
        if not future_games:
            print("[Schedule] ⚠️ No upcoming games found")
            return None, None
        
        future_games.sort(key=lambda x: x[0])
        next_game_date, next_game = future_games[0]
        
        home_team = next_game.get("home_team", {})
        visitor_team = next_game.get("visitor_team", {})
        
        if home_team.get("id") == team_id:
            opp_abbr = visitor_team.get("abbreviation")
            location = "vs"
        else:
            opp_abbr = home_team.get("abbreviation")
            location = "@"
        
        if opp_abbr:
            print(f"[Schedule] ✅ Next game: {next_game_date} {location} {opp_abbr}")
            return opp_abbr, team_abbr
        
        return None, None
        
    except Exception as e:
        print(f"[Schedule] ⚠️ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ===============================
# ✅ FIXED: get_live_opponent_from_schedule
# ===============================
def get_live_opponent_from_schedule(player, settings=None):
    """
    ✅ FIXED VERSION - Resilient opponent finder
    Uses game_header (team IDs) + static teams to resolve abbreviations.
    """
    try:
        from nba_api.stats.static import teams as nba_teams
        import pandas as pd
        import os
        from datetime import datetime
        
        # ✅ FIX: Ensure settings exists
        if settings is None:
            settings = {"data_path": "data"}
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Step 1: Get player's team from local CSV
        path = os.path.join(settings.get("data_path", "data"), f"{player.replace(' ', '_')}.csv")
        
        if not os.path.exists(path):
            print(f"[Schedule] ⚠️ No local logs for {player}")
            return get_upcoming_opponent_abbr(player, settings)
        
        df = pd.read_csv(path)
        
        team_abbr = None
        if "TEAM_ABBREVIATION" in df.columns:
            team_abbr = df["TEAM_ABBREVIATION"].mode()[0] if len(df) > 0 else None
        
        if not team_abbr:
            print(f"[Schedule] ⚠️ No team found in logs for {player}")
            return get_upcoming_opponent_abbr(player, settings)
        
        # Step 2: Check today's scoreboard
        try:
            board = scoreboardv2.ScoreboardV2(game_date=today)
            game_header = board.game_header.get_data_frame()
            
            if not game_header.empty:
                # ✅ FIX: Use static teams instead of CommonTeamRoster
                all_nba_teams = nba_teams.get_teams()
                id_to_abbr = {t["id"]: t["abbreviation"] for t in all_nba_teams}
                abbr_to_id = {t["abbreviation"]: t["id"] for t in all_nba_teams}
                
                team_id = abbr_to_id.get(team_abbr)
                
                if team_id:
                    for _, game in game_header.iterrows():
                        home_id = game["HOME_TEAM_ID"]
                        away_id = game["VISITOR_TEAM_ID"]
                        
                        if home_id == team_id:
                            opp_abbr = id_to_abbr.get(away_id)
                            print(f"[Schedule] ✅ {team_abbr} vs {opp_abbr} (today)")
                            return opp_abbr, team_abbr
                        elif away_id == team_id:
                            opp_abbr = id_to_abbr.get(home_id)
                            print(f"[Schedule] ✅ {team_abbr} @ {opp_abbr} (today)")
                            return opp_abbr, team_abbr
        
        except Exception as e:
            print(f"[Schedule] ⚠️ Scoreboard check failed: {e}")
        
        # Step 3: Fall back to upcoming opponent API
        print(f"[Schedule] ℹ️ {team_abbr} not playing today, checking next game...")
        return get_upcoming_opponent_abbr(player, settings)
        
    except Exception as e:
        print(f"[Schedule] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ===============================
# INJURY STATUS
# ===============================
def get_injury_status(player_name, api_key):
    if not api_key or "YOUR_SPORTSDATAIO_KEY" in api_key:
        return None
    try:
        url = "https://api.sportsdata.io/v4/nba/scores/json/Players"
        r = requests.get(url, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=8)
        if r.status_code != 200:
            return None
        for p in r.json():
            if player_name.lower() in p.get("Name", "").lower():
                return p.get("InjuryStatus", None)
    except Exception:
        return None
    return None

# ===============================
# UNIVERSAL PLAYER LOG FETCHER
# ===============================
def fetch_player_logs(player_name, save_dir="data/", settings=None, include_last_season=True):
    """Unified fetcher: tries BallDontLie V1 first, then Basketball Reference."""
    import requests
    import pandas as pd
    import os
    from datetime import datetime
    from bs4 import BeautifulSoup
    import time

    # === 1. Try BallDontLie V1 API first ===
    try:
        print(f"[BDL] Trying V1 API for {player_name}...")
        last_name = player_name.split()[-1]
        
        player_data = get_bdl("/players", {"search": last_name}, settings)
        
        if player_data and "data" in player_data and len(player_data["data"]) > 0:
            player = None
            for p in player_data["data"]:
                full_name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                if full_name.lower() == player_name.lower():
                    player = p
                    break
            
            if not player:
                player = player_data["data"][0]
            
            player_id = player.get("id")
            print(f"[BDL] ✅ Found {player_name} (ID {player_id})")
            
            today = datetime.now()
            if today.month >= 10:
                season = today.year - 1
            elif today.month <= 6:
                season = today.year - 1
            else:
                season = today.year
            print(f"[BDL] Calculated season: {season} (for {season}-{season+1} NBA season)")
            
            stats_data = get_bdl("/stats", {
                "player_ids[]": player_id,
                "seasons[]": season,
                "per_page": 100
            }, settings)
            
            if stats_data and "data" in stats_data:
                games = stats_data["data"]
                print(f"[BDL] Retrieved {len(games)} games for season {season}")
                
                if games:
                    rows = []
                    for g in games:
                        mins_raw = g.get("min", "0")
                        try:
                            if ":" in str(mins_raw):
                                mins = float(mins_raw.split(":")[0])
                            else:
                                mins = float(mins_raw or 0)
                        except:
                            mins = 0.0
                        
                        rows.append({
                            "GAME_ID": g.get("game", {}).get("id", ""),
                            "DATE": g.get("game", {}).get("date", ""),
                            "PTS": g.get("pts", 0),
                            "REB": g.get("reb", 0),
                            "AST": g.get("ast", 0),
                            "FG3M": g.get("fg3m", 0),
                            "MIN": mins
                        })
                    
                    df = pd.DataFrame(rows)
                    df = df[df["MIN"] > 0]
                    
                    if len(df) > 0:
                        team_abbr = player.get("team", {}).get("abbreviation", "UNK")
                        if "TEAM_ABBREVIATION" not in df.columns:
                            df["TEAM_ABBREVIATION"] = team_abbr
                        else:
                            df["TEAM_ABBREVIATION"].fillna(team_abbr, inplace=True)

                        if "MATCHUP" not in df.columns:
                            team_full = player.get("team", {}).get("full_name", "")
                            df["MATCHUP"] = [team_full] * len(df)

                        path = os.path.join(save_dir, f"{player_name.replace(' ', '_')}.csv")
                        os.makedirs(save_dir, exist_ok=True)
                        df.to_csv(path, index=False)
                        print(f"[Save] ✅ {len(df)} games saved → {path}")
                        print(f"[Meta] 🏀 Team = {team_abbr}")
                        return df

                print(f"[BDL] ⚠️ No data found via V1 API")

    except Exception as e:
        print(f"[BDL] ❌ V1 API error: {e}")

    # === 2. Fallback: Basketball Reference ===
    print("[Fallback] Trying Basketball Reference...")

    def bbref_name_format(name):
        """Generate Basketball Reference player ID slug"""
        name = name.lower().replace(".", "").replace("'", "").replace("-", "")
        parts = name.split()
        if len(parts) < 2:
            return None
        last, first = parts[-1], parts[0]
        return f"{last[:5]}{first[:2]}01"

    slug = bbref_name_format(player_name)
    if not slug:
        print(f"[BBRef] ❌ Invalid name format: {player_name}")
        return None
    
    last_name = player_name.split()[-1]
    first_letter = last_name[0].lower()
    
    year = datetime.now().year
    if datetime.now().month < 10:
        year = year
    else:
        year = year + 1
    
    rows = []
    seasons_to_try = [year, year - 1] if include_last_season else [year]
    
    for yr in seasons_to_try:
        url = f"https://www.basketball-reference.com/players/{first_letter}/{slug}/gamelog/{yr}"
        print(f"[BBRef] Fetching {yr} season: {url}")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            }
            
            time.sleep(3)
            html = requests.get(url, headers=headers, timeout=15)
            
            if html.status_code == 403:
                print(f"[BBRef] ⚠️ 403 Forbidden for {yr}")
                continue
            elif html.status_code == 404:
                print(f"[BBRef] ⚠️ 404 Not Found for {yr}")
                continue
            elif html.status_code != 200:
                print(f"[BBRef] ⚠️ Status {html.status_code} for {yr}")
                continue
            
            soup = BeautifulSoup(html.text, "html.parser")
            table = soup.find("table", {"id": "pgl_basic"})
            
            if not table:
                print(f"[BBRef] ⚠️ No game log table found for {yr}")
                continue
            
            print(f"[BBRef] ✅ Found game log table for {yr}")
            
            for tr in table.find_all("tr"):
                if not tr.find("td"):
                    continue
                
                tds = [td.text.strip() for td in tr.find_all("td")]
                
                if len(tds) < 27:
                    continue
                
                try:
                    pts = float(tds[26]) if tds[26] else 0
                    reb = float(tds[22]) if tds[22] else 0
                    ast = float(tds[23]) if tds[23] else 0
                    fg3m = float(tds[11]) if tds[11] else 0
                    
                    mins_str = tds[6]
                    if mins_str and mins_str != "Did Not Play":
                        try:
                            if ":" in mins_str:
                                m, s = mins_str.split(":")
                                mins = int(m) + int(s) / 60.0
                            else:
                                mins = float(mins_str)
                        except:
                            mins = 0
                    else:
                        mins = 0
                    
                    if mins > 0:
                        rows.append({
                            "PTS": pts,
                            "REB": reb,
                            "AST": ast,
                            "FG3M": fg3m,
                            "MIN": mins
                        })
                except (ValueError, IndexError):
                    continue
            
            print(f"[BBRef] ✅ Parsed {len([r for r in rows if r])} games from {yr}")
            
        except Exception as e:
            print(f"[BBRef] ❌ Error fetching {yr}: {e}")

    if not rows:
        print(f"[BBRef] ❌ No data found for {player_name}")
        return None

    df = pd.DataFrame(rows)
    df = df[df["MIN"] > 0]
    path = os.path.join(save_dir, f"{player_name.replace(' ', '_')}.csv")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Save] ✅ {len(df)} games saved → {path}")
    return df

# ===============================
# L20-WEIGHTED MODEL
# ===============================
def l20_weighted_mean(vals: pd.Series) -> float:
    if len(vals) == 0:
        return 0.0
    season_mean = float(pd.to_numeric(vals, errors="coerce").fillna(0).mean())
    last20 = pd.to_numeric(vals.tail(20), errors="coerce").fillna(0)
    l20_mean = float(last20.mean()) if len(last20) > 0 else season_mean
    return 0.60 * l20_mean + 0.40 * season_mean

# ===============================
# PROBABILITY CALIBRATION
# ===============================
def prob_calibrate(p: float, T: float = 1.15, b: float = 0.0, shrink: float = 0.20) -> float:
    """Temperature-calibrates model probability and shrinks toward 0.5."""
    p = max(1e-6, min(1 - 1e-6, float(p)))
    logit = math.log(p / (1 - p))
    logit = (logit / T) + b
    q = 1.0 / (1.0 + math.exp(-logit))
    return 0.5 + (1.0 - shrink) * (q - 0.5)

# ===============================
# ✅ FIXED: GRADE PROBABILITIES
# ===============================
def grade_probabilities(df, stat_col, line, proj_mins, avg_mins, injury_status=None, dvp_mult=1.0):
    """Fixed version without player_name parameter"""
    if stat_col not in df.columns:
        if stat_col == "REB+AST":
            df["REB+AST"] = df["REB"] + df["AST"]
        elif stat_col == "PRA":
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        else:
            raise KeyError(f"Missing stat {stat_col}")

    vals = pd.to_numeric(df[stat_col], errors="coerce").fillna(0.0)
    n = len(vals)
    std = float(vals.std(ddof=0)) if n > 1 else 1.0
    mean = l20_weighted_mean(vals)

    mean *= (proj_mins / avg_mins) if avg_mins > 0 else 1.0

    if injury_status and str(injury_status).lower() not in ["active", "probable"]:
        mean *= 0.9
    elif proj_mins < avg_mins * 0.8:
        mean *= 0.9

    pace_mult = globals().get("pace_mult", 1.0)
    mean *= pace_mult
    print(f"[Model] Pace multiplier: {pace_mult:.3f}")

    is_home = globals().get("is_home", None)
    if is_home is True:
        mean *= 1.03
    elif is_home is False:
        mean *= 0.97

    if stat_col in ["REB", "REB+AST", "PRA"]:
        dvp_mult *= 0.85

    sigma_scale = {"PTS":1.00, "REB":0.85, "AST":0.90, "PRA":0.95, "REB+AST":0.88, "FG3M":1.10}
    std *= sigma_scale.get(stat_col, 1.0)

    mean *= float(dvp_mult)
    print(f"[Model] DvP (adjusted): {dvp_mult:.3f} | Adjusted mean: {mean:.2f}")

    recent_vals = np.array(vals, dtype=float)
    multipliers = {
        "dvp": dvp_mult,
        "pace": pace_mult,
        "ha": 1.0,
        "h2h": 1.0
    }

    p_final = calibrated_prob_over(
        mu_base=mean,
        sigma_base=std,
        line=line,
        multipliers=multipliers,
        recent_vals=recent_vals,
        league_mu=None
    )
    
    try:
        T = 1.15
        p_cal = prob_calibrate(p_final, T=T, b=0.0, shrink=0.20)
        print(f"[Calib] raw={p_final:.3f} → calibrated={p_cal:.3f} (T={T}, shrink=0.20)")
        return p_cal, n, mean
    except Exception as e:
        print(f"[Model] ❌ Failsafe triggered: {e}")
        safe_p = 0.5
        safe_n = n if 'n' in locals() else 0
        safe_mean = mean if 'mean' in locals() else 0.0
        return safe_p, safe_n, safe_mean

# ===============================
# DvP MULTIPLIER
# ===============================
def get_dvp_multiplier(opponent_abbr, position, stat_key):
    """Get DvP multiplier for a stat against an opponent."""
    try:
        if not opponent_abbr or not position or not stat_key:
            return 1.0
        
        opponent_abbr = opponent_abbr.upper()
        position = position.upper()
        stat_key = stat_key.upper()
        
        if opponent_abbr not in dvp_data:
            print(f"[DvP] ⚠️ Team {opponent_abbr} not in DvP data")
            return 1.0
        
        team_dict = dvp_data[opponent_abbr]
        
        if position not in team_dict:
            print(f"[DvP] ⚠️ Position {position} not found for {opponent_abbr}")
            return 1.0
        
        pos_dict = team_dict[position]
        
        if stat_key == "REB+AST":
            reb_rank = pos_dict.get("REB")
            ast_rank = pos_dict.get("AST")
            
            if reb_rank is not None and ast_rank is not None:
                avg_rank = (reb_rank + ast_rank) / 2.0
                multiplier = 1.1 - (avg_rank - 1) / 300
                print(f"[DvP] {opponent_abbr} vs {position}: REB+AST avg={avg_rank:.1f} → {multiplier:.3f}")
                return multiplier
            else:
                return 1.0
        
        elif stat_key == "PRA":
            pts_rank = pos_dict.get("PTS")
            reb_rank = pos_dict.get("REB")
            ast_rank = pos_dict.get("AST")
            
            if all(r is not None for r in [pts_rank, reb_rank, ast_rank]):
                avg_rank = (pts_rank + reb_rank + ast_rank) / 3.0
                multiplier = 1.1 - (avg_rank - 1) / 300
                print(f"[DvP] {opponent_abbr} vs {position}: PRA avg={avg_rank:.1f} → {multiplier:.3f}")
                return multiplier
            else:
                return 1.0
        
        else:
            rank = pos_dict.get(stat_key)
            
            if rank is None:
                print(f"[DvP] ⚠️ Stat {stat_key} not found for {opponent_abbr} {position}")
                return 1.0
            
            multiplier = 1.1 - (rank - 1) / 300
            print(f"[DvP] {opponent_abbr} vs {position} on {stat_key}: rank={rank} → {multiplier:.3f}")
            return multiplier
            
    except Exception as e:
        print(f"[DvP] ❌ Error calculating multiplier: {e}")
        return 1.0

# ===============================
# UTILITY FUNCTIONS
# ===============================
def get_rest_days(player, settings):
    """Estimate player's team's rest days before today's game."""
    try:
        result = get_live_opponent_from_schedule(player, settings)
        if result is None or len(result) != 2:
            return 1
        
        opp, team_abbr = result
        if team_abbr is None:
            return 1

        sched_path = os.path.join(settings["data_path"], f"{team_abbr}_schedule.csv")
        if not os.path.exists(sched_path):
            return 1

        import pandas as pd
        sched = pd.read_csv(sched_path)
        sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"])

        today = datetime.now().date()
        past_games = sched[sched["GAME_DATE"] < pd.Timestamp(today)]
        if len(past_games) == 0:
            return 1

        last_game_date = past_games["GAME_DATE"].max().date()
        rest_days = (today - last_game_date).days
        return rest_days

    except Exception as e:
        print(f"[Rest] ⚠️ Could not determine rest days: {e}")
        return 1

def get_team_total(player, settings):
    """Estimate projected team total (points) for a player's team."""
    import random

    team_avgs = {
        "BOS": 117.5, "DEN": 115.8, "SAC": 118.2, "LAL": 116.3,
        "MIL": 120.1, "DAL": 118.0, "GSW": 117.9, "NYK": 113.0,
        "OKC": 116.5, "MIA": 110.2, "PHI": 114.8, "PHX": 115.2,
        "CHI": 111.5, "CLE": 112.8, "MIN": 113.9, "NOP": 114.1,
        "ATL": 118.6, "TOR": 112.0, "BKN": 112.4, "MEM": 109.5,
        "ORL": 111.8, "HOU": 112.3, "CHA": 108.1, "POR": 107.9,
        "UTA": 113.2, "IND": 121.3, "DET": 109.0, "WAS": 112.6,
        "SA": 110.9, "LAC": 115.4
    }

    try:
        result = get_live_opponent_from_schedule(player, settings)
        if result is None or len(result) != 2:
            team_abbr = None
        else:
            opp, team_abbr = result
    except Exception:
        team_abbr = None

    if team_abbr is None:
        return 112.0 + random.uniform(-2, 2)

    team_total = team_avgs.get(team_abbr, 112.0)

    try:
        dvp_mult = get_dvp_multiplier(opp, "TEAM", "PTS")
    except Exception:
        dvp_mult = 1.0

    team_total *= dvp_mult
    return round(team_total, 1)

# ===============================
# STAT MAP
# ===============================
STAT_MAP = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "PRA": ["PTS", "REB", "AST"],
    "RA": ["REB", "AST"],
    "P+R": ["PTS", "REB"],
    "P+A": ["PTS", "AST"],
    "FG3M": "FG3M"
}

def get_usage_factor(player, stat, settings):
    """Estimate player usage factor based on shot volume or minutes trend."""
    try:
        path = os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv")
        if not os.path.exists(path):
            return 1.0

        df = pd.read_csv(path)
        stat = STAT_MAP.get(stat, stat)

        possible_cols = ["FGA", "USG%", "TOUCHES", "POSS"]
        usage_col = next((c for c in possible_cols if c in df.columns), None)

        if usage_col:
            last10 = df.tail(10)[usage_col].astype(float)
            season_avg = df[usage_col].astype(float).mean()
        elif "MIN" in df.columns:
            last10 = df.tail(10)["MIN"].astype(float)
            season_avg = df["MIN"].astype(float).mean()
        else:
            return 1.0

        if season_avg <= 0:
            return 1.0

        recent_avg = last10.mean()
        ratio = recent_avg / season_avg
        usage_mult = np.clip(ratio, 0.95, 1.05)

        print(f"[Usage] {player}: recent={recent_avg:.1f}, season={season_avg:.1f} → mult={usage_mult:.3f}")
        return usage_mult
    except Exception as e:
        print(f"[Usage] ⚠️ Error: {e}")
        return 1.0

def get_recent_form(df, stat_col, line):
    """Compute recent-form probability (L10 games) of going over the line."""
    try:
        if stat_col not in df.columns:
            print(f"[L10] ⚠️ Missing stat column: {stat_col}")
            return 0.5

        last10 = df.tail(10)[stat_col].astype(float)
        if len(last10) == 0:
            return 0.5

        p_l10 = np.mean(last10 > line)
        smoothed = 0.5 + (p_l10 - 0.5) * 0.8
        print(f"[L10] {stat_col} → {p_l10:.2f} raw → {smoothed:.2f} smoothed")
        return smoothed
    except Exception as e:
        print(f"[L10] ⚠️ Error: {e}")
        return 0.5

def get_homeaway_adjustment(player, stat, line, settings):
    """Return probability adjustment based on home/away splits."""
    try:
        df = pd.read_csv(os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv"))
        if "MATCHUP" not in df.columns:
            return 1.0

        home_games = df[~df["MATCHUP"].str.contains("@", na=False)]
        away_games = df[df["MATCHUP"].str.contains("@", na=False)]

        if len(home_games) < 5 or len(away_games) < 5:
            return 1.0

        stat = STAT_MAP.get(stat, stat)
        if stat not in df.columns:
            return 1.0

        home_mean = home_games[stat].mean()
        away_mean = away_games[stat].mean()
        overall_mean = df[stat].mean()

        if overall_mean == 0:
            return 1.0

        adj = (home_mean - away_mean) / overall_mean
        adj = np.clip(1 + adj * 0.2, 0.95, 1.05)
        print(f"[Home/Away] {player}: home={home_mean:.2f}, away={away_mean:.2f}, adj={adj:.3f}")
        return adj
    except Exception as e:
        print(f"[Home/Away] ⚠️ Error: {e}")
        return 1.0

def debug_projection(df, stat="REB+AST", line=12.5, player_name=""):
    """Debug helper to understand the projection breakdown"""
    
    if stat == "REB+AST" and stat not in df.columns:
        df["REB+AST"] = df["REB"] + df["AST"]
    elif stat == "PRA" and stat not in df.columns:
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    
    vals = pd.to_numeric(df[stat], errors="coerce").fillna(0.0)
    
    print("\n" + "="*60)
    print(f"🔍 DEBUG: {player_name} {stat} Projection Analysis")
    print("="*60)
    
    season_mean = vals.mean()
    season_std = vals.std()
    season_median = vals.median()
    print(f"\n📊 Full Season Stats ({len(vals)} games):")
    print(f"   Mean: {season_mean:.2f}")
    print(f"   Median: {season_median:.2f}")
    print(f"   Std Dev: {season_std:.2f}")
    print(f"   Min: {vals.min():.1f} | Max: {vals.max():.1f}")

    last20 = vals.tail(20)
    l20_mean = last20.mean()
    l20_median = last20.median()
    print(f"\n📈 Last 20 Games:")
    print(f"   Mean: {l20_mean:.2f}")
    print(f"   Median: {l20_median:.2f}")
    print(f"   Difference from season: {l20_mean - season_mean:+.2f}")

    over_count = (vals > line).sum()
    hit_rate = over_count / len(vals) * 100
    print(f"\n🎯 Historical Performance vs Line {line}:")
    print(f"   Over: {over_count}/{len(vals)} ({hit_rate:.1f}%)")
    print(f"   Under: {len(vals)-over_count}/{len(vals)} ({100-hit_rate:.1f}%)")
    print("="*60 + "\n")

# ===============================
# ✅ FIXED: analyze_single_prop
# ===============================
def analyze_single_prop(player, stat, line, odds, settings, debug_mode=False):
    """Analyze a single prop and return results dict - FIXED VERSION"""
    path = os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv")
    need_refresh = not os.path.exists(path) or (time.time() - os.path.getmtime(path)) / 3600 > 24

    if need_refresh:
        print(f"[Data] ⏳ Refreshing logs for {player}...")
        try:
            df = fetch_player_logs(player, save_dir=settings["data_path"], settings=settings)
            if df is None or len(df) == 0:
                raise ValueError("No data from BallDon'tLie")
            print("[Source] ✅ Using BallDon'tLie data")
        except Exception as e:
            print(f"[BDL] ⚠️ BallDon'tLie failed: {e}")
            print("[Fallback] Trying Basketball Reference instead...")
            df = fetch_player_logs(
                player,
                save_dir=settings["data_path"],
                settings=settings,
                include_last_season=True
            )
        if df is None or len(df) == 0:
            print(f"[Logs] ❌ Could not fetch logs for {player}.")
            return None
        print("[Source] ✅ Data loaded successfully")
    else:
        df = pd.read_csv(path)
        print(f"[Data] ✅ Loaded {len(df)} games for {player}")

    # Clean minutes
    if "MIN" in df.columns:
        def parse_minutes(val):
            if isinstance(val, str) and ":" in val:
                try:
                    m, s = val.split(":")
                    return int(m) + int(s) / 60
                except:
                    return 0.0
            try:
                return float(val)
            except:
                return 0.0

        df["MIN"] = df["MIN"].apply(parse_minutes)
        before = len(df)
        df = df[df["MIN"] > 0]
        after = len(df)
        print(f"[Clean] Removed {before - after} DNP games ({after} remain).")

    avg_mins = df["MIN"].mean() if "MIN" in df.columns and len(df["MIN"]) > 0 else 30
    proj_mins = avg_mins

    # Stat extraction
    stat_col = STAT_MAP.get(stat, stat)
    if isinstance(stat_col, list):
        missing = [col for col in stat_col if col not in df.columns]
        if missing:
            print(f"[Error] ❌ Missing columns for {stat}: {missing}")
            return None
        df["COMPOSITE"] = df[stat_col].sum(axis=1)
        vals = df["COMPOSITE"].astype(float)
        print(f"[Composite] Using {'+'.join(stat_col)} for {stat}")
    else:
        if stat_col not in df.columns:
            print(f"[Error] ❌ Stat '{stat}' not found in logs for {player}.")
            return None
        vals = df[stat_col].astype(float)

    mean = vals.mean()
    std = vals.std() if len(vals) > 1 else 0.0
    p_emp = np.mean(vals > line)
    p_norm = 1 - norm.cdf(line, mean, std if std > 0 else 1)
    p_base = 0.6 * p_norm + 0.4 * p_emp

    # Contextual probabilities
    p_ha = get_homeaway_adjustment(player, stat, line, settings)
    p_l10 = get_recent_form(df, stat_col if not isinstance(stat_col, list) else "COMPOSITE", line)
    p_dvp = p_base
    p_usage = get_usage_factor(player, stat, settings)

    # ✅ FIX: Safe opponent handling
    try:
        result = get_live_opponent_from_schedule(player, settings)
        
        if result is None or len(result) != 2:
            print(f"[Warning] Could not determine opponent, using neutral DvP")
            opp, team_abbr = None, None
        else:
            opp, team_abbr = result
    except Exception as e:
        print(f"[Error] Opponent lookup failed: {e}")
        opp, team_abbr = None, None

    pos = get_player_position_auto(player, df_logs=df, settings=settings)
    
    try:
        if opp and pos:
            dvp_mult = get_dvp_multiplier(opp, pos, stat)
        else:
            dvp_mult = 1.0
            print("[DvP] Using neutral multiplier (no opponent/position)")
    except Exception as e:
        print(f"[DvP] ⚠️ Could not apply DvP: {e}")
        dvp_mult = 1.0

    # Probability & Scoring
    n_games = len(df)
    maturity = min(1.0, n_games / 40)

    w_base = 0.30 + 0.20 * maturity
    w_l10 = 0.20 - 0.07 * maturity
    w_ha = 0.10
    w_dvp = 0.15 + 0.02 * maturity
    w_usage = 0.20 - 0.05 * maturity
    total_w = w_base + w_l10 + w_ha + w_dvp + w_usage
    w_base, w_l10, w_ha, w_dvp, w_usage = [w / total_w for w in (w_base, w_l10, w_ha, w_dvp, w_usage)]

    p_model = (
        p_base * w_base +
        p_l10 * w_l10 +
        p_ha * w_ha +
        p_dvp * w_dvp +
        p_usage * w_usage
    )

    print(
        f"[ProbStack] base={p_base:.3f}({w_base*100:.0f}%) | "
        f"L10={p_l10:.3f}({w_l10*100:.0f}%) | "
        f"HA={p_ha:.3f}({w_ha*100:.0f}%) | "
        f"DvP={p_dvp:.3f}({w_dvp*100:.0f}%) | "
        f"Usage={p_usage:.3f}({w_usage*100:.0f}%) → "
        f"Model={p_model:.3f}"
    )

    base_conf = 1 - (std / mean) if mean > 0 else 0.5
    confidence = max(0.1, base_conf * maturity)

    trend_strength = abs(p_l10 - 0.5) * 2
    matchup_strength = abs(dvp_mult - 1.0) / 0.15

    if (p_l10 > 0.5 and dvp_mult > 1.0) or (p_l10 < 0.5 and dvp_mult < 1.0):
        synergy_strength = min(1.0, (trend_strength + matchup_strength) / 2)
        boost = 1.03 + 0.05 * synergy_strength
        confidence = min(confidence * boost, 1.0)
        print(f"[Confidence] 🔼 Synergy boost {boost:.3f}× ({player})")
    elif (p_l10 > 0.55 and dvp_mult < 0.97) or (p_l10 < 0.45 and dvp_mult > 1.03):
        conflict_strength = min(1.0, (trend_strength + matchup_strength) / 2)
        penalty = 1.00 - 0.05 * conflict_strength
        confidence *= penalty
        print(f"[Confidence] 🔻 Counter-synergy penalty {penalty:.3f}× ({player})")

    # --- Stat-type reliability
    stat_baseline = {
        "PTS": 0.9, "REB": 1.1, "AST": 1.05, "PRA": 0.95,
        "RA": 0.95, "FG3M": 0.85, "STL": 0.75, "BLK": 0.70
    }
    stat_mult = stat_baseline.get(stat.upper(), 1.0)
    confidence = max(0.1, min(base_conf * maturity * stat_mult, 0.95))

    # --- Apply confidence to probability
    p_model *= (0.5 + 0.5 * confidence)
    p_model = max(0.05, min(0.95, p_model))

    # --- Confidence-weighted EV
    p_book = american_to_prob(odds)
    ev_raw = ev_sportsbook(p_model, odds)
    conf_weight = 0.5 + 0.5 * confidence
    ev = ev_raw * conf_weight
    edge = (p_model - p_book) * 100
    ev_cents = ev * 100

    print(f"[EV] ModelProb={p_model:.3f} | BookProb={p_book:.3f} | Edge={edge:+.2f}% | "
          f"EV(raw)={ev_raw*100:+.2f}¢ | ConfAdj={conf_weight:.2f}× → EV(final)={ev_cents:+.2f}¢")

    # --- Confidence penalty for noisy data
    if confidence < 0.5:
        ev_cents *= 0.75
        edge *= 0.85

    # --- Injury + Position
    inj = get_injury_status(player, settings.get("injury_api_key"))
    pos = get_player_position_auto(player, df_logs=df, settings=settings)

    # --- Game context multipliers
    dvp_mult = max(0.85, min(1.15, dvp_mult))
    pace_mult = 1.0
    league_avg_total = 112
    team_total = get_team_total(player, settings)
    team_total_mult = min(1.10, max(0.90, (team_total / league_avg_total) if team_total else 1.0))

    rest_days = get_rest_days(player, settings)
    rest_mult = {0: 0.96, 1: 1.00, 2: 1.03}.get(rest_days, 1.05)
    context_mult = dvp_mult * pace_mult * team_total_mult * rest_mult
    print(f"[Context] DvP={dvp_mult:.3f} | TeamTotal={team_total_mult:.3f} | Rest={rest_mult:.3f} → Final={context_mult:.3f}×")

    proj_stat = mean * context_mult

    # --- Grading System
    ev_score = ev_cents * confidence
    if ev_score >= 8.0:
        grade = "🔥 ELITE"
    elif ev_score >= 4.0:
        grade = "💎 SOLID"
    elif ev_score >= 1.0:
        grade = "⚖️ NEUTRAL"
    else:
        grade = "🚫 FADE"

    print(f"[Grade] EV¢={ev_cents:+.2f} | Conf={confidence:.2f} → Score={ev_score:.2f} → {grade}")

    # --- Debug Mode optional detailed printout
    if debug_mode:
        debug_projection(df, stat=stat, line=line, player_name=player)

    # --- Return structured result
    return {
        "player": player,
        "stat": stat,
        "line": line,
        "odds": odds,
        "opponent": opp,
        "position": pos,
        "n_games": n_games,
        "p_model": p_model,
        "p_book": p_book,
        "projection": round(proj_stat, 2),
        "ev": ev,
        "dvp_mult": dvp_mult,
        "injury": inj,
        "confidence": round(confidence, 2),
        "grade": grade
    }
# ===============================
# UTILITIES (contextual functions)
# ===============================
from nba_api.stats.endpoints import scoreboardv2, commonteamroster
from datetime import datetime
import pandas as pd, os

def get_live_opponent_from_schedule(player, settings=None):
    """
    ✅ PropPulse+ 2025.6 — Resilient opponent finder
    Uses game_header (team IDs) + commonteamroster to resolve abbreviations.
    Falls back to next opponent if no game today.
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        board = scoreboardv2.ScoreboardV2(game_date=today)
        game_header = board.game_header.get_data_frame()

        if game_header.empty:
            print(f"[Schedule] ℹ️ No games today — checking next opponent...")
            return get_upcoming_opponent_abbr(player, settings)

        # --- Grab home/away team IDs
        pairs = list(zip(game_header["HOME_TEAM_ID"], game_header["VISITOR_TEAM_ID"]))

        # --- Load player's team abbreviation from local logs
        path = os.path.join("data", f"{player.replace(' ', '_')}.csv")
        if not os.path.exists(path):
            print(f"[Schedule] ⚠️ No local logs for {player}.")
            return None, None

        df = pd.read_csv(path)
        if "TEAM_ABBREVIATION" not in df.columns:
            print(f"[Schedule] ⚠️ TEAM_ABBREVIATION missing for {player}.")
            return None, None

        team_abbr = df["TEAM_ABBREVIATION"].mode()[0]

        # --- Build lookup for teamID → abbreviation (using commonteamroster)
        all_teams = commonteamroster.CommonTeamRoster(season=datetime.now().year).get_data_frames()[1]
        id_map = dict(zip(all_teams["TEAM_ID"], all_teams["ABBREVIATION"]))

        # --- Reverse lookup: abbreviation → ID
        inv_map = {v: k for k, v in id_map.items()}
        team_id = inv_map.get(team_abbr)

        if team_id is None:
            print(f"[Schedule] ⚠️ Could not map team ID for {team_abbr}.")
            return None, None

        # --- Match today’s schedule
        for home_id, away_id in pairs:
            if home_id == team_id:
                opp_id = away_id
                side = "home"
            elif away_id == team_id:
                opp_id = home_id
                side = "away"
            else:
                continue

            opp_abbr = id_map.get(opp_id, None)
            if opp_abbr:
                symbol = "vs" if side == "home" else "@"
                print(f"[Schedule] ✅ {team_abbr} plays today {symbol} {opp_abbr}")
                return opp_abbr, team_abbr

        print(f"[Schedule] ℹ️ {team_abbr} not in today's scoreboard — checking next opponent...")
        return get_upcoming_opponent_abbr(player, settings)

    except Exception as e:
        print(f"[Schedule] ⚠️ Could not resolve opponent for {player}: {e}")
        return get_upcoming_opponent_abbr(player, settings)
def get_dvp_multiplier(opp_abbr, position, stat, player_df=None):
    """
    Returns a Defense-vs-Position (DvP) multiplier based on opponent strength
    for a given stat and position. Scales around 1.00 (neutral),
    lower = tougher matchup, higher = easier matchup.
    """
    try:
        if not opp_abbr or not position:
            print("[DvP] ⚠️ Missing opponent or position, using neutral 1.0")
            return 1.0

        # Simplified baseline table (can be replaced by live API or CSV later)
        dvp_baseline = {
            "PTS": {"PG": 1.03, "SG": 1.00, "SF": 0.98, "PF": 0.96, "C": 0.95},
            "REB": {"PG": 0.97, "SG": 0.99, "SF": 1.02, "PF": 1.05, "C": 1.06},
            "AST": {"PG": 1.05, "SG": 1.02, "SF": 0.98, "PF": 0.95, "C": 0.90},
            "PRA": {"PG": 1.03, "SG": 1.00, "SF": 0.99, "PF": 0.97, "C": 0.96},
            "RA":  {"PG": 1.01, "SG": 1.00, "SF": 1.00, "PF": 0.99, "C": 1.01},
            "FG3M": {"PG": 1.04, "SG": 1.03, "SF": 1.00, "PF": 0.95, "C": 0.85},
        }

        base = dvp_baseline.get(stat.upper(), {}).get(position.upper(), 1.0)

        # Add random small context noise if needed (keeps projections dynamic)
        if player_df is not None and "MATCHUP" in player_df.columns:
            games = player_df[player_df["MATCHUP"].str.contains(opp_abbr, na=False)]
            if len(games) > 0:
                stat_col = STAT_MAP.get(stat, stat)
                opp_mean = games[stat_col].astype(float).mean()
                player_mean = player_df[stat_col].astype(float).mean()
                if player_mean > 0:
                    perf_ratio = opp_mean / player_mean
                    adj = np.clip(perf_ratio, 0.90, 1.10)
                    base *= adj
                    print(f"[DvP] {opp_abbr} vs {position} on {stat}: ratio={perf_ratio:.3f} → adj={adj:.3f}")

        base = round(base, 3)
        print(f"[DvP] {opp_abbr} vs {position} on {stat}: → mult={base:.3f}")
        return base

    except Exception as e:
        print(f"[DvP] ⚠️ Error calculating DvP multiplier: {e}")
        return 1.0
def get_usage_factor(player, stat, settings):
    """
    Estimate player usage factor (scaled 0.95–1.05) based on shot volume,
    touches, or minutes trend. Falls back to minutes-based trend when
    advanced stats are missing.
    """
    try:
        path = os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv")
        if not os.path.exists(path):
            return 1.0

        df = pd.read_csv(path)
        stat = STAT_MAP.get(stat, stat)

        # --- Prefer usage-like columns if available
        possible_cols = ["FGA", "USG%", "TOUCHES", "POSS"]
        usage_col = next((c for c in possible_cols if c in df.columns), None)

        if usage_col:
            last10 = df.tail(10)[usage_col].astype(float)
            season_avg = df[usage_col].astype(float).mean()
        elif "MIN" in df.columns:
            last10 = df.tail(10)["MIN"].astype(float)
            season_avg = df["MIN"].astype(float).mean()
        else:
            return 1.0

        if season_avg <= 0:
            return 1.0

        recent_avg = last10.mean()
        ratio = recent_avg / season_avg
        usage_mult = np.clip(ratio, 0.95, 1.05)

        print(f"[Usage] {player}: recent_avg={recent_avg:.1f}, season_avg={season_avg:.1f} → mult={usage_mult:.3f}")
        return usage_mult
    except Exception as e:
        print(f"[Usage] ⚠️ Could not compute usage for {player}: {e}")
        return 1.0
def get_recent_form(df, stat_col, line):
    """
    Compute recent-form probability (L10 games) of going over the line.
    Returns the fraction of games over in the last 10, scaled slightly
    toward 0.5 to prevent overreaction to small streaks.
    """
    try:
        if stat_col not in df.columns:
            print(f"[L10] ⚠️ Missing stat column: {stat_col}")
            return 0.5

        last10 = df.tail(10)[stat_col].astype(float)
        if len(last10) == 0:
            return 0.5

        p_l10 = np.mean(last10 > line)
        smoothed = 0.5 + (p_l10 - 0.5) * 0.8  # regression to mean (dampens streaks)
        print(f"[L10] {stat_col} → {p_l10:.2f} raw → {smoothed:.2f} smoothed")
        return smoothed
    except Exception as e:
        print(f"[L10] ⚠️ Error computing recent form: {e}")
        return 0.5
def get_homeaway_adjustment(player, stat, line, settings):
    """
    Return a probability adjustment based on home/away splits.
    Increases confidence slightly for home players and decreases for away.
    """
    try:
        df = pd.read_csv(os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv"))
        if "MATCHUP" not in df.columns:
            return 1.0

        # Detect home (no '@') vs away ('@')
        home_games = df[~df["MATCHUP"].str.contains("@", na=False)]
        away_games = df[df["MATCHUP"].str.contains("@", na=False)]

        if len(home_games) < 5 or len(away_games) < 5:
            return 1.0  # too few samples

        stat = STAT_MAP.get(stat, stat)
        if stat not in df.columns:
            return 1.0

        home_mean = home_games[stat].mean()
        away_mean = away_games[stat].mean()
        overall_mean = df[stat].mean()

        if overall_mean == 0:
            return 1.0

        adj = (home_mean - away_mean) / overall_mean
        adj = np.clip(1 + adj * 0.2, 0.95, 1.05)
        print(f"[Home/Away] {player}: home_mean={home_mean:.2f}, away_mean={away_mean:.2f}, adj={adj:.3f}")
        return adj
    except Exception as e:
        print(f"[Home/Away] ⚠️ Could not compute adjustment for {player}: {e}")
        return 1.0

# ===============================
# MAIN (stable interactive version)
# ===============================
def main():
    settings = load_settings()

    print("\n🧠 PropPulse+ Model v2025.3 — Player Prop EV Analyzer", flush=True)
    print("=====================================================\n", flush=True)

    while True:
        try:
            mode = input("Mode [1] Single  [2] Batch (manual)  [3] CSV file  [Q] Quit: ").strip().lower()
        except EOFError:
            print("\n(Input stream closed) Exiting.")
            return

        if mode in ("q", "quit", "exit"):
            print("Goodbye!")
            return

        # ---------- Single Prop ----------
        if mode in ("1", ""):
            try:
                player = input("Player name: ").strip()
                stat = input("Stat (PTS/REB/AST/REB+AST/PRA/FG3M): ").strip().upper()
                line = float(input("Line (e.g., 20.5): ").strip())
                odds = int(input("Odds (e.g., -110): ").strip())
                debug_mode = input("Enable debug mode? (y/n, default=n): ").strip().lower() == "y"

                result = analyze_single_prop(player, stat, line, odds, settings, debug_mode)
                if not result:
                    print("❌ Analysis returned no result.")
                else:
                    print("\n" + "=" * 60)
                    print(f"📊 {player} | {stat} Line {line}")
                    print(f"Games Analyzed: {result['n_games']}")
                    print(f"Model Projection: {result['projection']:.2f} {stat}")
                    print(f"Model Prob:  {result['p_model'] * 100:.1f}%")
                    print(f"Book Prob:   {result['p_book'] * 100:.1f}%")

                    ev_cents = result['ev'] * 100
                    edge_pct = (result['p_model'] - result['p_book']) * 100
                    conf = result.get('confidence', 0.0)
                    grade = result.get('grade', 'N/A')

                    print(f"EV: {ev_cents:+.1f}¢ per $1 | Edge: {edge_pct:+.2f}% | Confidence: {conf:.2f}")
                    print(f"Grade: {grade}")
                    print("🟢 Over Value" if result['projection'] > line else "🔴 Under Value")
                    print(f"Context → {result.get('position','N/A')} vs {result.get('opponent','N/A')} "
                          f"| DvP x{result.get('dvp_mult',1.0):.3f} | Injury: {result.get('injury','N/A')}")
                    print("=" * 60 + "\n")

            except ValueError as ve:
                print(f"❌ Invalid input: {ve}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif mode == "2":
            print("Batch mode not implemented yet.")
        elif mode == "3":
            print("CSV mode not implemented yet.")
        else:
            print("Please choose 1, 2, 3, or Q.")
            input("\nPress Enter to continue...")




# ===============================
# Program entry point
# ===============================
if __name__ == "__main__":
    print("🧠 PropPulse+ Model v2025.3 — Player Prop EV Analyzer")
    print("=" * 60, flush=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Exiting...")
    except Exception as e:
        import traceback
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        # On Windows console, keep the window open if launched via double-click
        try:
            if sys.stdin.isatty():
                pass  # running in an open console
            else:
                input("\nPress Enter to close...")
        except Exception:
            pass
