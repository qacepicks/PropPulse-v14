#!/usr/bin/env python3
"""
PropPulse+ v2025.6 — Professional NBA Prop Analyzer
Advanced L20-weighted projection + DvP + auto position detection
Clean, professional CLI interface with enhanced visuals
"""

import os
import sys
import time
import math
import json
import platform
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import pytz

from dvp_updater import load_dvp_data
from nba_stats_fetcher import fetch_player_logs
from nba_api.stats.static import teams as nba_teams, players as nba_players
from nba_api.stats.endpoints import commonplayerinfo

# ===============================
# 🎨 TERMINAL COLORS & FORMATTING
# ===============================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'
    
    # Custom gradients
    ELITE = '\033[38;2;16;185;129m'  # Green
    SOLID = '\033[38;2;59;130;246m'  # Blue
    NEUTRAL = '\033[38;2;156;163;175m'  # Gray
    FADE = '\033[38;2;239;68;68m'  # Red

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header(text: str, char: str = "═"):
    """Print formatted header"""
    width = 70
    print(f"\n{Colors.BLUE}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BLUE}{char * width}{Colors.END}\n")

def print_section(text: str):
    """Print section divider"""
    print(f"\n{Colors.GRAY}{'─' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.GRAY}{'─' * 70}{Colors.END}")

def print_metric(label: str, value: Any, color: str = Colors.CYAN):
    """Print formatted metric"""
    print(f"  {Colors.GRAY}•{Colors.END} {Colors.BOLD}{label}:{Colors.END} {color}{value}{Colors.END}")

def print_success(msg: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_warning(msg: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_error(msg: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

# ===============================
# 🔧 CONFIGURATION
# ===============================

DATA_PATH_DEFAULT = "data/"

STAT_MAP = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",
    "PRA": ["PTS", "REB", "AST"],
    "REB+AST": ["REB", "AST"],
    "PTS+REB": ["PTS", "REB"],
    "PTS+AST": ["PTS", "AST"],
    "RA": ["REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "P+R": ["PTS", "REB"],
    "P+A": ["PTS", "AST"],
}

def load_tuned_config(path="proppulse_config_latest.json") -> Dict:
    """Load tuned configuration parameters"""
    if not os.path.exists(path):
        print_info(f"Tuned config not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print_success(f"Loaded tuned parameters from {path}")
    return cfg

CONFIG_TUNED = load_tuned_config()

# Default calibration constants
try:
    from calibration import (
        TEMP_Z, BALANCE_BIAS, CLIP_MIN, CLIP_MAX,
        MULT_CENTER, MULT_MAX_DEV, EMP_PRIOR_K, W_EMP_MAX,
        INCLUDE_LAST_SEASON, SHRINK_TO_LEAGUE,
    )
except Exception:
    TEMP_Z = 1.05
    BALANCE_BIAS = 0.10
    CLIP_MIN, CLIP_MAX = 0.08, 0.92
    MULT_CENTER, MULT_MAX_DEV = 1.0, 0.25
    EMP_PRIOR_K, W_EMP_MAX = 20, 0.30
    INCLUDE_LAST_SEASON = True
    SHRINK_TO_LEAGUE = 0.10

# ===============================
# ⚙️ SETTINGS MANAGEMENT
# ===============================

def load_settings() -> Dict:
    """Load or create settings.json"""
    default = {
        "default_sportsbook": "Fliff",
        "default_region": "us",
        "data_path": DATA_PATH_DEFAULT,
        "injury_api_key": "YOUR_SPORTSDATAIO_KEY",
        "balldontlie_api_key": "free",
        "cache_hours": 24,
    }
    path = "settings.json"
    
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f, indent=4)
        print_success("Created new settings.json with defaults")
        return default

    with open(path, "r") as f:
        settings = json.load(f)

    for k, v in default.items():
        if k not in settings:
            settings[k] = v

    os.makedirs(settings["data_path"], exist_ok=True)
    return settings

# ===============================
# 📡 BALLDONTLIE API WRAPPER
# ===============================

def get_bdl(endpoint: str, params: Optional[Dict] = None, 
            settings: Optional[Dict] = None, timeout: int = 10) -> Optional[Dict]:
    """Fetch data from BallDontLie API"""
    base_url = "https://api.balldontlie.io/v1"
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
        if r.status_code == 200:
            data = r.json()
            return data
        if r.status_code in (401, 403, 404, 429):
            print_warning(f"BDL API returned {r.status_code}")
            return None
        return None
    except Exception as e:
        print_error(f"BDL request failed: {e}")
        return None

# ===============================
# 📊 DVP DATA LOADING
# ===============================

try:
    dvp_data = load_dvp_data()
except Exception as e:
    print_warning(f"Failed to load DvP: {e}")
    dvp_data = {}

# ===============================
# 🔍 POSITION DETECTION
# ===============================

TEAM_MAP = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL", "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA", "MILWAUKEE BUCKS": "MIL", "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP", "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL", "PHILADELPHIA 76ERS": "PHI", "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR", "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR", "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS"
}

def normalize_position(pos: str) -> str:
    """Normalize position abbreviations"""
    pos = (pos or "").upper().strip()
    mapping = {
        "G": "SG", "G-F": "SF", "F": "SF", "F-G": "SF",
        "F-C": "PF", "C-F": "C",
    }
    return mapping.get(pos, pos)

def infer_position_from_stats(df_logs: pd.DataFrame) -> str:
    """Infer player position from statistical profile"""
    def avg(col):
        if col not in df_logs.columns:
            return 0.0
        return pd.to_numeric(df_logs[col], errors="coerce").fillna(0).mean()

    a_pts = avg("PTS")
    a_reb = avg("REB")
    a_ast = avg("AST")
    a_fg3 = avg("FG3M")

    if a_ast >= 6.5:
        return "PG"
    if a_reb >= 9 and a_fg3 < 1.0:
        return "C"
    if a_reb >= 7.5 and a_ast < 5.5:
        return "PF"
    if 4.5 <= a_reb <= 8 and 3 <= a_ast <= 6 and a_pts >= 12:
        return "SF"
    if a_ast >= 3 and a_reb < 5.5 and (a_fg3 >= 1.5 or a_pts >= 15):
        return "SG"
    if a_reb >= 7:
        return "PF"
    if a_reb >= 5:
        return "SF"
    return "SG"

def get_player_position_auto(player_name: str, df_logs: Optional[pd.DataFrame], 
                            settings: Optional[Dict] = None) -> str:
    """Auto-detect player position using BDL API and statistical inference"""
    try:
        last_name = player_name.split()[-1]
        data = get_bdl("/players", {"search": last_name}, settings)
        if data and data.get("data"):
            for p in data["data"]:
                full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                if full.lower() == player_name.lower():
                    pos = normalize_position(p.get("position", ""))
                    if pos:
                        return pos
            first = data["data"][0]
            pos = normalize_position(first.get("position", ""))
            if pos:
                return pos
    except Exception as e:
        print_warning(f"BDL position lookup failed: {e}")

    if df_logs is not None and len(df_logs) > 0:
        pos = infer_position_from_stats(df_logs)
        return pos

    return "SF"

# ===============================
# 🕒 OPPONENT DETECTION
# ===============================

def get_live_opponent_from_schedule(player_name: str, settings: Optional[Dict] = None) -> str:
    """Get opponent from today's NBA schedule"""
    try:
        pinfo = next(
            (p for p in nba_players.get_players() if p["full_name"].lower() == player_name.lower()),
            None,
        )
        if not pinfo:
            return "N/A"

        info = commonplayerinfo.CommonPlayerInfo(player_id=pinfo["id"]).get_data_frames()[0]
        team_abbr = info.loc[0, "TEAM_ABBREVIATION"]

        url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        r = requests.get(url, timeout=8)
        games = r.json().get("scoreboard", {}).get("games", [])

        for g in games:
            home = g["homeTeam"]["teamTricode"]
            away = g["awayTeam"]["teamTricode"]
            if team_abbr == home:
                return away
            if team_abbr == away:
                return home
        return "N/A"
    except Exception:
        return "N/A"

def get_upcoming_opponent_abbr(player_name: str, settings: Optional[Dict] = None) -> tuple:
    """Get next opponent from upcoming schedule"""
    try:
        last_name = player_name.split()[-1]
        player_data = get_bdl("/players", {"search": last_name}, settings)
        if not player_data or not player_data.get("data"):
            return None, None

        player_match = None
        for p in player_data["data"]:
            full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if full.lower() == player_name.lower():
                player_match = p
                break
        if not player_match:
            player_match = player_data["data"][0]

        player_id = player_match.get("id")
        bdl_team_abbr = player_match.get("team", {}).get("abbreviation", "UNK")
        team_abbr = bdl_team_abbr

        from dateutil import parser
        today = datetime.now().date()
        future = today + timedelta(days=7)
        games_data = get_bdl(
            "/games",
            {"player_ids[]": player_id, "start_date": str(today), "end_date": str(future)},
            settings,
        )
        
        if not games_data or not games_data.get("data"):
            return None, team_abbr

        games = sorted(games_data["data"], key=lambda x: parser.parse(x["date"]).date())

        next_game = None
        for g in games:
            g_date = parser.parse(g["date"]).date()
            if g_date >= today:
                next_game = g
                break
        
        if not next_game:
            return None, team_abbr

        home = next_game.get("home_team", {})
        away = next_game.get("visitor_team", {})
        player_team_id = player_match.get("team", {}).get("id")

        if home.get("id") == player_team_id:
            opp_abbr = away.get("abbreviation")
        else:
            opp_abbr = home.get("abbreviation")

        return opp_abbr, team_abbr
    except Exception:
        return None, None

# ===============================
# 📊 DvP MULTIPLIER
# ===============================

def get_dvp_multiplier(opponent_abbr: Optional[str], position: Optional[str], stat_key: str) -> float:
    """Calculate DvP-based matchup multiplier"""
    try:
        if not opponent_abbr or not isinstance(opponent_abbr, str):
            return 1.0
        if not position or not isinstance(position, str):
            return 1.0
        if not stat_key:
            return 1.0

        opp = opponent_abbr.upper()
        pos = position.upper()
        sk = stat_key.replace(" ", "").upper()

        alias_map = {
            "RA": "REB+AST",
            "PR": "PTS+REB",
            "PA": "PTS+AST",
        }
        sk = alias_map.get(sk, sk)

        if opp not in dvp_data or pos not in dvp_data[opp]:
            return 1.0

        pos_dict = dvp_data[opp][pos]

        if sk in ("REB+AST",):
            ranks = [pos_dict.get("REB"), pos_dict.get("AST")]
        elif sk in ("PRA",):
            ranks = [pos_dict.get("PTS"), pos_dict.get("REB"), pos_dict.get("AST")]
        elif sk in ("PTS+REB",):
            ranks = [pos_dict.get("PTS"), pos_dict.get("REB")]
        elif sk in ("PTS+AST",):
            ranks = [pos_dict.get("PTS"), pos_dict.get("AST")]
        else:
            ranks = [pos_dict.get(sk)]

        ranks = [r for r in ranks if r is not None]
        if not ranks:
            return 1.0

        avg_rank = sum(ranks) / len(ranks)
        mult = 1.1 - (avg_rank - 1) / 300.0

        if mult > 1.05:
            mult = 1 + (mult - 1) * 0.65
        elif mult < 0.95:
            mult = 1 - (1 - mult) * 0.85

        return round(mult, 3)
    except Exception:
        return 1.0

# ===============================
# 📈 CORE STATISTICS HELPERS
# ===============================

def _cap(x: float, lo: float, hi: float) -> float:
    """Clamp value between bounds"""
    return max(lo, min(hi, x))

def normalize_multiplier(raw: float, center: float = MULT_CENTER, 
                        max_dev: float = MULT_MAX_DEV) -> float:
    """Normalize multiplier to prevent extreme adjustments"""
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    log_m = math.log(raw / center)
    log_m *= 0.5
    m = math.exp(log_m)
    return _cap(m, 1.0 - max_dev, 1.0 + max_dev)

def adjusted_mean(mu_base: float, multipliers: Dict[str, float],
                 league_mu: Optional[float] = None,
                 shrink_to_league: float = SHRINK_TO_LEAGUE) -> float:
    """Calculate adjusted projection mean"""
    prod = 1.0
    for v in (multipliers or {}).values():
        prod *= normalize_multiplier(float(v))
    mu = mu_base * prod
    if league_mu is not None and math.isfinite(league_mu):
        mu = (1.0 - shrink_to_league) * mu + shrink_to_league * league_mu
    return mu

def prob_over_from_normal(mu: float, sigma: float, line: float,
                         temp_z: float = TEMP_Z) -> float:
    """Calculate probability of exceeding line from normal distribution"""
    eps = 1e-9
    sigma_eff = max(sigma, eps)
    z = (line - mu) / sigma_eff
    z /= max(1.0, temp_z)
    return float(1.0 - norm.cdf(z))

def smooth_empirical_prob(vals: np.ndarray, line: float, k: int = EMP_PRIOR_K) -> float:
    """Calculate smoothed empirical probability"""
    n = int(vals.size)
    hits = int((vals > line).sum())
    alpha = hits + 0.5 * k
    beta = (n - hits) + 0.5 * k
    return alpha / (alpha + beta)

def finalize_prob(p_raw: float, balance_bias: float = BALANCE_BIAS,
                 clip_min: float = CLIP_MIN, clip_max: float = CLIP_MAX) -> float:
    """Finalize probability with bias correction and clipping"""
    p = (1.0 - balance_bias) * p_raw + balance_bias * 0.5
    return _cap(p, clip_min, clip_max)

def calibrated_prob_over(mu_base: float, sigma_base: float, line: float,
                        multipliers: Dict[str, float], recent_vals: np.ndarray,
                        league_mu: Optional[float] = None) -> float:
    """Calculate calibrated probability of going over"""
    mu = adjusted_mean(mu_base, multipliers, league_mu=league_mu)
    p_model = prob_over_from_normal(mu, sigma_base, line, temp_z=TEMP_Z)
    p_emp = smooth_empirical_prob(np.array(recent_vals, dtype=float), line)
    n = int(len(recent_vals))
    w_emp = min(W_EMP_MAX, n / (n + EMP_PRIOR_K))
    p_blend = (1 - w_emp) * p_model + w_emp * p_emp
    return finalize_prob(p_blend)

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability"""
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

def net_payout(odds: int) -> float:
    """Calculate net payout from American odds"""
    return 100 / abs(odds) if odds < 0 else odds / 100

def ev_sportsbook(p: float, odds: int) -> float:
    """Calculate expected value"""
    return p * net_payout(odds) - (1 - p)

# ===============================
# 🧾 PLAYER DATA FETCHER
# ===============================

def fetch_player_data(player: str, settings: Optional[Dict] = None) -> Optional[pd.DataFrame]:
    """Fetch player game logs"""
    settings = settings or {}
    data_path = settings.get("data_path", DATA_PATH_DEFAULT)
    os.makedirs(data_path, exist_ok=True)
    safe = player.replace(" ", "_")
    path = os.path.join(data_path, f"{safe}.csv")

    try:
        df = fetch_player_logs(player, save_dir=data_path)
        if df is not None and len(df) > 0:
            df.to_csv(path, index=False)
            return df
    except Exception as e:
        print_warning(f"Primary data source failed: {e}")

    # Fallback to BallDontLie
    try:
        last_name = player.split()[-1]
        player_data = get_bdl("/players", {"search": last_name}, settings)
        if not player_data or not player_data.get("data"):
            return None
        
        cand = None
        for p in player_data["data"]:
            full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if full.lower() == player.lower():
                cand = p
                break
        if not cand:
            cand = player_data["data"][0]
        
        player_id = cand.get("id")
        season = datetime.now().year - (1 if datetime.now().month >= 10 else 0)
        stats = get_bdl(
            "/stats",
            {"player_ids[]": player_id, "seasons[]": season, "per_page": 100},
            settings,
        )
        
        if not stats or not stats.get("data"):
            return None
        
        rows = []
        for g in stats["data"]:
            mins_raw = g.get("min", "0")
            if isinstance(mins_raw, str) and ":" in mins_raw:
                try:
                    m, s = mins_raw.split(":")
                    mins = int(m) + int(s) / 60.0
                except Exception:
                    mins = 0.0
            else:
                try:
                    mins = float(mins_raw or 0)
                except Exception:
                    mins = 0.0
            
            rows.append({
                "DATE": g.get("game", {}).get("date"),
                "PTS": g.get("pts", 0),
                "REB": g.get("reb", 0),
                "AST": g.get("ast", 0),
                "FG3M": g.get("fg3m", 0),
                "MIN": mins,
            })
        
        df = pd.DataFrame(rows)
        df = df[df["MIN"] > 0]
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        print_error(f"Failed to fetch player data: {e}")
        return None

# ===============================
# 🎯 GRADING LOGIC
# ===============================

def assign_grade(ev_cents: float, confidence: float, model_prob: float, stat: str) -> str:
    """Assign prop grade based on EV and confidence"""
    stat_weights = {
        "PTS": {"ev": 7.0, "conf": 0.58},
        "REB": {"ev": 5.0, "conf": 0.54},
        "AST": {"ev": 5.0, "conf": 0.53},
        "PRA": {"ev": 6.0, "conf": 0.55},
        "REB+AST": {"ev": 5.5, "conf": 0.54},
        "FG3M": {"ev": 3.5, "conf": 0.50},
    }
    cfg = stat_weights.get(stat.upper(), {"ev": 5.0, "conf": 0.50})

    if ev_cents >= cfg["ev"] * 1.8 and confidence >= (cfg["conf"] + 0.07) and model_prob >= 0.59:
        return f"{Colors.ELITE}🔥 ELITE{Colors.END}"
    if ev_cents >= cfg["ev"] and confidence >= cfg["conf"] and model_prob >= 0.54:
        return f"{Colors.SOLID}💎 SOLID{Colors.END}"
    if ev_cents >= 1.0 and model_prob >= 0.50 and confidence >= 0.40:
        return f"{Colors.NEUTRAL}⚖️ NEUTRAL{Colors.END}"
    return f"{Colors.FADE}🚫 FADE{Colors.END}"

# ===============================
# 🧠 MAIN ANALYZER
# ===============================

def analyze_single_prop(player: str, stat: str, line: float, odds: int,
                       settings: Dict, debug_mode: bool = False) -> Optional[Dict]:
    """Analyze a single player prop"""
    
    data_path = settings.get("data_path", DATA_PATH_DEFAULT)
    os.makedirs(data_path, exist_ok=True)
    safe = player.replace(" ", "_")
    path = os.path.join(data_path, f"{safe}.csv")

    # Load or refresh logs
    need_refresh = not os.path.exists(path) or \
                   (time.time() - os.path.getmtime(path)) / 3600.0 > settings.get("cache_hours", 24)
    
    if need_refresh:
        print_info(f"Refreshing logs for {player}...")
        df = fetch_player_data(player, settings)
        if df is None or len(df) == 0:
            print_error(f"No logs available for {player}")
            return None
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass
    else:
        df = pd.read_csv(path)

    # Clean minutes
    if "MIN" in df.columns:
        def parse_min(v):
            if isinstance(v, str) and ":" in v:
                try:
                    m, s = v.split(":")
                    return int(m) + int(s) / 60.0
                except Exception:
                    return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0
        df["MIN"] = df["MIN"].apply(parse_min)
        df = df[df["MIN"] > 0]
    
    if df.empty:
        print_error(f"All games filtered for {player}")
        return None

    # Stat handling
    stat_norm = stat.replace(" ", "").upper()
    stat_map_entry = STAT_MAP.get(stat_norm)
    
    if not stat_map_entry:
        print_error(f"Stat '{stat}' not recognized")
        return None

    if isinstance(stat_map_entry, list):
        df = df.copy()
        df["COMPOSITE"] = df[stat_map_entry].sum(axis=1)
        vals = pd.to_numeric(df["COMPOSITE"], errors="coerce").dropna()
        stat_col_for_recent = "COMPOSITE"
    else:
        col = stat_map_entry
        if col not in df.columns:
            print_error(f"Stat column '{col}' not found for {player}")
            return None
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        stat_col_for_recent = col

    n_games = len(vals)
    season_mean = vals.mean()
    std = vals.std(ddof=0) if n_games > 1 else 1.0
    mean_l10 = vals.tail(10).mean() if n_games >= 10 else season_mean
    mean_l20 = vals.tail(20).mean() if n_games >= 20 else season_mean
    recent_trend = (mean_l10 + mean_l20) / 2
    trend_weight = 0.15
    base_projection = (1 - trend_weight) * season_mean + trend_weight * recent_trend

    # Minutes adjustment
    if "MIN" in df.columns:
        season_mins = df["MIN"].mean()
        l10_mins = df["MIN"].tail(10).mean() if n_games >= 10 else season_mins
        if season_mins > 0:
            ratio = l10_mins / season_mins
            if abs(ratio - 1.0) > 0.10:
                base_projection *= ratio

    # Probability calculations
    p_emp = float(np.mean(vals > line)) if n_games > 0 else 0.5
    p_norm = 1 - norm.cdf(line, season_mean, std if std > 0 else 1.0)
    p_base = 0.6 * p_norm + 0.4 * p_emp

    # Opponent + DvP
    opp = None
    try:
        live_opp = get_live_opponent_from_schedule(player, settings)
        if isinstance(live_opp, str) and live_opp not in ("N/A", "", None):
            opp = live_opp
    except Exception:
        pass

    if not opp:
        try:
            opp_res, team_abbr = get_upcoming_opponent_abbr(player, settings)
            if opp_res:
                opp = opp_res
        except Exception:
            opp = None

    if isinstance(opp, tuple):
        opp = opp[0]
    if opp is None or not isinstance(opp, str) or not opp.strip():
        opp = "UNKNOWN"
    opp = opp.upper().strip()

    pos = get_player_position_auto(player, df_logs=df, settings=settings)
    
    try:
        dvp_mult = get_dvp_multiplier(opp, pos, stat_norm)
    except Exception:
        dvp_mult = 1.0

    # Context multipliers
    pace_mult = 1.0
    p_ha = 1.0  # Home/away adjustment placeholder
    p_usage = 1.0  # Usage adjustment placeholder
    
    multipliers = {"dvp": dvp_mult, "pace": pace_mult, "ha": p_ha, "usage": p_usage}
    
    p_model_core = calibrated_prob_over(
        mu_base=season_mean,
        sigma_base=std,
        line=line,
        multipliers=multipliers,
        recent_vals=vals.values,
        league_mu=None,
    )

    maturity = min(1.0, n_games / 40.0)
    base_conf = 1 - (std / season_mean) if season_mean > 0 else 0.5
    confidence = max(0.1, base_conf * maturity)
    confidence = max(0.1, min(0.99, confidence))

    # Context adjustments
    team_total = 112.0  # Default league average
    rest_days = 1
    team_mult = 1.0
    rest_mult = 1.0
    dvp_mult_adj = max(0.80, min(1.25, dvp_mult))

    context_mult = dvp_mult_adj * team_mult * rest_mult
    projection = base_projection * context_mult
    
    # Line sanity check
    line_trust = 0.25
    if line > 0:
        deviation_ratio = projection / line
        if deviation_ratio < 0.65 or deviation_ratio > 1.35:
            projection = (1 - line_trust) * projection + line_trust * line

    proj_stat = float(projection)

    deviation_pct = abs(proj_stat - line) / line * 100 if line > 0 else 0.0
    if deviation_pct > 25:
        confidence *= 0.7
        confidence = max(0.1, min(0.99, confidence))

    p_model = max(0.05, min(p_model_core * (0.5 + 0.5 * confidence), 0.95))
    p_book = american_to_prob(odds)
    ev_raw = ev_sportsbook(p_model, odds)
    ev = ev_raw * (0.5 + 0.5 * confidence)
    ev_cents = ev * 100.0
    edge_pct = (p_model - p_book) * 100.0

    grade = assign_grade(ev_cents, confidence, p_model, stat_norm)

    direction = "OVER" if proj_stat > line else "UNDER"
    result_symbol = "⚠️" if abs(proj_stat - line) < 0.5 else ("✓" if direction == "OVER" else "✗")

    return {
        "player": player,
        "stat": stat,
        "line": float(line),
        "odds": int(odds),
        "projection": round(proj_stat, 2),
        "p_model": float(p_model),
        "p_book": float(p_book),
        "ev": float(ev),
        "n_games": int(n_games),
        "confidence": float(confidence),
        "grade": str(grade),
        "opponent": opp,
        "position": pos,
        "dvp_mult": round(float(dvp_mult), 3),
        "direction": direction,
        "result": result_symbol,
        "EV¢": round(ev_cents, 2),
        "edge": round(edge_pct, 2),
    }

# ===============================
# 📦 BATCH ANALYSIS
# ===============================

def batch_analyze_props(props_list: List[Dict], settings: Dict) -> List[Dict]:
    """Analyze multiple props in batch"""
    results = []
    total = len(props_list)
    
    print_section(f"Batch Analysis: {total} Props")
    
    for i, prop in enumerate(props_list, start=1):
        print(f"\n{Colors.CYAN}[{i}/{total}]{Colors.END} {prop['player']} — {prop['stat']} {prop['line']}")
        try:
            res = analyze_single_prop(
                prop["player"], prop["stat"], prop["line"], 
                prop["odds"], settings, debug_mode=False
            )
            if res:
                results.append(res)
                # Show quick result
                ev_color = Colors.GREEN if res['ev'] > 0 else Colors.RED
                print(f"  {Colors.GRAY}→{Colors.END} Projection: {Colors.BOLD}{res['projection']:.1f}{Colors.END} | "
                      f"EV: {ev_color}{res['EV¢']:+.1f}¢{Colors.END} | Grade: {res['grade']}")
        except Exception as e:
            print_error(f"Failed: {e}")
    
    return results

# ===============================
# 🎨 DISPLAY HELPERS
# ===============================

def display_result(result: Dict):
    """Display analysis result with professional formatting"""
    
    print_header(f"🏀 {result['player']} | {result['stat']} Analysis")
    
    # Key Metrics
    print_section("📊 Key Metrics")
    print_metric("Model Projection", f"{result['projection']:.2f} {result['stat']}", Colors.CYAN)
    print_metric("Line", f"{result['line']}", Colors.GRAY)
    print_metric("Games Analyzed", f"{result['n_games']}", Colors.GRAY)
    
    # Probabilities & EV
    print_section("🎯 Probability Analysis")
    print_metric("Model Win Prob", f"{result['p_model']*100:.1f}%", Colors.CYAN)
    print_metric("Book Implied Prob", f"{result['p_book']*100:.1f}%", Colors.GRAY)
    print_metric("Edge", f"{result['edge']:+.2f}%", 
                Colors.GREEN if result['edge'] > 0 else Colors.RED)
    
    # Expected Value
    ev_color = Colors.GREEN if result['ev'] > 0 else Colors.RED
    ev_symbol = "+" if result['ev'] > 0 else ""
    print(f"\n  {Colors.BOLD}Expected Value:{Colors.END} {ev_color}{ev_symbol}{result['EV¢']:.2f}¢{Colors.END} per $1")
    print(f"  {Colors.BOLD}Confidence:{Colors.END} {Colors.CYAN}{result['confidence']:.2%}{Colors.END}")
    print(f"  {Colors.BOLD}Grade:{Colors.END} {result['grade']}")
    
    # Recommendation
    recommendation = f"{Colors.GREEN}BET {result['direction']}{Colors.END}" if result['ev'] > 0 else f"{Colors.RED}FADE THIS PROP{Colors.END}"
    print(f"\n  {Colors.BOLD}→ Recommendation:{Colors.END} {recommendation}")
    
    # Context
    print_section("🔍 Matchup Context")
    print_metric("Opponent", result['opponent'], Colors.YELLOW)
    print_metric("Position", result['position'], Colors.GRAY)
    print_metric("DvP Multiplier", f"{result['dvp_mult']:.3f}x", 
                Colors.GREEN if result['dvp_mult'] > 1 else Colors.RED if result['dvp_mult'] < 1 else Colors.GRAY)
    
    print(f"\n{Colors.GRAY}{'═' * 70}{Colors.END}\n")

def display_batch_summary(results: List[Dict]):
    """Display batch analysis summary"""
    
    print_header("📊 Batch Analysis Summary")
    
    total = len(results)
    positive_ev = sum(1 for r in results if r['ev'] > 0)
    elite = sum(1 for r in results if "ELITE" in r['grade'])
    solid = sum(1 for r in results if "SOLID" in r['grade'])
    
    print_metric("Total Props Analyzed", total, Colors.CYAN)
    print_metric("Positive EV Plays", f"{positive_ev} ({positive_ev/total*100:.1f}%)", Colors.GREEN)
    print_metric("Elite Grades", elite, Colors.ELITE)
    print_metric("Solid Grades", solid, Colors.SOLID)
    
    # Top 5 by EV
    sorted_results = sorted(results, key=lambda x: x['ev'], reverse=True)
    
    print_section("🔥 Top 5 Props by EV")
    for i, r in enumerate(sorted_results[:5], 1):
        ev_color = Colors.GREEN if r['ev'] > 0 else Colors.RED
        print(f"  {i}. {Colors.BOLD}{r['player']:20}{Colors.END} {r['stat']:8} {r['line']:5.1f} | "
              f"Proj: {Colors.CYAN}{r['projection']:5.1f}{Colors.END} | "
              f"EV: {ev_color}{r['EV¢']:>6.1f}¢{Colors.END} | {r['grade']}")
    
    print(f"\n{Colors.GRAY}{'═' * 70}{Colors.END}\n")

# ===============================
# 💾 EXPORT FUNCTIONS
# ===============================

def export_results(results: List[Dict], filename: str = None):
    """Export results to CSV"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"propulse_results_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print_success(f"Results exported to {filename}")

# ===============================
# 🖥️ MAIN CLI
# ===============================

def main():
    """Main CLI interface"""
    settings = load_settings()
    
    clear_screen()
    
    # ASCII Art Banner
    print(f"""{Colors.BLUE}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   {Colors.CYAN}██████╗ ██████╗  ██████╗ ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗{Colors.BLUE}║
║   {Colors.CYAN}██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██║   ██║██║     ██╔════╝{Colors.BLUE}║
║   {Colors.CYAN}██████╔╝██████╔╝██║   ██║██████╔╝██████╔╝██║   ██║██║     ███████╗{Colors.BLUE}║
║   {Colors.CYAN}██╔═══╝ ██╔══██╗██║   ██║██╔═══╝ ██╔═══╝ ██║   ██║██║     ╚════██║{Colors.BLUE}║
║   {Colors.CYAN}██║     ██║  ██║╚██████╔╝██║     ██║     ╚██████╔╝███████╗███████║{Colors.BLUE}║
║   {Colors.CYAN}╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚══════╝{Colors.BLUE}║
║                                                                   ║
║            {Colors.BOLD}Professional NBA Player Prop Analyzer v2025.6{Colors.END}{Colors.BLUE}         ║
║               {Colors.GRAY}Advanced DvP · L20 Weighted · Auto Position{Colors.BLUE}             ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.END}""")
    
    while True:
        print(f"\n{Colors.BOLD}{Colors.CYAN}═══ MAIN MENU ═══{Colors.END}\n")
        print(f"  {Colors.CYAN}[1]{Colors.END} Single Prop Analysis")
        print(f"  {Colors.CYAN}[2]{Colors.END} Batch Analysis")
        print(f"  {Colors.CYAN}[3]{Colors.END} Compare Props")
        print(f"  {Colors.GRAY}[Q]{Colors.END} Quit\n")
        
        try:
            mode = input(f"{Colors.BOLD}Select mode:{Colors.END} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.GRAY}Exiting...{Colors.END}")
            return

        # Quit
        if mode in ("q", "quit", "exit"):
            print(f"\n{Colors.GREEN}Thanks for using PropPulse+! 🏀{Colors.END}\n")
            return

        # Single Analysis
        if mode in ("1", ""):
            try:
                print(f"\n{Colors.BOLD}═══ Single Prop Analysis ═══{Colors.END}\n")
                
                player = input(f"{Colors.CYAN}Player name:{Colors.END} ").strip()
                if not player:
                    print_error("Player name required")
                    continue
                
                print(f"\n{Colors.GRAY}Stats: PTS, REB, AST, REB+AST, PRA, P+R, P+A, FG3M{Colors.END}")
                stat = input(f"{Colors.CYAN}Stat:{Colors.END} ").strip().upper()
                
                line = float(input(f"{Colors.CYAN}Line:{Colors.END} ").strip())
                odds = int(input(f"{Colors.CYAN}Odds (e.g., -110):{Colors.END} ").strip())
                
                debug_input = input(f"{Colors.CYAN}Debug mode? (y/N):{Colors.END} ").strip().lower()
                debug_mode = debug_input in ("y", "yes")

                print(f"\n{Colors.YELLOW}⏳ Analyzing {player}'s {stat} projection...{Colors.END}\n")
                
                result = analyze_single_prop(player, stat, line, odds, settings, debug_mode)
                
                if not result:
                    print_error("Analysis returned no result")
                    continue

                display_result(result)
                
                # Ask to save
                save = input(f"{Colors.CYAN}Save result to CSV? (y/N):{Colors.END} ").strip().lower()
                if save in ("y", "yes"):
                    export_results([result])
                
            except ValueError as e:
                print_error(f"Invalid input: {e}")
            except Exception as e:
                print_error(f"Analysis failed: {e}")
            
            input(f"\n{Colors.GRAY}Press Enter to continue...{Colors.END}")
            clear_screen()
            continue

        # Batch Analysis
        if mode == "2":
            print(f"\n{Colors.BOLD}═══ Batch Analysis ═══{Colors.END}\n")
            print(f"{Colors.GRAY}Enter props one per line in format: Player STAT LINE ODDS{Colors.END}")
            print(f"{Colors.GRAY}Example: LeBron James PRA 35.5 -110{Colors.END}")
            print(f"{Colors.GRAY}Type 'done' when finished{Colors.END}\n")

            props = []
            while True:
                line_in = input(f"{Colors.CYAN}Prop:{Colors.END} ").strip()
                if line_in.lower() in ("done", "d", ""):
                    break

                try:
                    parts = line_in.split()
                    player = " ".join(parts[:-3])
                    stat = parts[-3].upper()
                    line = float(parts[-2])
                    odds = int(parts[-1])

                    props.append({
                        "player": player,
                        "stat": stat,
                        "line": line,
                        "odds": odds,
                    })
                    print_success(f"Added: {player} {stat} {line}")
                except Exception:
                    print_error("Invalid format. Use: PlayerName STAT LINE ODDS")
                    continue

            if not props:
                print_warning("No props entered")
                continue

            print(f"\n{Colors.YELLOW}⏳ Analyzing {len(props)} props...{Colors.END}\n")
            results = batch_analyze_props(props, settings)

            if not results:
                print_error("No results returned")
                continue

            display_batch_summary(results)
            
            # Ask to save
            save = input(f"{Colors.CYAN}Save results to CSV? (Y/n):{Colors.END} ").strip().lower()
            if save not in ("n", "no"):
                export_results(results)
            
            input(f"\n{Colors.GRAY}Press Enter to continue...{Colors.END}")
            clear_screen()
            continue

        # Compare Props
        if mode == "3":
            print(f"\n{Colors.BOLD}═══ Prop Comparison ═══{Colors.END}\n")
            
            try:
                print(f"{Colors.CYAN}Prop 1:{Colors.END}")
                p1 = input("  Player: ").strip()
                s1 = input("  Stat: ").strip().upper()
                l1 = float(input("  Line: ").strip())
                o1 = int(input("  Odds: ").strip())
                
                print(f"\n{Colors.CYAN}Prop 2:{Colors.END}")
                p2 = input("  Player: ").strip()
                s2 = input("  Stat: ").strip().upper()
                l2 = float(input("  Line: ").strip())
                o2 = int(input("  Odds: ").strip())
                
                print(f"\n{Colors.YELLOW}⏳ Comparing props...{Colors.END}\n")
                
                r1 = analyze_single_prop(p1, s1, l1, o1, settings)
                r2 = analyze_single_prop(p2, s2, l2, o2, settings)
                
                if not r1 or not r2:
                    print_error("Failed to analyze one or both props")
                    continue
                
                print_header("⚖️ Prop Comparison")
                
                # Side by side comparison
                print(f"\n{Colors.BOLD}{'Metric':<20} {'Prop 1':<25} {'Prop 2':<25}{Colors.END}")
                print(Colors.GRAY + "─" * 70 + Colors.END)
                
                print(f"{'Player':<20} {p1:<25} {p2:<25}")
                print(f"{'Stat':<20} {s1:<25} {s2:<25}")
                print(f"{'Line':<20} {l1:<25} {l2:<25}")
                print(f"{'Projection':<20} {Colors.CYAN}{r1['projection']:<25.2f}{Colors.END} {Colors.CYAN}{r2['projection']:<25.2f}{Colors.END}")
                
                ev1_color = Colors.GREEN if r1['ev'] > 0 else Colors.RED
                ev2_color = Colors.GREEN if r2['ev'] > 0 else Colors.RED
                print(f"{'EV (¢)':<20} {ev1_color}{r1['EV¢']:<25.2f}{Colors.END} {ev2_color}{r2['EV¢']:<25.2f}{Colors.END}")
                
                print(f"{'Confidence':<20} {r1['confidence']:<25.2%} {r2['confidence']:<25.2%}")
                print(f"{'Grade':<20} {r1['grade']:<40} {r2['grade']}")
                
                # Recommendation
                better = 1 if r1['ev'] > r2['ev'] else 2
                print(f"\n{Colors.BOLD}→ Better Value:{Colors.END} {Colors.GREEN}Prop {better}{Colors.END}")
                
                print(f"\n{Colors.GRAY}{'═' * 70}{Colors.END}\n")
                
            except Exception as e:
                print_error(f"Comparison failed: {e}")
            
            input(f"\n{Colors.GRAY}Press Enter to continue...{Colors.END}")
            clear_screen()
            continue

        print_warning("Invalid option. Please select 1, 2, 3, or Q")

# ===============================
# 🚀 ENTRY POINT
# ===============================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GRAY}Interrupted by user. Exiting...{Colors.END}\n")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(1)