#!/usr/bin/env python3
# prop_ev.py — PropPulse+ v2025.6 Hybrid Edition
# Hybrid logs (NBA Stats → BDL) + DvP + auto position + NBA/BDL opponent + manual odds

import os, sys, time, math, json, glob, platform, subprocess
from datetime import datetime, timedelta, timezone

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
# 🔧 GLOBAL CONFIG / CONSTANTS
# ===============================

DATA_PATH_DEFAULT = "data/"

STAT_MAP = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",

    # Core combos
    "PRA": ["PTS", "REB", "AST"],
    "REB+AST": ["REB", "AST"],
    "PTS+REB": ["PTS", "REB"],
    "PTS+AST": ["PTS", "AST"],

    # Aliases — allow shorthand commonly used on books
    "RA": ["REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "P+R": ["PTS", "REB"],
    "P+A": ["PTS", "AST"],
}

# Load tuned config if present (for advanced grading)
def load_tuned_config(path="proppulse_config_latest.json"):
    if not os.path.exists(path):
        print(f"[Config] ℹ️ Tuned config not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[Config] ✅ Loaded tuned parameters from {path}")
    return cfg

CONFIG_TUNED = load_tuned_config()

# Default calibration constants (fallback if calibration.py missing)
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
# ⚙️ SETTINGS
# ===============================

def load_settings():
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
        print("[Config] Created new settings.json with defaults.")
        return default

    with open(path, "r") as f:
        settings = json.load(f)

    for k, v in default.items():
        if k not in settings:
            settings[k] = v

    os.makedirs(settings["data_path"], exist_ok=True)
    return settings

# ===============================
# 📡 BALLDONTLIE WRAPPER
# ===============================

def get_bdl(endpoint, params=None, settings=None, timeout=10):
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
        print(f"[BDL] GET {endpoint} params={params} → {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            n = len(data.get("data", []))
            print(f"[BDL] ✅ {n} records")
            return data
        if r.status_code in (401, 403, 404, 429):
            print(f"[BDL] ⚠️ HTTP {r.status_code}")
            return None
        print(f"[BDL] ⚠️ Unexpected status {r.status_code}")
        return None
    except Exception as e:
        print(f"[BDL] ❌ Request failed: {e}")
        return None

# ===============================
# 📊 DvP DATA (NBA Stats–based)
# ===============================

try:
    dvp_data = load_dvp_data()  # pulls from cache or NBA Stats
except Exception as e:
    print(f"[DVP] ⚠️ Failed to load DvP: {e}")
    dvp_data = {}

# ===============================
# 🔍 POSITION DETECTION
# ===============================

def normalize_position(pos: str) -> str:
    pos = (pos or "").upper().strip()
    mapping = {
        "G": "SG",
        "G-F": "SF",
        "F": "SF",
        "F-G": "SF",
        "F-C": "PF",
        "C-F": "C",
    }
    return mapping.get(pos, pos)

def infer_position_from_stats(df_logs: pd.DataFrame) -> str:
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

def get_player_position_auto(player_name: str, df_logs: pd.DataFrame | None, settings=None) -> str:
    try:
        last_name = player_name.split()[-1]
        print(f"[Position] 🔍 BallDontLie search for {last_name}")
        data = get_bdl("/players", {"search": last_name}, settings)
        if data and data.get("data"):
            for p in data["data"]:
                full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                print(f"[Position] Checking {full}")
                if full.lower() == player_name.lower():
                    pos = normalize_position(p.get("position", ""))
                    if pos:
                        print(f"[Position] ✅ BDL → {pos}")
                        return pos
            first = data["data"][0]
            full = f"{first.get('first_name','')} {first.get('last_name','')}".strip()
            pos = normalize_position(first.get("position", ""))
            if pos:
                print(f"[Position] ✅ BDL (first match {full}) → {pos}")
                return pos
    except Exception as e:
        print(f"[Position] ⚠️ BDL error: {e}")

    if df_logs is not None and len(df_logs) > 0:
        pos = infer_position_from_stats(df_logs)
        print(f"[Position] 🧮 Inferred → {pos}")
        return pos

    print("[Position] ⚠️ Defaulting to SF")
    return "SF"

# ===============================
# 🏥 INJURY STATUS
# ===============================

def get_injury_status(player_name: str, api_key: str | None):
    if not api_key or "YOUR_SPORTSDATAIO_KEY" in api_key:
        return None
    try:
        url = "https://api.sportsdata.io/v4/nba/scores/json/Players"
        r = requests.get(url, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=8)
        if r.status_code != 200:
            return None
        for p in r.json():
            if player_name.lower() in p.get("Name", "").lower():
                return p.get("InjuryStatus")
    except Exception:
        return None
    return None

# ===============================
# 🕒 SCHEDULE / OPPONENT DETECTION
# ===============================

def get_live_opponent_from_schedule(player_name: str, settings=None):
    """
    Returns opponent abbreviation for today's NBA game, or "N/A".
    """
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
    except Exception as e:
        print(f"[Schedule] ⚠️ live opponent failed: {e}")
        return "N/A"

def get_upcoming_opponent_abbr(player_name: str, settings=None):
    """
    Fallback: uses BallDontLie + NBA API to find the next opponent in next 7 days.
    Returns (opp_abbr, team_abbr) or (None, team_abbr).
    """
    try:
        print(f"[Fallback] 🔍 Searching upcoming opponent via BallDontLie...")
        last_name = player_name.split()[-1]
        player_data = get_bdl("/players", {"search": last_name}, settings)
        if not player_data or not player_data.get("data"):
            print(f"[Fallback] ⚠️ No BDL player for {player_name}")
            return None, None

        player_match = None
        for p in player_data["data"]:
            full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if full.lower() == player_name.lower():
                player_match = p
                break
        if not player_match:
            player_match = player_data["data"][0]
            print(f"[Fallback] Using first match: {player_match.get('first_name')} {player_match.get('last_name')}")

        player_id = player_match.get("id")
        bdl_team_abbr = player_match.get("team", {}).get("abbreviation", "UNK")
        bdl_team_id = player_match.get("team", {}).get("id")

        # Override with real NBA team if available
        try:
            nba_info = next(
                (p for p in nba_players.get_players() if p["full_name"].lower() == player_name.lower()),
                None,
            )
            if nba_info:
                info = commonplayerinfo.CommonPlayerInfo(player_id=nba_info["id"]).get_data_frames()[0]
                team_abbr = info.loc[0, "TEAM_ABBREVIATION"]
                player_team_id = int(info.loc[0, "TEAM_ID"])
                print(f"[Fallback] 🔄 Corrected team to NBA: {team_abbr}")
            else:
                team_abbr = bdl_team_abbr
                player_team_id = bdl_team_id
        except Exception as e:
            print(f"[Fallback] ⚠️ NBA correction failed: {e}")
            team_abbr = bdl_team_abbr
            player_team_id = bdl_team_id

        from dateutil import parser

        today = datetime.now().date()
        future = today + timedelta(days=7)
        games_data = get_bdl(
            "/games",
            {"player_ids[]": player_id, "start_date": str(today), "end_date": str(future)},
            settings,
        )
        if not games_data or not games_data.get("data"):
            print("[Fallback] ⚠️ No upcoming games from BDL")
            return None, team_abbr

        games = sorted(
            games_data["data"],
            key=lambda x: parser.parse(x["date"]).date(),
        )

        next_game = None
        for g in games:
            g_date = parser.parse(g["date"]).date()
            if g_date >= today:
                next_game = g
                break
        if not next_game:
            print("[Fallback] ⚠️ No future games in list")
            return None, team_abbr

        home = next_game.get("home_team", {})
        away = next_game.get("visitor_team", {})

        if home.get("id") == player_team_id:
            opp_abbr = away.get("abbreviation")
        else:
            opp_abbr = home.get("abbreviation")

        g_date_str = parser.parse(next_game["date"]).date().strftime("%Y-%m-%d")
        print(f"[Fallback] ✅ Next matchup: {player_name} vs {opp_abbr} on {g_date_str}")
        return opp_abbr, team_abbr
    except Exception as e:
        print(f"[Fallback] ❌ Error in get_upcoming_opponent_abbr: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ===============================
# 📊 DvP MULTIPLIER (Medium weight)
# ===============================

def get_dvp_multiplier(opponent_abbr: str | None, position: str | None, stat_key: str) -> float:
    """
    Returns a medium-impact DvP multiplier, clamped to ~±8%.
    """
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

        # Map aliases to core keys
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
        # Base mapping: average rank around middle → ~1.0, extreme ranks → a bit off 1
        mult = 1.06 - (avg_rank - 1) / 350.0  # slightly tighter than old 1.1/300

        # Damp extremes further for "medium" effect
        if mult > 1.05:
            mult = 1 + (mult - 1) * 0.6
        elif mult < 0.95:
            mult = 1 - (1 - mult) * 0.75

        # Final clamp: ±8% around 1.0
        mult = max(0.92, min(1.08, mult))

        return round(mult, 3)
    except Exception as e:
        print(f"[DvP] ❌ Error: {e}")
        return 1.0

# ===============================
# 📈 CORE STATS HELPERS
# ===============================

def _cap(x, lo, hi):
    return max(lo, min(hi, x))

def normalize_multiplier(raw: float,
                         center: float = MULT_CENTER,
                         max_dev: float = MULT_MAX_DEV) -> float:
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    log_m = math.log(raw / center)
    log_m *= 0.5
    m = math.exp(log_m)
    return _cap(m, 1.0 - max_dev, 1.0 + max_dev)

def adjusted_mean(mu_base: float,
                  multipliers: dict,
                  league_mu: float | None = None,
                  shrink_to_league: float = SHRINK_TO_LEAGUE) -> float:
    prod = 1.0
    for v in (multipliers or {}).values():
        prod *= normalize_multiplier(float(v))
    mu = mu_base * prod
    if league_mu is not None and math.isfinite(league_mu):
        mu = (1.0 - shrink_to_league) * mu + shrink_to_league * league_mu
    return mu

def prob_over_from_normal(mu: float, sigma: float, line: float,
                          temp_z: float = TEMP_Z) -> float:
    eps = 1e-9
    sigma_eff = max(sigma, eps)
    z = (line - mu) / sigma_eff
    z /= max(1.0, temp_z)
    return float(1.0 - norm.cdf(z))

def smooth_empirical_prob(vals: np.ndarray, line: float, k: int = EMP_PRIOR_K) -> float:
    n = int(vals.size)
    hits = int((vals > line).sum())
    alpha = hits + 0.5 * k
    beta = (n - hits) + 0.5 * k
    return alpha / (alpha + beta)

def finalize_prob(p_raw: float,
                  balance_bias: float = BALANCE_BIAS,
                  clip_min: float = CLIP_MIN,
                  clip_max: float = CLIP_MAX) -> float:
    p = (1.0 - balance_bias) * p_raw + balance_bias * 0.5
    return _cap(p, clip_min, clip_max)

def calibrated_prob_over(mu_base: float,
                         sigma_base: float,
                         line: float,
                         multipliers: dict,
                         recent_vals: np.ndarray,
                         league_mu: float | None = None) -> float:
    mu = adjusted_mean(mu_base, multipliers, league_mu=league_mu)
    p_model = prob_over_from_normal(mu, sigma_base, line, temp_z=TEMP_Z)
    p_emp = smooth_empirical_prob(np.array(recent_vals, dtype=float), line)
    n = int(len(recent_vals))
    w_emp = min(W_EMP_MAX, n / (n + EMP_PRIOR_K))
    p_blend = (1 - w_emp) * p_model + w_emp * p_emp
    return finalize_prob(p_blend)

def american_to_prob(odds: int) -> float:
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

def net_payout(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100

def ev_sportsbook(p: float, odds: int) -> float:
    return p * net_payout(odds) - (1 - p)

# ===============================
# 🧾 PLAYER LOG FETCHER (Hybrid: NBA Stats → BDL)
# ===============================

def fetch_player_data(player: str, settings=None):
    """
    Hybrid fetcher:
      1) Try nba_stats_fetcher (NBA Stats wrapper, saved to data_path)
      2) Fallback to BallDontLie season stats
    """
    settings = settings or {}
    data_path = settings.get("data_path", DATA_PATH_DEFAULT)
    os.makedirs(data_path, exist_ok=True)
    safe = player.replace(" ", "_")
    path = os.path.join(data_path, f"{safe}.csv")

    # Prefer local / nba_stats_fetcher (NBA Stats)
    try:
        df = fetch_player_logs(player, save_dir=data_path)
        if df is not None and len(df) > 0:
            df.to_csv(path, index=False)
            print(f"[Data] ✅ Saved {len(df)} games via nba_stats_fetcher → {path}")
            return df
    except Exception as e:
        print(f"[Data] ⚠️ nba_stats_fetcher failed: {e}")

    # Fallback to BallDontLie basic logs
    try:
        print(f"[Data] 🔄 Fallback BDL stats for {player}")
        last_name = player.split()[-1]
        player_data = get_bdl("/players", {"search": last_name}, settings)
        if not player_data or not player_data.get("data"):
            print(f"[BDL] ❌ No player for {player}")
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
            print(f"[BDL] ❌ No stats for {player}")
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
        print(f"[Data] ✅ Saved {len(df)} games via BDL → {path}")
        return df
    except Exception as e:
        print(f"[Data] ❌ Failed BDL logs: {e}")
        return None

# ===============================
# 📈 RECENT FORM / HOME-AWAY / USAGE
# ===============================

def get_recent_form(df: pd.DataFrame, stat_col: str, line: float) -> float:
    try:
        if stat_col not in df.columns:
            return 0.5
        last10 = pd.to_numeric(df[stat_col], errors="coerce").tail(10).dropna()
        if len(last10) == 0:
            return 0.5
        return float(np.mean(last10 > line))
    except Exception as e:
        print(f"[Recent] ⚠️ {e}")
        return 0.5

def get_homeaway_adjustment(player: str, stat: str, line: float, settings) -> float:
    try:
        path = os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv")
        if not os.path.exists(path):
            return 1.0
        df = pd.read_csv(path)
        if "MATCHUP" not in df.columns:
            return 1.0

        home_games = df[~df["MATCHUP"].fillna("").str.contains("@")]
        away_games = df[df["MATCHUP"].fillna("").str.contains("@")]
        if len(home_games) < 5 or len(away_games) < 5:
            return 1.0

        stat_norm = stat.replace(" ", "").upper()
        col = STAT_MAP.get(stat_norm, stat_norm)
        if isinstance(col, list):
            home = home_games[col].sum(axis=1)
            away = away_games[col].sum(axis=1)
            overall = df[col].sum(axis=1)
        else:
            if col not in df.columns:
                return 1.0
            home = home_games[col]
            away = away_games[col]
            overall = df[col]

        home_mean = home.mean()
        away_mean = away.mean()
        overall_mean = overall.mean()
        if overall_mean == 0:
            return 1.0
        adj = (home_mean - away_mean) / overall_mean
        adj_mult = float(np.clip(1 + adj * 0.2, 0.95, 1.05))
        print(f"[HomeAway] {player} adj={adj_mult:.3f}")
        return adj_mult
    except Exception as e:
        print(f"[HomeAway] ⚠️ {e}")
        return 1.0

def get_usage_factor(player: str, stat: str, settings) -> float:
    try:
        path = os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv")
        if not os.path.exists(path):
            return 1.0
        df = pd.read_csv(path)
        if "MIN" not in df.columns:
            return 1.0
        last10 = pd.to_numeric(df["MIN"], errors="coerce").tail(10).dropna()
        season = pd.to_numeric(df["MIN"], errors="coerce").dropna()
        if len(season) == 0:
            return 1.0
        recent = last10.mean() if len(last10) > 0 else season.mean()
        season_mean = season.mean()
        ratio = recent / season_mean if season_mean > 0 else 1.0
        mult = float(np.clip(ratio, 0.95, 1.05))
        print(f"[Usage] {player}: recent={recent:.1f}, season={season_mean:.1f} → {mult:.3f}")
        return mult
    except Exception as e:
        print(f"[Usage] ⚠️ {e}")
        return 1.0

def get_rest_days(player: str, settings) -> int:
    try:
        # Simple stub: 1 rest day (extend later if you wire in schedule history)
        return 1
    except Exception:
        return 1

def get_team_total(player: str, settings):
    """Very rough team total proxy."""
    try:
        info = next((p for p in nba_players.get_players() if p["full_name"].lower() == player.lower()), None)
        if not info:
            return 112.0
        df = commonplayerinfo.CommonPlayerInfo(player_id=info["id"]).get_data_frames()[0]
        team_abbr = df.loc[0, "TEAM_ABBREVIATION"]
        baselines = {
            "BOS": 117.5, "DEN": 115.8, "SAC": 118.2, "LAL": 116.3,
            "MIL": 120.1, "DAL": 118.0, "GSW": 117.9, "NYK": 113.0,
            "OKC": 116.5, "MIA": 110.2, "PHI": 114.8, "PHX": 115.2,
            "CHI": 111.5, "CLE": 112.8, "MIN": 113.9, "NOP": 114.1,
            "ATL": 118.6, "TOR": 112.0, "BKN": 112.4, "MEM": 109.5,
            "ORL": 111.8, "HOU": 112.3, "CHA": 108.1, "POR": 107.9,
            "UTA": 113.2, "IND": 121.3, "DET": 109.0, "WAS": 112.6,
            "SAS": 110.9, "LAC": 115.4,
        }
        base = baselines.get(team_abbr, 112.0)
        return base
    except Exception:
        return 112.0

# ===============================
# 🔬 DEBUG PROJECTION
# ===============================

def debug_projection(df: pd.DataFrame, stat: str, line: float, player_name: str):
    try:
        print("\n" + "=" * 60)
        print(f"🔍 DEBUG: {player_name} {stat} Projection Analysis")
        print("=" * 60)
        stat_norm = stat.replace(" ", "").upper()
        col = STAT_MAP.get(stat_norm, stat_norm)
        if isinstance(col, list):
            df = df.copy()
            df["COMPOSITE"] = df[col].sum(axis=1)
            col = "COMPOSITE"
        if col not in df.columns:
            print(f"[Debug] ❌ Missing stat column {col}")
            return
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            print("[Debug] ⚠️ No valid values")
            return
        mean = vals.mean()
        median = vals.median()
        std = vals.std()
        print(f"Mean={mean:.2f} Median={median:.2f} Std={std:.2f} n={len(vals)}")
        over = (vals > line).sum()
        print(f"Over {line}: {over}/{len(vals)} = {over/len(vals)*100:.1f}%")
        print("=" * 60)
    except Exception as e:
        print(f"[Debug] ⚠️ {e}")

# ===============================
# 🧠 MAIN ANALYZER
# ===============================

def analyze_single_prop(player, stat, line, odds, settings, debug_mode=False):
    # ===============================
    # 🔧 INPUT NORMALIZATION (Fix streamlit types)
    # ===============================

    # Normalize stat
    try:
        if not isinstance(stat, str):
            stat = str(stat)
        stat = stat.strip().upper()
    except Exception:
        print(f"[TypeFix] ⚠️ Failed to normalize stat '{stat}', defaulting to PTS")
        stat = "PTS"

    # Normalize line → float
    try:
        if isinstance(line, str):
            line = float(line.strip())
        else:
            line = float(line)
    except Exception:
        print(f"[TypeFix] ⚠️ Failed to convert line '{line}' to float, defaulting to 0.0")
        line = 0.0

    # Normalize odds → int
    try:
        if isinstance(odds, str):
            odds = int(odds.replace("+", "").strip())
        else:
            odds = int(odds)
    except Exception:
        print(f"[TypeFix] ⚠️ Failed to convert odds '{odds}' to int, defaulting to -110")
        odds = -110

    # ===============================
    # 📁 PATH / SAVE DIR
    # ===============================

    data_path = settings.get("data_path", DATA_PATH_DEFAULT)
    os.makedirs(data_path, exist_ok=True)
    safe = player.replace(" ", "_")
    path = os.path.join(data_path, f"{safe}.csv")

    # ===============================
    # 📥 LOAD / REFRESH LOGS
    # ===============================

    need_refresh = (
        not os.path.exists(path)
        or (time.time() - os.path.getmtime(path)) / 3600.0 > settings.get("cache_hours", 24)
    )
    if need_refresh:
        print(f"[Data] ⏳ Refreshing logs for {player}...")
        df = fetch_player_data(player, settings)
        if df is None or len(df) == 0:
            print(f"[EXIT] ❌ No logs for {player}")
            return None
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass
    else:
        df = pd.read_csv(path)
        print(f"[Data] ✅ Loaded {len(df)} games for {player}")

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
        print(f"[EXIT] ❌ All games filtered for {player}")
        return None

    # Stat handling (supports RA/PR/PA/PRA etc.)
    stat_norm = stat.replace(" ", "").upper()
    stat_map_entry = STAT_MAP.get(stat_norm)
    if not stat_map_entry:
        print(f"[Error] ❌ Stat '{stat}' not recognized (normalized '{stat_norm}')")
        return None

    if isinstance(stat_map_entry, list):
        df = df.copy()
        df["COMPOSITE"] = df[stat_map_entry].sum(axis=1)
        vals = pd.to_numeric(df["COMPOSITE"], errors="coerce").dropna()
        stat_col_for_recent = "COMPOSITE"
    else:
        col = stat_map_entry
        if col not in df.columns:
            print(f"[Error] ❌ Stat column '{col}' not found for {player}")
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
    print(f"[Projection] Season={season_mean:.2f}, Recent={recent_trend:.2f} → base={base_projection:.2f}")

    if "MIN" in df.columns:
        season_mins = df["MIN"].mean()
        l10_mins = df["MIN"].tail(10).mean() if n_games >= 10 else season_mins
        if season_mins > 0:
            ratio = l10_mins / season_mins
            if abs(ratio - 1.0) > 0.10:
                print(f"[Minutes] Season={season_mins:.1f}, L10={l10_mins:.1f} → ratio={ratio:.3f}")
                base_projection *= ratio

    p_emp = float(np.mean(vals > line)) if n_games > 0 else 0.5
    p_norm = 1 - norm.cdf(line, season_mean, std if std > 0 else 1.0)
    p_base = 0.6 * p_norm + 0.4 * p_emp

    p_ha = get_homeaway_adjustment(player, stat, line, settings)
    p_l10 = get_recent_form(df, stat_col_for_recent, line)  # currently used implicitly via vals
    p_usage = get_usage_factor(player, stat, settings)

    # Opponent + DvP
    opp = None
    team_abbr = None
    try:
        live_opp = get_live_opponent_from_schedule(player, settings)
        if isinstance(live_opp, str) and live_opp not in ("N/A", "", None):
            opp = live_opp
    except Exception as e:
        print(f"[Schedule] ⚠️ live failed: {e}")

    if not opp:
        try:
            opp_res, team_abbr = get_upcoming_opponent_abbr(player, settings)
            if opp_res:
                opp = opp_res
        except Exception as e:
            print(f"[Schedule] ⚠️ fallback failed: {e}")
            opp = None

    if isinstance(opp, tuple):
        opp = opp[0]
    if opp is None or not isinstance(opp, str) or not opp.strip():
        opp = "UNKNOWN"
    opp = opp.upper().strip()

    pos = get_player_position_auto(player, df_logs=df, settings=settings)
    try:
        dvp_mult = get_dvp_multiplier(opp, pos, stat_norm)
    except Exception as e:
        print(f"[DvP] ⚠️ {e}")
        dvp_mult = 1.0

    # Context multipliers for calibrated_prob_over
    pace_mult = 1.0
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

    team_total = get_team_total(player, settings)
    rest_days = get_rest_days(player, settings)

    team_mult = min(1.20, max(0.85, team_total / 112.0 if team_total else 1.0))
    rest_mult = {0: 0.96, 1: 1.00, 2: 1.03}.get(rest_days, 1.05)
    dvp_mult_adj = max(0.92, min(1.08, dvp_mult))  # match the DvP clamping

    context_mult = dvp_mult_adj * team_mult * rest_mult
    print(f"[Context] DvP={dvp_mult_adj:.3f} × Team={team_mult:.3f} × Rest={rest_mult:.3f} = {context_mult:.3f}")

    projection = base_projection * context_mult
    line_trust = 0.25
    if line > 0:
        deviation_ratio = projection / line
        if deviation_ratio < 0.65 or deviation_ratio > 1.35:
            print(f"[Sanity] ⚠️ Adjusting projection vs line")
            projection = (1 - line_trust) * projection + line_trust * line

    proj_stat = float(projection)
    print(f"[Final] Projection={proj_stat:.2f} (base={base_projection:.2f} × ctx={context_mult:.3f})")

    deviation_pct = abs(proj_stat - line) / line * 100 if line > 0 else 0.0
    if deviation_pct > 25:
        print(f"[⚠️ ALERT] Projection deviates {deviation_pct:.1f}% from line")
        confidence *= 0.7
        confidence = max(0.1, min(0.99, confidence))

    p_model = max(0.05, min(p_model_core * (0.5 + 0.5 * confidence), 0.95))
    p_book = american_to_prob(odds)
    ev_raw = ev_sportsbook(p_model, odds)
    ev = ev_raw * (0.5 + 0.5 * confidence)
    ev_cents = ev * 100.0
    edge_pct = (p_model - p_book) * 100.0
    ev_score = ev_cents * confidence

    # ===============================
    # 🎯 UNIFIED GRADING (CCB choice)
    # ===============================
    try:
        gap_abs = abs(proj_stat - line)
        if CONFIG_TUNED:
            # Use tuned JSON config when available
            grade = grade_prop(edge_pct, confidence, gap_abs, dvp_mult)
        else:
            # Fallback to stat-based thresholds
            grade = assign_grade(ev_cents, confidence, p_model, stat_norm)
    except Exception as e:
        print(f"[Grading] ⚠️ {e}")
        grade = "NEUTRAL"

    # grade_simple kept for compatibility but matches main grade
    grade_simple = grade

    injury = get_injury_status(player, settings.get("injury_api_key"))

    if debug_mode:
        debug_projection(df, stat=stat, line=line, player_name=player)

    direction = "Higher" if proj_stat > line else "Lower"
    result_symbol = "⚠️" if abs(proj_stat - line) < 0.5 else ("✓" if direction == "Higher" else "✗")

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
        "grade_simple": str(grade_simple),
        "opponent": opp,
        "position": pos,
        "dvp_mult": round(float(dvp_mult), 3),
        "injury": injury or "N/A",
        "direction": direction,
        "result": result_symbol,
        "EV¢": round(ev_cents, 2),
    }

# ===============================
# 🎯 GRADING LOGIC
# ===============================

def grade_prop(ev_pct, conf, gap_abs, dvp):
    """
    Tuned grading using external JSON config.
    Used when CONFIG_TUNED is available.
    """
    cfg = CONFIG_TUNED
    if not cfg:
        return "NEUTRAL"

    filters = cfg["filters"]
    grading = cfg["grading"]
    dvp_rules = cfg["dvp_rules"]

    if dvp and not np.isnan(dvp):
        if dvp < dvp_rules["penalty_threshold"]:
            conf *= dvp_rules["penalty_factor"]
        elif dvp > dvp_rules["boost_threshold"]:
            conf *= dvp_rules["boost_factor"]
        conf = max(0.0, min(1.0, conf))

    if gap_abs < filters["exclude_close_to_line_gap_abs"]:
        return "⚠️ TOO CLOSE"
    if conf < filters["exclude_low_confidence"]:
        return "LOW CONF"
    if ev_pct < filters["ev_floor_percent"]:
        return "LOW EV"

    if (
        ev_pct >= grading["elite"]["ev_min_pct"]
        and conf >= grading["elite"]["conf_min"]
        and gap_abs >= grading["elite"]["gap_min"]
    ):
        return "🔥 ELITE"

    if (
        ev_pct >= grading["solid"]["ev_min_pct"]
        and conf >= grading["solid"]["conf_min"]
        and gap_abs >= grading["solid"]["gap_min"]
    ):
        return "✅ SOLID"

    return "NEUTRAL"

def assign_grade(ev_cents: float, confidence: float, model_prob: float, stat: str) -> str:
    """
    Fallback stat-based grading when tuned JSON config isn't available.
    """
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
        return "🔥 ELITE"
    if ev_cents >= cfg["ev"] and confidence >= cfg["conf"] and model_prob >= 0.54:
        return "💎 SOLID"
    if ev_cents >= 1.0 and model_prob >= 0.50 and confidence >= 0.40:
        return "⚖️ NEUTRAL"
    return "🚫 FADE"

def get_betting_grade(ev: float, model_prob: float, confidence: float) -> str:
    if ev >= 5.0 and model_prob >= 0.55:
        return "STRONG_BET"
    if ev >= 3.0 and model_prob >= 0.52:
        return "LEAN"
    return "PASS"

# ===============================
# 📦 BATCH + CLI + STREAMLIT HOOKS
# ===============================

def batch_analyze_props(props_list, settings):
    """
    Simple list-of-dicts batch (used by CLI).
    props_list: [{"player":..., "stat":..., "line":..., "odds":...}, ...]
    """
    results = []
    for i, prop in enumerate(props_list, start=1):
        print(f"\n[{i}/{len(props_list)}] 📊 {prop['player']} — {prop['stat']} {prop['line']}")
        try:
            res = analyze_single_prop(
                prop["player"], prop["stat"], prop["line"], prop["odds"], settings, debug_mode=False
            )
            if res:
                results.append(res)
        except Exception as e:
            print(f"[Batch] ⚠️ Error on {prop['player']}: {e}")
    return results

def analyze_batch_df(df_input: pd.DataFrame, settings=None, debug_mode=False) -> pd.DataFrame:
    """
    Streamlit-friendly batch interface:
    Expects columns: Player, Stat, Line, Odds (case-insensitive).
    """
    if settings is None:
        settings = load_settings()

    col_map = {c.lower(): c for c in df_input.columns}

    def get_col(name_options):
        for name in name_options:
            if name.lower() in col_map:
                return col_map[name.lower()]
        return None

    col_player = get_col(["Player", "player"])
    col_stat   = get_col(["Stat", "stat"])
    col_line   = get_col(["Line", "line"])
    col_odds   = get_col(["Odds", "odds"])

    if not all([col_player, col_stat, col_line, col_odds]):
        print("[BatchDF] ❌ Missing required columns in input DataFrame.")
        return pd.DataFrame()

    results = []
    for idx, row in df_input.iterrows():
        try:
            player = str(row[col_player]).strip()
            stat = str(row[col_stat]).strip()
            line = row[col_line]
            odds = row[col_odds]
            if not player or not stat:
                continue
            res = analyze_single_prop(player, stat, line, odds, settings, debug_mode=debug_mode)
            if res:
                results.append(res)
        except Exception as e:
            print(f"[BatchDF] ⚠️ Row {idx} failed: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()

def analyze_batch(df_input, settings=None, debug_mode=False):
    """
    Generic batch alias:
      - If df_input is a DataFrame → uses analyze_batch_df
      - If df_input is a list-of-dicts → uses batch_analyze_props
    """
    if isinstance(df_input, pd.DataFrame):
        return analyze_batch_df(df_input, settings=settings, debug_mode=debug_mode)
    elif isinstance(df_input, list):
        settings = settings or load_settings()
        results = batch_analyze_props(df_input, settings)
        return pd.DataFrame(results) if results else pd.DataFrame()
    else:
        print("[Batch] ⚠️ Unsupported input type for analyze_batch.")
        return pd.DataFrame()

def main():
    settings = load_settings()
    print("\n🧠 PropPulse+ Model v2025.6 — Calibrated NBA Player Prop EV Analyzer")
    print("=================================================================\n")

    while True:
        try:
            mode = input("Mode [1] Single  [2] Batch  [Q] Quit: ").strip().lower()
        except EOFError:
            print("\n(Input closed) Exiting.")
            return

        # ------------------------
        # Quit
        # ------------------------
        if mode in ("q", "quit", "exit"):
            print("Goodbye!")
            return

        # ------------------------
        # SINGLE MODE
        # ------------------------
        if mode in ("1", ""):
            try:
                player = input("Player name: ").strip()
                stat = input("Stat (PTS/REB/AST/REB+AST/PRA/FG3M/RA/PR/PA): ").strip().upper()
                line = float(input("Line (e.g., 20.5): ").strip())
                odds = int(input("Odds (e.g., -110): ").strip())
                dbg_in = input("Enable debug mode? (y/n, default=y): ").strip().lower()
                debug_mode = dbg_in not in ("n", "no")

                try:
                    result = analyze_single_prop(player, stat, line, odds, settings, debug_mode)
                except Exception as e:
                    print(f"[Retry] ⚠️ First run failed ({e}); retrying once...")
                    time.sleep(1)
                    result = analyze_single_prop(player, stat, line, odds, settings, debug_mode)

                if not result:
                    print("❌ Analysis returned no result.")
                    continue

                print("\n" + "=" * 60)
                print(f"📊 {player} | {stat} Line {line}")
                print(f"Games Analyzed: {result['n_games']}")
                print(f"Model Projection: {result['projection']:.2f} {stat}")
                print(f"Model Prob:  {result['p_model']*100:.1f}%")
                print(f"Book Prob:   {result['p_book']*100:.1f}%")
                ev_cents = result['ev'] * 100
                edge_pct = (result['p_model'] - result['p_book']) * 100
                conf = result['confidence']
                grade = result['grade']
                print(f"EV: {ev_cents:+.1f}¢ per $1 | Edge: {edge_pct:+.2f}% | Confidence: {conf:.2f}")
                print(f"Grade: {grade}")
                print("🟢 Over Value" if result['projection'] > line else "🔴 Under Value")
                print(
                    f"Context → {result['position']} vs {result['opponent']} "
                    f"| DvP x{result['dvp_mult']:.3f} | Injury: {result['injury']}"
                )
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"❌ Error: {e}")
            continue

        # ------------------------
        # BATCH MODE (CLI)
        # ------------------------
        if mode == "2":
            print("\n📦 Batch Mode: enter one prop per line.\nExample:  Jalen Brunson PRA 35.5 -110")
            print("Type 'done' when finished.\n")

            props = []
            while True:
                line_in = input("Prop: ").strip()
                if line_in.lower() in ("done", "d", ""):
                    break

                try:
                    parts = line_in.split()
                    player = " ".join(parts[:-3])
                    stat = parts[-3].upper()
                    line_val = float(parts[-2])
                    odds_val = int(parts[-1])

                    props.append({
                        "player": player,
                        "stat": stat,
                        "line": line_val,
                        "odds": odds_val,
                    })
                except Exception:
                    print("⚠️ Invalid format. Use: PlayerName STAT LINE ODDS")
                    continue

            if not props:
                print("No props entered.")
                continue

            print(f"\nRunning batch on {len(props)} props...\n")
            results = batch_analyze_props(props, settings)

            if not results:
                print("❌ No results returned.")
                continue

            print("\n========== BATCH RESULTS ==========\n")
            for r in results:
                print(
                    f"{r['player']:20} {r['stat']:8} Line {r['line']:5.1f} | "
                    f"Proj {r['projection']:5.2f} | EV {r['EV¢']:>6.1f}¢ | "
                    f"Conf {r['confidence']:.2f} | Grade {r['grade']}"
                )
            print("\n===================================\n")
            continue

        print("⚠️ Invalid option.")

# ===============================
# 🚀 ENTRY POINT
# ===============================

if __name__ == "__main__":
    main()
