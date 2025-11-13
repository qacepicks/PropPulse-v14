"""
NBA Schedule & Matchup Checker - Enhanced Version
Checks both past game logs AND upcoming schedule
"""

from datetime import datetime, timedelta
import time
import requests
import json
import os


def find_past_matchup_dates(player_name, opponent, season='2024-25'):
    """
    Find all PAST games where a player faced a specific opponent.
    Uses nba_api game logs (only shows games already played).
    """
    try:
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import playergamelog
        
        # Find player
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            alt_name = player_name.replace('C.J.', 'CJ').replace('J.J.', 'JJ')
            player_dict = players.find_players_by_full_name(alt_name)
        
        if not player_dict:
            return None
        
        player_id = player_dict[0]['id']
        
        # Get full game log
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        
        if df.empty:
            return None
        
        # Parse dates
        def parse_game_date(date_str):
            try:
                return datetime.strptime(date_str.upper(), '%b %d, %Y'.upper())
            except:
                return None
        
        df['parsed_date'] = df['GAME_DATE'].apply(parse_game_date)
        df = df[df['parsed_date'].notna()]
        
        # Find games vs this opponent
        opponent_games = df[df['MATCHUP'].str.upper().str.contains(opponent.upper())]
        
        return opponent_games[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'parsed_date']].to_dict('records')
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def find_upcoming_matchup(player_name, days_ahead=7):
    """
    Find UPCOMING games for a player in the next N days.
    Uses live NBA schedule APIs.
    """
    try:
        from nba_api.stats.static import players, teams
        
        # Find player
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            alt_name = player_name.replace('C.J.', 'CJ').replace('J.J.', 'JJ')
            player_dict = players.find_players_by_full_name(alt_name)
        
        if not player_dict:
            return None
        
        # Get player's team
        from nba_api.stats.endpoints import commonplayerinfo
        player_id = player_dict[0]['id']
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        team_abbr = player_info.get_data_frames()[0]['TEAM_ABBREVIATION'].iloc[0]
        
        # Get team ID
        team_dict = next((t for t in teams.get_teams() if t['abbreviation'] == team_abbr), None)
        if not team_dict:
            return None
        
        team_id = team_dict['id']
        
        # Check next N days for games
        from nba_api.stats.endpoints import scoreboardv2
        
        upcoming_games = []
        today = datetime.now().date()
        
        for d in range(days_ahead + 1):
            check_date = (today + timedelta(days=d)).strftime("%Y-%m-%d")
            
            try:
                board = scoreboardv2.ScoreboardV2(game_date=check_date)
                game_header = board.game_header.get_data_frame()
                
                if not game_header.empty:
                    # Find games with player's team
                    team_games = game_header[
                        (game_header['HOME_TEAM_ID'] == team_id) | 
                        (game_header['VISITOR_TEAM_ID'] == team_id)
                    ]
                    
                    for _, game in team_games.iterrows():
                        if game['HOME_TEAM_ID'] == team_id:
                            opp_id = game['VISITOR_TEAM_ID']
                            location = "vs"
                        else:
                            opp_id = game['HOME_TEAM_ID']
                            location = "@"
                        
                        # Get opponent abbreviation
                        opp_dict = next((t for t in teams.get_teams() if t['id'] == opp_id), None)
                        opp_abbr = opp_dict['abbreviation'] if opp_dict else "UNK"
                        
                        upcoming_games.append({
                            'date': check_date,
                            'parsed_date': datetime.strptime(check_date, "%Y-%m-%d"),
                            'opponent': opp_abbr,
                            'location': location,
                            'matchup': f"{team_abbr} {location} {opp_abbr}"
                        })
                
                time.sleep(0.6)  # Rate limiting
                
            except Exception:
                continue
        
        return upcoming_games
        
    except Exception as e:
        print(f"Error finding upcoming games: {e}")
        return None


def check_specific_player(player_name, opponent=None):
    """
    Check both past and future matchups for a specific player.
    """
    print(f"\n{'='*70}")
    print(f"🔍 CHECKING: {player_name}")
    print(f"{'='*70}\n")
    
    # Check upcoming games
    print("📅 UPCOMING GAMES (Next 7 Days):")
    print("-" * 70)
    
    upcoming = find_upcoming_matchup(player_name, days_ahead=7)
    
    if upcoming:
        today = datetime.now()
        
        for game in upcoming:
            date = game['parsed_date']
            
            if date.date() == today.date():
                status = "⏰ TODAY"
                when = "today"
            else:
                days = (date - today).days
                status = "📅 FUTURE"
                when = f"in {days} days"
            
            matchup_display = game['matchup']
            
            # Highlight if matches opponent we're looking for
            if opponent and opponent.upper() in game['opponent'].upper():
                matchup_display = f"🎯 {matchup_display} ← TARGET MATCHUP"
            
            print(f"   {status} {game['date']} ({when})")
            print(f"      {matchup_display}")
    else:
        print("   ❌ No upcoming games found")
    
    # Check past games (if opponent specified)
    if opponent:
        print(f"\n📊 PAST GAMES vs {opponent.upper()}:")
        print("-" * 70)
        
        past_games = find_past_matchup_dates(player_name, opponent)
        
        if past_games:
            today = datetime.now()
            
            for game in past_games[-5:]:  # Show last 5 games
                date = game['parsed_date']
                days_ago = (today - date).days
                
                stats = f"{game.get('PTS', 'N/A')} PTS, {game.get('REB', 'N/A')} REB, {game.get('AST', 'N/A')} AST"
                
                print(f"   ✅ {game['GAME_DATE']} ({days_ago} days ago)")
                print(f"      {game['MATCHUP']} - {stats}")
        else:
            print(f"   ❌ No past games found vs {opponent}")
    
    print()


def check_todays_full_slate():
    """
    Show ALL games scheduled for today.
    """
    print(f"\n{'='*70}")
    print("📅 TODAY'S FULL NBA SLATE")
    print(f"{'='*70}\n")
    
    try:
        import pytz
        est = pytz.timezone("US/Eastern")
        today = datetime.now(est).strftime("%Y-%m-%d")
        
        # Try live scoreboard
        url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        games = data.get("scoreboard", {}).get("games", [])
        
        if not games:
            print("ℹ️ No games scheduled today (off-day)")
            return
        
        print(f"🏀 {len(games)} games scheduled for {today}:\n")
        
        for i, game in enumerate(games, 1):
            away = game["awayTeam"]["teamTricode"]
            home = game["homeTeam"]["teamTricode"]
            game_id = game.get("gameId", "")
            
            print(f"   {i:2d}. {away} @ {home} (ID: {game_id})")
        
        print()
        
    except Exception as e:
        print(f"❌ Could not fetch today's slate: {e}")


def quick_check_aaron_gordon():
    """
    Quick check specifically for Aaron Gordon (your current issue).
    """
    print(f"\n{'='*70}")
    print("🔍 AARON GORDON QUICK CHECK")
    print(f"{'='*70}")
    
    check_specific_player("Aaron Gordon", opponent=None)
    
    # Also show today's full slate
    check_todays_full_slate()


def check_csv_matchup_dates():
    """
    Check when the matchups in your CSV are actually scheduled.
    """
    print("=" * 70)
    print("MATCHUP DATE CHECKER - BATCH MODE")
    print("=" * 70)
    print("Checking when these games are/were scheduled...\n")
    
    # Key matchups from your CSV
    matchups = [
        ('Aaron Gordon', 'LAC'),  # Today's actual game
        ('Rui Hachimura', 'CHA'),
        ('Jeremy Sochan', 'CHI'),
        ('CJ McCollum', 'DET'),
        ('Shaedon Sharpe', 'ORL'),
        ('Darius Garland', 'MIA'),
        ('Austin Reaves', 'CHA'),
    ]
    
    for player, opp in matchups:
        check_specific_player(player, opponent=opp)
        time.sleep(1)  # Rate limiting
    
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("⏰ TODAY = Game is happening today")
    print("📅 FUTURE = Game is upcoming")
    print("✅ PAST = Game already happened")
    print()


# ============================================================
# MAIN MENU
# ============================================================

def main_menu():
    """Interactive menu for schedule checking."""
    
    while True:
        print("\n" + "="*70)
        print("NBA SCHEDULE & MATCHUP CHECKER")
        print("="*70)
        print("\n[1] Check Aaron Gordon (today's issue)")
        print("[2] Check specific player")
        print("[3] Show today's full slate")
        print("[4] Batch check CSV matchups")
        print("[Q] Quit")
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        
        elif choice == '1':
            quick_check_aaron_gordon()
        
        elif choice == '2':
            player = input("Player name: ").strip()
            opponent = input("Opponent (optional, press Enter to skip): ").strip()
            check_specific_player(player, opponent if opponent else None)
        
        elif choice == '3':
            check_todays_full_slate()
        
        elif choice == '4':
            check_csv_matchup_dates()
        
        else:
            print("Invalid choice. Try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Quick check on startup
    print("🚀 Starting NBA Schedule Checker...\n")
    
    # Uncomment one of these:
    quick_check_aaron_gordon()  # Quick Aaron Gordon check
    # main_menu()  # Full interactive menu