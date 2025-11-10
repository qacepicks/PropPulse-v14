"""
NBA Player Props Stats Fetcher - FIXED VERSION
Key fixes:
- Stricter opponent matching
- Better date filtering
- Clearer error messages
- No fallback to wrong games
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

class NBAStatsFetcher:
    def __init__(self, file_path):
        """Initialize with your CSV or Excel file path"""
        self.file_path = file_path
        self.file_type = 'csv' if file_path.endswith('.csv') else 'excel'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def fetch_player_game_stats(self, player_name, opponent=None, game_date=None, max_date=None, expected_date=None):
        """
        Fetch player stats from NBA API
        opponent: 3-letter team abbreviation (e.g., 'OKC', 'MEM')
        game_date: Optional date string 'YYYY-MM-DD' or 'MM/DD/YYYY'
        max_date: Maximum date to consider (defaults to today)
        expected_date: Date we expect the game to have occurred (for validation)
        Returns: dict with PTS, REB, AST, FG3M
        """
        try:
            from nba_api.stats.static import players
            from nba_api.stats.endpoints import playergamelog
            
            # Find player
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                print(f"  ⚠️ Player not found: {player_name}")
                return None
            
            player_id = player_dict[0]['id']
            
            # Get current season game log
            season = '2024-25'
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                print(f"  ⚠️ No games found for {player_name}")
                return None
            
            # Set maximum date
            if max_date is None:
                max_date = datetime.now()
            
            # Parse game dates
            def parse_game_date(date_str):
                """Parse NBA game date format like 'NOV 09, 2024'"""
                try:
                    return datetime.strptime(date_str.upper(), '%b %d, %Y'.upper())
                except:
                    try:
                        return datetime.strptime(date_str, '%b %d, %Y')
                    except:
                        return datetime(2099, 1, 1)
            
            df['parsed_date'] = df['GAME_DATE'].apply(parse_game_date)
            
            # Filter for only COMPLETED games (on or before max_date)
            df = df[df['parsed_date'] <= max_date]
            df = df[df['PTS'].notna() & (df['PTS'] != '')]
            
            if df.empty:
                print(f"  ⚠️ No completed games found for {player_name} on or before {max_date.strftime('%b %d, %Y')}")
                return None
            
            print(f"  📅 Checking {len(df)} completed games (latest: {df.iloc[0]['GAME_DATE']})")
            
            game_found = None
            
            # Strategy 1: Match by opponent abbreviation
            if opponent:
                opponent = opponent.upper().strip()
                
                # FIXED: More precise opponent matching
                # Look for the opponent code in the matchup string
                matching_games = df[df['MATCHUP'].str.contains(f' {opponent}$', case=False, na=False, regex=True)]
                
                if matching_games.empty:
                    # Also try matching with "vs." or "@" prefix
                    matching_games = df[
                        df['MATCHUP'].str.contains(f'vs\. {opponent}', case=False, na=False, regex=True) |
                        df['MATCHUP'].str.contains(f'@ {opponent}', case=False, na=False, regex=True)
                    ]
                
                if not matching_games.empty:
                    # FIXED: Check if the matched game is close to max_date
                    most_recent_match = matching_games.iloc[0]
                    game_date_parsed = most_recent_match['parsed_date']
                    days_old = (max_date - game_date_parsed).days
                    
                    # Only use this game if it's within last 10 days
                    if days_old <= 10:
                        game_found = most_recent_match
                        print(f"  ✅ Found game vs {opponent}: {game_found['GAME_DATE']} ({days_old} days ago)")
                    else:
                        print(f"  ⚠️ Found game vs {opponent} but it's {days_old} days old ({most_recent_match['GAME_DATE']})")
                        print(f"  ⚠️ NO RECENT GAME FOUND - Skipping this entry")
                        return None
                else:
                    print(f"  ⚠️ No game found vs {opponent}")
                    print(f"  ℹ️ Available opponents: {df['MATCHUP'].head(5).tolist()}")
                    print(f"  ⚠️ NO MATCHING GAME - Skipping this entry")
                    return None
            
            # Strategy 2: Match by specific date
            elif game_date:
                if '/' in game_date:
                    date_obj = datetime.strptime(game_date, '%m/%d/%Y')
                else:
                    date_obj = datetime.strptime(game_date, '%Y-%m-%d')
                target_date = date_obj.strftime('%b %d, %Y').upper()
                
                df['GAME_DATE_UPPER'] = df['GAME_DATE'].str.upper()
                matching_games = df[df['GAME_DATE_UPPER'] == target_date]
                
                if not matching_games.empty:
                    game_found = matching_games.iloc[0]
                    print(f"  ✅ Found game on {target_date}")
                else:
                    print(f"  ⚠️ No game found on {target_date}")
                    return None
            
            # Strategy 3: Most recent completed game (only if no opponent specified)
            else:
                game_found = df.iloc[0]
                print(f"  📅 Using most recent game: {game_found['GAME_DATE']}")
            
            if game_found is None:
                return None
            
            return {
                'PTS': int(game_found['PTS']) if game_found['PTS'] else 0,
                'REB': int(game_found['REB']) if game_found['REB'] else 0,
                'AST': int(game_found['AST']) if game_found['AST'] else 0,
                'FG3M': int(game_found['FG3M']) if game_found['FG3M'] else 0,
                'game_date': game_found['GAME_DATE'],
                'matchup': game_found['MATCHUP']
            }
            
        except Exception as e:
            print(f"  ❌ Error fetching stats for {player_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_excel_with_results(self, target_date=None, use_opponent_matching=True, max_date=None):
        """
        Read file (CSV or Excel), fetch stats, and add results columns
        """
        # Read file
        if self.file_type == 'csv':
            df = pd.read_csv(self.file_path)
            print(f"📁 Reading CSV file: {self.file_path}")
        else:
            df = pd.read_excel(self.file_path)
            print(f"📁 Reading Excel file: {self.file_path}")
        
        print(f"🎯 Cutoff Date: {max_date.strftime('%B %d, %Y') if max_date else 'Today'}")
        if use_opponent_matching and 'opponent' in df.columns:
            print(f"🎯 Mode: Matching by OPPONENT column")
        elif target_date:
            print(f"🎯 Mode: Specific date ({target_date})")
        else:
            print(f"🎯 Mode: Most recent games")
        print()
        
        # Add new columns
        new_columns = ['Actual_Stat', 'Hit_Miss', 'Result', 'Game_Date', 'Matchup']
        for col in new_columns:
            if col not in df.columns:
                df[col] = ''
        
        print(f"Processing {len(df)} players...")
        print("-" * 60)
        
        # Track stats
        processed = 0
        skipped = 0
        
        # Process each row
        for idx, row in df.iterrows():
            player_name = row['Player']
            stat_type = row['Stat']
            line = row['Line']
            
            # Get opponent if available
            opponent = None
            if use_opponent_matching and 'opponent' in df.columns:
                opponent = str(row['opponent']).strip() if pd.notna(row['opponent']) else None
            
            print(f"\n{idx+1}. {player_name} - {stat_type} (Line: {line})")
            if opponent:
                print(f"   Looking for game vs: {opponent}")
            
            # Fetch stats with expected date validation
            stats = self.fetch_player_game_stats(
                player_name, 
                opponent=opponent, 
                game_date=target_date, 
                max_date=max_date,
                expected_date=max_date  # Pass expected date for validation
            )
            
            if stats:
                # Check for date mismatch flag
                if stats.get('date_mismatch'):
                    print(f"  ⚠️ WRONG DATE - Expected: {stats['expected_date']}, Found: {stats['found_date']}")
                    print(f"  🏀 Matchup: {stats['matchup']}")
                    # Flag as wrong date instead of pending
                    df.at[idx, 'Actual_Stat'] = 'WRONG DATE'
                    df.at[idx, 'Hit_Miss'] = 'WRONG_DATE'
                    df.at[idx, 'Result'] = '⚠️'
                    df.at[idx, 'Game_Date'] = stats['found_date']
                    df.at[idx, 'Matchup'] = stats['matchup']
                    skipped += 1
                    time.sleep(0.6)
                    continue
                
                actual_value = stats.get(stat_type, None)
                
                if actual_value is not None:
                    df.at[idx, 'Actual_Stat'] = actual_value
                    df.at[idx, 'Game_Date'] = stats.get('game_date', '')
                    df.at[idx, 'Matchup'] = stats.get('matchup', '')
                    
                    # Determine hit/miss
                    if actual_value > line:
                        df.at[idx, 'Hit_Miss'] = 'HIT'
                        df.at[idx, 'Result'] = '✓'
                        result_emoji = '✅'
                    else:
                        df.at[idx, 'Hit_Miss'] = 'MISS'
                        df.at[idx, 'Result'] = '✗'
                        result_emoji = '❌'
                    
                    print(f"  📊 {stat_type}: {actual_value} (Line: {line}) {result_emoji}")
                    print(f"  🏀 {stats.get('matchup', '')} on {stats.get('game_date', '')}")
                    processed += 1
                else:
                    print(f"  ⚠️ Stat type '{stat_type}' not found in game data")
                    # Flag as pending
                    df.at[idx, 'Actual_Stat'] = 'N/A'
                    df.at[idx, 'Hit_Miss'] = 'PENDING'
                    df.at[idx, 'Result'] = '⏳'
                    df.at[idx, 'Game_Date'] = 'Not Available'
                    df.at[idx, 'Matchup'] = f"vs {opponent}" if opponent else 'N/A'
                    skipped += 1
            else:
                print(f"  ⚠️ PENDING - Game not found (may not be played yet)")
                # Flag as pending instead of leaving blank
                df.at[idx, 'Actual_Stat'] = 'N/A'
                df.at[idx, 'Hit_Miss'] = 'PENDING'
                df.at[idx, 'Result'] = '⏳'
                df.at[idx, 'Game_Date'] = 'Game Not Found'
                df.at[idx, 'Matchup'] = f"vs {opponent}" if opponent else 'N/A'
                skipped += 1
            
            time.sleep(0.6)
        
        # Save updated file
        base_name = os.path.splitext(self.file_path)[0]
        if self.file_type == 'csv':
            output_file = f"{base_name}_updated.csv"
            df.to_csv(output_file, index=False)
        else:
            output_file = f"{base_name}_updated.xlsx"
            df.to_excel(output_file, index=False)
            
        print("\n" + "=" * 60)
        print(f"✅ Results saved to: {output_file}")
        print(f"\n📊 Summary:")
        print(f"   ✅ Completed: {processed}")
        print(f"   ⏳ Pending: {(df['Hit_Miss'] == 'PENDING').sum()}")
        print(f"   ⚠️ Wrong Date: {(df['Hit_Miss'] == 'WRONG_DATE').sum()}")
        
        # Calculate hit rate (ONLY from completed games)
        total_bets = ((df['Hit_Miss'] == 'HIT') | (df['Hit_Miss'] == 'MISS')).sum()
        hits = (df['Hit_Miss'] == 'HIT').sum()
        pending = (df['Hit_Miss'] == 'PENDING').sum()
        wrong_date = (df['Hit_Miss'] == 'WRONG_DATE').sum()
        
        if total_bets > 0:
            hit_rate = hits / total_bets * 100
            excluded = pending + wrong_date
            print(f"   Hit Rate: {hits}/{total_bets} = {hit_rate:.1f}% (excluding {excluded} unverified)")
        elif pending > 0 or wrong_date > 0:
            print(f"   Hit Rate: N/A - {pending} pending, {wrong_date} wrong date")
        
        return df


def main():
    """Main function"""
    # ⬇️ UPDATE THESE SETTINGS ⬇️
    file_path = "nba_props.csv"  # Your file
    
    # IMPORTANT: Set this to TOMORROW if you're checking today's games
    # The API updates the next day after games complete
    cutoff_date_str = "11/09/2024"  # ⚠️ Set to day AFTER games are played
    
    # Parse cutoff date
    if '/' in cutoff_date_str:
        cutoff_date = datetime.strptime(cutoff_date_str, '%m/%d/%Y')
    else:
        cutoff_date = datetime.strptime(cutoff_date_str, '%Y-%m-%d')
    
    print("=" * 60)
    print("NBA PLAYER PROPS STATS FETCHER - FIXED VERSION")
    print("=" * 60)
    print(f"⏰ Cutoff Date: {cutoff_date.strftime('%B %d, %Y')}")
    print(f"   (Only games on or before this date)")
    print()
    print("💡 IMPORTANT: If checking yesterday's games, run this script")
    print("   the next day to ensure stats are updated in the API!")
    print()
    
    fetcher = NBAStatsFetcher(file_path)
    
    try:
        results_df = fetcher.update_excel_with_results(
            target_date=None, 
            use_opponent_matching=True, 
            max_date=cutoff_date
        )
        print("\n✨ Done! Check your updated file.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file: {file_path}")
        print("Please update the 'file_path' variable.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()