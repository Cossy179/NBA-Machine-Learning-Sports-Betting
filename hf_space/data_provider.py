"""
NBA Data Provider - Fetches games and odds
"""
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import sbrscrape
try:
    from sbrscrape import Scoreboard
    SBRSCRAPE_AVAILABLE = True
except ImportError:
    SBRSCRAPE_AVAILABLE = False
    logger.warning("sbrscrape not available - using fallback data")

class NBADataProvider:
    """Provides NBA game data"""
    
    def __init__(self, sportsbook='fanduel'):
        self.sportsbook = sportsbook
    
    def get_todays_games(self) -> List[Dict]:
        """
        Get today's NBA games with odds
        
        Returns:
            List of game dictionaries with team names, times, and odds
        """
        games = []
        
        # Try to get real games from SBR
        if SBRSCRAPE_AVAILABLE:
            games = self._get_games_from_sbr()
        
        # If no games found, return empty list (off-season or off-day)
        if not games:
            logger.info("No games found for today")
            return []
        
        return games
    
    def _get_games_from_sbr(self) -> List[Dict]:
        """Fetch games from SBR scraper"""
        try:
            logger.info("Fetching games from SBR...")
            
            games = []
            
            # Get today's games
            try:
                sb = Scoreboard(sport="NBA")
                
                if hasattr(sb, 'games') and sb.games:
                    for game in sb.games:
                        try:
                            home_team = game.get('home_team', '').replace("Los Angeles Clippers", "LA Clippers")
                            away_team = game.get('away_team', '').replace("Los Angeles Clippers", "LA Clippers")
                            
                            # Get odds
                            home_odds = None
                            away_odds = None
                            spread = None
                            total = None
                            
                            if self.sportsbook in game.get('home_ml', {}):
                                home_odds = game['home_ml'][self.sportsbook]
                            
                            if self.sportsbook in game.get('away_ml', {}):
                                away_odds = game['away_ml'][self.sportsbook]
                            
                            if self.sportsbook in game.get('spread', {}):
                                spread_data = game['spread'][self.sportsbook]
                                if isinstance(spread_data, (int, float)):
                                    spread = spread_data
                                elif isinstance(spread_data, dict) and 'point' in spread_data:
                                    spread = spread_data['point']
                            
                            if self.sportsbook in game.get('total', {}):
                                total_data = game['total'][self.sportsbook]
                                if isinstance(total_data, (int, float)):
                                    total = total_data
                                elif isinstance(total_data, dict) and 'point' in total_data:
                                    total = total_data['point']
                            
                            # Get game time
                            game_time = game.get('event_time', game.get('commence_time', 'TBD'))
                            
                            games.append({
                                'home_team': home_team,
                                'away_team': away_team,
                                'game_time': game_time,
                                'home_odds': home_odds,
                                'away_odds': away_odds,
                                'spread': spread,
                                'total': total
                            })
                            
                            logger.info(f"Found game: {away_team} @ {home_team}")
                            
                        except Exception as e:
                            logger.error(f"Error processing game: {e}")
                            continue
                    
                    logger.info(f"Fetched {len(games)} games from SBR")
                else:
                    logger.info("No games found in SBR response")
                
            except Exception as e:
                logger.error(f"Error fetching from SBR: {e}")
            
            # Try tomorrow's games if today is empty
            if not games:
                try:
                    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                    sb = Scoreboard(sport="NBA", date=tomorrow)
                    
                    if hasattr(sb, 'games') and sb.games:
                        for game in sb.games[:3]:  # Limit to first 3 games
                            try:
                                home_team = game.get('home_team', '').replace("Los Angeles Clippers", "LA Clippers")
                                away_team = game.get('away_team', '').replace("Los Angeles Clippers", "LA Clippers")
                                
                                home_odds = None
                                away_odds = None
                                
                                if self.sportsbook in game.get('home_ml', {}):
                                    home_odds = game['home_ml'][self.sportsbook]
                                
                                if self.sportsbook in game.get('away_ml', {}):
                                    away_odds = game['away_ml'][self.sportsbook]
                                
                                spread = None
                                if self.sportsbook in game.get('spread', {}):
                                    spread_data = game['spread'][self.sportsbook]
                                    if isinstance(spread_data, (int, float)):
                                        spread = spread_data
                                
                                total = None
                                if self.sportsbook in game.get('total', {}):
                                    total_data = game['total'][self.sportsbook]
                                    if isinstance(total_data, (int, float)):
                                        total = total_data
                                
                                game_time = game.get('event_time', game.get('commence_time', 'TBD'))
                                
                                games.append({
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'game_time': game_time,
                                    'home_odds': home_odds,
                                    'away_odds': away_odds,
                                    'spread': spread,
                                    'total': total
                                })
                                
                                logger.info(f"Found game (tomorrow): {away_team} @ {home_team}")
                                
                            except Exception as e:
                                logger.error(f"Error processing tomorrow's game: {e}")
                                continue
                        
                        logger.info(f"Fetched {len(games)} games from tomorrow")
                
                except Exception as e:
                    logger.error(f"Error fetching tomorrow's games: {e}")
            
            return games
            
        except Exception as e:
            logger.error(f"Error in _get_games_from_sbr: {e}")
            return []
    
    def _get_mock_games(self) -> List[Dict]:
        """Return mock games for testing when no real games available"""
        return [
            {
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'game_time': '19:00',
                'home_odds': -150,
                'away_odds': +130,
                'spread': -3.5,
                'total': 225.5
            },
            {
                'home_team': 'Boston Celtics',
                'away_team': 'Milwaukee Bucks',
                'game_time': '19:30',
                'home_odds': -110,
                'away_odds': -110,
                'spread': -1.5,
                'total': 222.0
            }
        ]

