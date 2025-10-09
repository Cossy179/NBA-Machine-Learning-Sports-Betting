from sbrscrape import Scoreboard
from datetime import datetime, timedelta
import pytz


class SbrOddsProvider:
    """ Abbreviations dictionary for team location which are sometimes saved with abbrev instead of full name.
    Moneyline options name require always full name
    Returns:
        string: Full location name
    """

    def __init__(self, sportsbook="fanduel", hours_ahead=20):
        """
        Initialize SbrOddsProvider
        
        Args:
            sportsbook: Sportsbook to get odds from
            hours_ahead: Number of hours ahead to fetch games (default: 20 for next 20 hours in UK timezone)
        """
        self.sportsbook = sportsbook
        self.games = []
        
        # Get UK timezone
        uk_tz = pytz.timezone('Europe/London')
        now_uk = datetime.now(uk_tz)
        cutoff_time_uk = now_uk + timedelta(hours=hours_ahead)
        
        # Fetch games for today and next day to cover the time range
        for day_offset in range(2):  # Today and tomorrow
            try:
                if day_offset == 0:
                    # Today's games
                    sb = Scoreboard(sport="NBA")
                else:
                    # Tomorrow's games
                    target_date = (now_uk + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                    sb = Scoreboard(sport="NBA", date=target_date)
                
                if hasattr(sb, 'games') and sb.games:
                    # Filter games to only include those within the UK time window
                    for game in sb.games:
                        try:
                            # First try to parse from the raw date field (most accurate)
                            game_datetime_uk = self._parse_game_datetime_from_raw(game, day_offset, now_uk)
                            
                            if game_datetime_uk:
                                # Use the parsed datetime for filtering
                                if now_uk <= game_datetime_uk <= cutoff_time_uk:
                                    self.games.append(game)
                            else:
                                # Fallback to parsing event_time string
                                game_time_str = game.get('event_time', '')
                                if game_time_str and game_time_str != 'TBD':
                                    game_time_uk = self._parse_game_time_uk(game_time_str, day_offset, now_uk)
                                    if game_time_uk and now_uk <= game_time_uk <= cutoff_time_uk:
                                        self.games.append(game)
                                else:
                                    # If no time available, include games from today only
                                    if day_offset == 0:
                                        self.games.append(game)
                        except Exception as e:
                            # If we can't parse the time, include games from today only
                            if day_offset == 0:
                                self.games.append(game)
            except Exception as e:
                print(f"Warning: Could not fetch games for day {day_offset}: {e}")
                continue

    def _parse_game_time_uk(self, time_str, day_offset, now_uk):
        """Parse game time string and return datetime object in UK timezone"""
        try:
            # Handle various time formats that SBR might return
            uk_tz = pytz.timezone('Europe/London')
            target_date = now_uk + timedelta(days=day_offset)
            
            # Try different time formats
            time_formats = [
                '%H:%M',      # 19:30
                '%I:%M %p',   # 7:30 PM
                '%I:%M%p',    # 7:30PM
                '%H:%M:%S',   # 19:30:00
            ]
            
            for fmt in time_formats:
                try:
                    # Parse just the time part
                    time_part = datetime.strptime(time_str.strip(), fmt).time()
                    # Combine with the target date and localize to UK timezone
                    game_datetime_naive = datetime.combine(target_date.date(), time_part)
                    game_datetime_uk = uk_tz.localize(game_datetime_naive)
                    return game_datetime_uk
                except ValueError:
                    continue
            
            # If no format works, return None
            return None
            
        except Exception:
            return None

    def _parse_game_datetime_from_raw(self, game_data, day_offset, now_uk):
        """Parse game datetime from raw game data (date field) and convert to UK timezone"""
        try:
            # Get the date field from the raw game data
            date_str = game_data.get('date', '')
            if not date_str:
                return None
            
            # Parse the UTC datetime
            game_datetime_utc = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Convert to UK timezone
            uk_tz = pytz.timezone('Europe/London')
            game_datetime_uk = game_datetime_utc.astimezone(uk_tz)
            
            return game_datetime_uk
            
        except Exception as e:
            print(f"Error parsing game datetime: {e}")
            return None

    def get_odds(self):
        """Function returning odds from Sbr server's json content

        Returns:
            dictionary: [home_team_name + ':' + away_team_name: { home_team: money_line_odds, away_team: money_line_odds }, under_over_odds: val]
        """
        dict_res = {}
        for game in self.games:
            # Get team names
            home_team_name = game['home_team'].replace("Los Angeles Clippers", "LA Clippers")
            away_team_name = game['away_team'].replace("Los Angeles Clippers", "LA Clippers")

            money_line_home_value = money_line_away_value = totals_value = None

            # Get money line bet values
            if self.sportsbook in game['home_ml']:
                money_line_home_value = game['home_ml'][self.sportsbook]
            if self.sportsbook in game['away_ml']:
                money_line_away_value = game['away_ml'][self.sportsbook]

            # Get totals bet value
            if self.sportsbook in game['total']:
                totals_value = game['total'][self.sportsbook]

            dict_res[home_team_name + ':' + away_team_name] = {
                'under_over_odds': totals_value,
                home_team_name: {'money_line_odds': money_line_home_value},
                away_team_name: {'money_line_odds': money_line_away_value}
            }
        return dict_res
