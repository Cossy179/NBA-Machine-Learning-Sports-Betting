"""
ESPN data provider for NBA game results and team box scores.

stats.nba.com refuses requests from many cloud/data-center networks, so this
provider rebuilds everything the models need from ESPN's public JSON API:

* game results (date, teams, scores, season, season type, overtime periods)
* team box scores (FGM/FGA, 3PM/3PA, FTM/FTA, rebounds, assists, ...)

Results are cached in Data/espn_cache/ so each game is only downloaded once.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE_DIR = os.path.join("Data", "espn_cache")
GAMES_CSV = os.path.join(CACHE_DIR, "games.csv")
BOX_CSV = os.path.join(CACHE_DIR, "team_box.csv")

# ESPN season types: 1 preseason, 2 regular season, 3 postseason, 5 play-in
COUNTED_SEASON_TYPES = {2, 3, 5}

NBA_TEAMS = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans",
    "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers",
    "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs",
    "Toronto Raptors", "Utah Jazz", "Washington Wizards",
}

# Alternate spellings used by ESPN, sportsbookreview and older datasets
TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "L.A. Clippers": "LA Clippers",
    "L.A. Lakers": "Los Angeles Lakers",
}

BOX_FIELDS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("FGM", "FGA"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("FG3M", "FG3A"),
    "freeThrowsMade-freeThrowsAttempted": ("FTM", "FTA"),
    "offensiveRebounds": "OREB",
    "defensiveRebounds": "DREB",
    "totalRebounds": "REB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "totalTurnovers": "TOV",
    "fouls": "PF",
}


def normalize_team(name):
    return TEAM_ALIASES.get(name, name)


class EspnDataProvider:
    def __init__(self, cache_dir=CACHE_DIR, session=None):
        self.cache_dir = cache_dir
        self.games_csv = os.path.join(cache_dir, "games.csv")
        self.box_csv = os.path.join(cache_dir, "team_box.csv")
        os.makedirs(cache_dir, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    # ------------------------------------------------------------------ http
    def _get_json(self, url, params=None, retries=4):
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        raise RuntimeError(f"ESPN request failed: {url} {params}")

    # ----------------------------------------------------------------- games
    def _fetch_games_range(self, start, end, workers=8):
        """Fetch all games between two dates (inclusive), one scoreboard per day."""
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for day_rows in pool.map(self._fetch_games_day, days):
                rows.extend(day_rows)
        return rows

    def _fetch_games_day(self, day):
        data = self._get_json(f"{BASE_URL}/scoreboard", params={"dates": f"{day:%Y%m%d}"})
        rows = []
        for event in data.get("events", []):
            comp = event["competitions"][0]
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            if "home" not in teams or "away" not in teams:
                continue
            home = normalize_team(teams["home"]["team"]["displayName"])
            away = normalize_team(teams["away"]["team"]["displayName"])
            if home not in NBA_TEAMS or away not in NBA_TEAMS:
                continue  # All-Star and exhibition games
            start_utc = pd.Timestamp(event["date"])
            status = event["status"]["type"]
            rows.append({
                "game_id": str(event["id"]),
                "start_utc": start_utc.isoformat(),
                "date": start_utc.tz_convert("America/New_York").strftime("%Y-%m-%d"),
                "season": int(event["season"]["year"]),
                "season_type": int(event["season"]["type"]),
                # "CC" marks the NBA Cup final, which doesn't count in season stats
                "competition_type": comp.get("type", {}).get("abbreviation", "STD"),
                "home_team": home,
                "away_team": away,
                "home_score": pd.to_numeric(teams["home"].get("score"), errors="coerce"),
                "away_score": pd.to_numeric(teams["away"].get("score"), errors="coerce"),
                "periods": event["status"].get("period", 4),
                "completed": bool(status.get("completed", False)),
            })
        return rows

    def load_games(self):
        if os.path.exists(self.games_csv):
            return pd.read_csv(self.games_csv, dtype={"game_id": str})
        return pd.DataFrame()

    def update_games(self, start, end, refetch_days=3):
        """Make sure every game between start and end is cached.

        Days before the last cached completed day are skipped; the most recent
        `refetch_days` are always re-fetched so late results are picked up.
        """
        cached = self.load_games()
        ranges = [(start, end)]
        if not cached.empty:
            first_cached = pd.Timestamp(cached["date"].min()).date()
            done = cached[cached["completed"]]
            last_done = pd.Timestamp(done["date"].max()).date() if not done.empty else first_cached
            ranges = []
            if start < first_cached:
                ranges.append((start, min(end, first_cached - timedelta(days=1))))
            tail_from = max(start, last_done - timedelta(days=refetch_days))
            if tail_from <= end:
                ranges.append((tail_from, end))
        rows = []
        for range_start, range_end in ranges:
            rows.extend(self._fetch_games_range(range_start, range_end))
        new = pd.DataFrame(rows)
        if not new.empty:
            games = pd.concat([cached, new], ignore_index=True)
            games = games.drop_duplicates("game_id", keep="last")
        else:
            games = cached
        games = games.sort_values(["date", "start_utc", "game_id"]).reset_index(drop=True)
        games.to_csv(self.games_csv, index=False)
        return games

    # ------------------------------------------------------------ box scores
    def _fetch_box(self, game_id):
        try:
            data = self._get_json(f"{BASE_URL}/summary", params={"event": game_id})
        except RuntimeError:
            return []  # retried on the next call
        rows = []
        for team in data.get("boxscore", {}).get("teams", []):
            row = {"game_id": str(game_id), "team": normalize_team(team["team"]["displayName"])}
            stats = {s["name"]: s.get("displayValue") for s in team.get("statistics", [])}
            for key, target in BOX_FIELDS.items():
                value = stats.get(key)
                if isinstance(target, tuple):
                    made, att = (value or "0-0").split("-")
                    row[target[0]], row[target[1]] = int(made), int(att)
                else:
                    row[target] = int(float(value or 0))
            rows.append(row)
        return rows

    def load_boxes(self):
        if os.path.exists(self.box_csv):
            return pd.read_csv(self.box_csv, dtype={"game_id": str})
        return pd.DataFrame()

    def ensure_boxes(self, game_ids, workers=8):
        """Download (and cache) team box scores for the given completed games."""
        boxes = self.load_boxes()
        have = set(boxes["game_id"]) if not boxes.empty else set()
        missing = [g for g in dict.fromkeys(map(str, game_ids)) if g not in have]
        if missing:
            for i in range(0, len(missing), 500):
                new_rows = []
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for rows in pool.map(self._fetch_box, missing[i:i + 500]):
                        new_rows.extend(rows)
                new = pd.DataFrame(new_rows)
                boxes = pd.concat([boxes, new], ignore_index=True) if not boxes.empty else new
                boxes.to_csv(self.box_csv, index=False)
        return boxes

    # ------------------------------------------------------------- shortcuts
    def get_history(self, start, end, with_boxes_from_season=None):
        """Games from start to end; box scores for seasons >= with_boxes_from_season."""
        games = self.update_games(start, end)
        games = games[(games["date"] >= f"{start:%Y-%m-%d}") & (games["date"] <= f"{end:%Y-%m-%d}")]
        boxes = pd.DataFrame()
        if with_boxes_from_season is not None:
            need = games[
                games["completed"]
                & (games["season"] >= with_boxes_from_season)
                & games["season_type"].isin(COUNTED_SEASON_TYPES)
            ]
            boxes = self.ensure_boxes(need["game_id"])
        return games, boxes

    def get_games_on(self, day):
        """Scheduled games for one US date (not cached)."""
        rows = self._fetch_games_range(day, day)
        return [r for r in rows if r["date"] == f"{day:%Y-%m-%d}"]
