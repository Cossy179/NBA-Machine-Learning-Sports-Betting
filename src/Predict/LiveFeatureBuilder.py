"""
Feature pipeline for the LightGBM v2 game-winner model.

The same code builds the training dataset (src/Train-Models/LightGBM_V2.py)
and the inputs for tonight's games (predict.py), so the two cannot drift apart.

For a game on day D, every feature uses only information from before D:

* Base team stats (GP, W, L, ..., PLUS_MINUS and their league ranks): the
  season-to-date per-game numbers, exactly like stats.nba.com's
  leaguedashteamstats. Up to 2021-22 they come from the official tables in
  Data/TeamData.sqlite; from 2022-23 on they are rebuilt from ESPN box scores
  (stats.nba.com blocks most cloud networks).
* Elo ratings: EnhancedFeatureEngine's Elo, replayed game by game with real
  scores and read as of D.
* Recent form, head-to-head, travel, momentum, scoring trends and rest: computed
  with EnhancedFeatureEngine itself on the games of the previous 400 days.

The old training pipeline also produced simulated betting-line, injury, travel
and coaching numbers (random draws). Those carry no information and are left out.
"""
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "Process-Data"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "DataProviders"))

from Enhanced_Features import EnhancedFeatureEngine  # noqa: E402
from EspnDataProvider import COUNTED_SEASON_TYPES, EspnDataProvider, normalize_team  # noqa: E402

TEAM_DATA_PATH = os.path.join(REPO_ROOT, "Data", "TeamData.sqlite")
HISTORY_START = date(2012, 10, 1)
# Official NBA tables are used before this date, ESPN-derived stats after it
ESPN_STATS_FROM = "2022-10-01"
HISTORY_WINDOW_DAYS = 400

BASE_STATS = [
    "GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA",
    "PF", "PFD", "PTS", "PLUS_MINUS",
]
BASE_COLUMNS = BASE_STATS + [f"{c}_RANK" for c in BASE_STATS]
# Stats where the NBA gives rank 1 to the lowest value
ASCENDING_RANK = {"L", "TOV", "BLKA", "PF"}
PCT_STATS = {"W_PCT", "FG_PCT", "FG3_PCT", "FT_PCT"}
BOX_STATS = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
             "AST", "STL", "BLK", "TOV", "PF"]
# Names in the official tables for franchises that have since been renamed
OFFICIAL_NAME_ALIASES = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "Los Angeles Clippers": "LA Clippers",
}
# Keys produced by EnhancedFeatureEngine.calculate_advanced_situational_factors
# that are random draws
SIMULATED_SITUATIONAL = {"rest_advantage", "travel_fatigue", "time_zone_advantage", "coaching_advantage"}


def season_of(day):
    """ESPN season year: the 2026-27 season is 2027."""
    day = pd.Timestamp(day)
    return day.year + 1 if day.month >= 8 else day.year


# ------------------------------------------------------------------ history
def game_history(espn_games):
    """Completed regular-season, play-in and playoff games with real scores."""
    done = espn_games[espn_games["completed"] & espn_games["season_type"].isin(COUNTED_SEASON_TYPES)]
    games = pd.DataFrame({
        "game_id": done["game_id"].astype(str),
        "date": pd.to_datetime(done["date"]),
        "season": done["season"].astype(int),
        "season_type": done["season_type"].astype(int),
        "home_team": done["home_team"],
        "away_team": done["away_team"],
        "home_score": done["home_score"].astype(float),
        "away_score": done["away_score"].astype(float),
    })
    games["home_win"] = games["home_score"] > games["away_score"]
    games["total_points"] = games["home_score"] + games["away_score"]
    return games.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)


class PointInTimeElo:
    """EnhancedFeatureEngine.calculate_advanced_elo_ratings, replayed game by game
    so ratings can be read as of any date (only games before that date count)."""

    TYPES = ["overall", "home", "away", "offense", "defense", "recent"]

    def __init__(self, games, k_factor=20):
        self.k = k_factor
        self.snapshots = {}
        ratings = defaultdict(lambda: {t: 1500.0 for t in self.TYPES})
        last_date = None
        for row in games.sort_values("date", kind="mergesort").itertuples(index=False):
            if row.date != last_date:
                self.snapshots[row.date] = {t: dict(r) for t, r in ratings.items()}
                last_date = row.date
            home, away = ratings[row.home_team], ratings[row.away_team]
            actual = 1 if row.home_win else 0
            adjusted_k = k_factor * min(2.0, abs(row.home_score - row.away_score) / 10.0 + 1.0)
            for rt in ["overall", "home", "offense", "defense", "recent"]:
                expected = 1 / (1 + 10 ** ((away[rt] - home[rt]) / 400))
                home[rt] += adjusted_k * (actual - expected)
                away[rt] += adjusted_k * ((1 - actual) - (1 - expected))
            away_expected = 1 / (1 + 10 ** ((home["home"] - away["away"]) / 400))
            away["away"] += k_factor * ((1 - actual) - away_expected)
        self.final = {t: dict(r) for t, r in ratings.items()}
        self.dates = np.array(sorted(self.snapshots), dtype="datetime64[ns]")

    def as_of(self, day):
        """Ratings from every game played before `day`."""
        idx = np.searchsorted(self.dates, np.datetime64(pd.Timestamp(day)), side="left")
        if idx >= len(self.dates):
            return self.final
        return self.snapshots[pd.Timestamp(self.dates[idx])]


# --------------------------------------------------------------- team stats
def team_game_rows(espn_games, boxes):
    """One row per team per completed counted game, with its and its opponent's box."""
    done = espn_games[espn_games["completed"] & espn_games["season_type"].isin(COUNTED_SEASON_TYPES)]
    parts = []
    for side, opp in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame({
            "game_id": done["game_id"].astype(str),
            "date": done["date"],
            "season": done["season"],
            "season_type": done["season_type"],
            "competition_type": done.get("competition_type", "STD"),
            "team": done[f"{side}_team"],
            "opponent": done[f"{opp}_team"],
            "PTS": done[f"{side}_score"].astype(float),
            "OPP_PTS": done[f"{opp}_score"].astype(float),
            "MIN": 48.0 + 5.0 * (done["periods"].clip(lower=4) - 4),
        }))
    tg = pd.concat(parts, ignore_index=True)
    tg["WIN"] = (tg["PTS"] > tg["OPP_PTS"]).astype(int)
    tg["PLUS_MINUS"] = tg["PTS"] - tg["OPP_PTS"]
    boxes = boxes.copy()
    boxes["game_id"] = boxes["game_id"].astype(str)
    boxes.loc[boxes["FGA"] == 0, BOX_STATS] = np.nan  # some older ESPN box scores are empty
    tg = tg.merge(boxes.rename(columns={"team": "box_team"}), left_on=["game_id", "team"],
                  right_on=["game_id", "box_team"], how="left").drop(columns="box_team")
    opp_box = boxes[["game_id", "team", "BLK", "PF"]].rename(
        columns={"team": "opponent", "BLK": "BLKA", "PF": "PFD"})
    return tg.merge(opp_box, on=["game_id", "opponent"], how="left")


def season_team_stats(team_games, season, before_day):
    """NBA-style season-to-date per-game team stats and ranks before `before_day`.

    Mirrors stats.nba.com leaguedashteamstats (regular season, PerMode=PerGame):
    per-game averages, shooting percentages from season totals, league ranks
    computed on unrounded values. The NBA Cup final does not count."""
    tg = team_games[
        (team_games["season"] == season)
        & (team_games["season_type"] == 2)
        & (team_games["competition_type"] != "CC")
        & (team_games["date"] < f"{pd.Timestamp(before_day):%Y-%m-%d}")
    ]
    if tg.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    g = tg.groupby("team")
    stats = pd.DataFrame({
        "GP": g.size(),
        "W": g["WIN"].sum(),
        "MIN": g["MIN"].mean(),
        "PTS": g["PTS"].mean(),
        "PLUS_MINUS": g["PLUS_MINUS"].mean(),
    })
    stats["L"] = stats["GP"] - stats["W"]
    stats["W_PCT"] = stats["W"] / stats["GP"]
    for col in BOX_STATS + ["BLKA", "PFD"]:
        stats[col] = g[col].mean()  # games without a box score are skipped
    totals = g[["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]].sum(min_count=1)
    stats["FG_PCT"] = totals["FGM"] / totals["FGA"]
    stats["FG3_PCT"] = totals["FG3M"] / totals["FG3A"]
    stats["FT_PCT"] = totals["FTM"] / totals["FTA"]
    for col in BASE_STATS:
        stats[f"{col}_RANK"] = stats[col].rank(ascending=col in ASCENDING_RANK, method="min")
    for col in BASE_STATS:
        if col in PCT_STATS:
            stats[col] = stats[col].round(3)
        elif col not in ("GP", "W", "L"):
            stats[col] = stats[col].round(1)
    return stats[BASE_COLUMNS].astype(float)


class TeamStatsSource:
    """Season-to-date team stats for any day, official tables first, then ESPN."""

    def __init__(self, team_games):
        self.team_games = team_games
        self.cache = {}
        self.official_tables = set()
        if os.path.exists(TEAM_DATA_PATH):
            with sqlite3.connect(TEAM_DATA_PATH) as con:
                self.official_tables = {
                    r[0] for r in con.execute("select name from sqlite_master where type='table'")}

    def _official(self, key):
        with sqlite3.connect(TEAM_DATA_PATH) as con:
            df = pd.read_sql_query(f'select * from "{key}"', con)
        df["TEAM_NAME"] = df["TEAM_NAME"].replace(OFFICIAL_NAME_ALIASES)
        return df.set_index("TEAM_NAME")[BASE_COLUMNS].astype(float)

    def on(self, day):
        key = f"{pd.Timestamp(day):%Y-%m-%d}"
        if key not in self.cache:
            if key < ESPN_STATS_FROM and key in self.official_tables:
                self.cache[key] = self._official(key)
            else:
                self.cache[key] = season_team_stats(self.team_games, season_of(day), day)
        return self.cache[key]


# ----------------------------------------------------------------- features
def game_features(home, away, day, history, elo_ratings, stats, season_type=2):
    """All v2 features for one game on `day`.

    `history` holds games before `day`; `elo_ratings` and `stats` are as of `day`."""
    day = pd.Timestamp(day)
    recent = history[history["date"] >= day - pd.Timedelta(days=HISTORY_WINDOW_DAYS)]
    engine = EnhancedFeatureEngine()
    engine.team_elo_ratings = {t: r["overall"] for t, r in elo_ratings.items()}
    f = {}
    for rt in PointInTimeElo.TYPES:
        h = elo_ratings.get(home, {}).get(rt, 1500.0)
        a = elo_ratings.get(away, {}).get(rt, 1500.0)
        f[f"home_elo_{rt}"], f[f"away_elo_{rt}"], f[f"elo_{rt}_diff"] = h, a, h - a
    for side, team in (("home", home), ("away", away)):
        for key, value in engine.calculate_advanced_recent_form(team, day, recent).items():
            f[f"{side}_{key}"] = value
    f.update(engine.calculate_head_to_head(home, away, day, recent))
    for side, team in (("home", home), ("away", away)):
        for key, value in engine.calculate_travel_fatigue(team, day, recent).items():
            f[f"{side}_{key}"] = value
    situational = engine.calculate_advanced_situational_factors(home, away, day, recent)
    f.update({k: v for k, v in situational.items() if k not in SIMULATED_SITUATIONAL})
    for side, team in (("home", home), ("away", away)):
        for key, value in engine.calculate_advanced_team_metrics(team, day, recent).items():
            f[f"{side}_{key}"] = value
        for key, value in engine.calculate_momentum_indicators(team, day, recent).items():
            f[f"{side}_{key}"] = value
    home_elo = engine.team_elo_ratings.get(home, 1500.0)
    away_elo = engine.team_elo_ratings.get(away, 1500.0)
    f["team_strength_diff"] = home_elo - away_elo
    f["team_strength_ratio"] = home_elo / away_elo
    f["season_progression"] = min(1.0, (day - pd.Timestamp(day.year, 10, 1)).days / 200)
    f["home_rest_days"] = engine._calculate_rest_days(home, day, recent)
    f["away_rest_days"] = engine._calculate_rest_days(away, day, recent)
    f["rest_advantage"] = f["home_rest_days"] - f["away_rest_days"]
    f["is_postseason"] = int(season_type != 2)
    for col in BASE_COLUMNS:
        f[col] = stats.at[home, col]
        f[f"{col}.1"] = stats.at[away, col]
    return f


class FeaturePipeline:
    """Loads ESPN history and builds features for upcoming games."""

    def __init__(self, espn=None, verbose=True):
        self.espn = espn or EspnDataProvider(cache_dir=os.path.join(REPO_ROOT, "Data", "espn_cache"))
        self.verbose = verbose
        self.loaded_for = None

    def load(self, game_day):
        """Refresh results and box scores for everything before `game_day`."""
        if self.loaded_for == game_day:
            return
        last = max(HISTORY_START, game_day - timedelta(days=1))
        espn_games = self.espn.update_games(HISTORY_START, last)
        counted = espn_games[espn_games["completed"] & espn_games["season_type"].isin(COUNTED_SEASON_TYPES)]
        boxes = self.espn.ensure_boxes(counted[counted["date"] >= ESPN_STATS_FROM]["game_id"])
        self.history = game_history(espn_games[espn_games["date"] < f"{game_day:%Y-%m-%d}"])
        self.team_games = team_game_rows(espn_games, boxes)
        self.stats = TeamStatsSource(self.team_games)
        self.elo = PointInTimeElo(self.history)
        self.loaded_for = game_day
        if self.verbose:
            print(f"[OK] Game history loaded: {len(self.history)} games, "
                  f"latest result {self.history['date'].max():%Y-%m-%d}")

    def features_for(self, home, away, game_day, season_type=2):
        home, away = normalize_team(home), normalize_team(away)
        self.load(game_day)
        stats = self.stats.on(game_day)
        for team in (home, away):
            if team not in stats.index:
                raise KeyError(f"{team} has no games yet this season")
        day = pd.Timestamp(game_day)
        return game_features(home, away, day, self.history[self.history["date"] < day],
                             self.elo.as_of(day), stats, season_type)
