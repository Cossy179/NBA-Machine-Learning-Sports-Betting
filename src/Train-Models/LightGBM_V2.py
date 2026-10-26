"""
Train the LightGBM v2 game-winner model on every game from 2012-13 to yesterday.

    python src/Train-Models/LightGBM_V2.py            # rebuild data, evaluate, train, save
    python src/Train-Models/LightGBM_V2.py --skip-final   # evaluation only

Steps:
1. Download (or update) ESPN results and box scores.
2. Build features for every game with src/Predict/LiveFeatureBuilder.py,
   the same code predict.py uses for tonight's games.
3. Evaluate honestly: train on 2012-13..2023-24, pick the number of trees on
   2024-25, report on 2025-26 (and on the current season so far).
4. Retrain on everything and save Models/LightGBM_V2/.
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "Predict"))
os.chdir(REPO_ROOT)

import LiveFeatureBuilder as fb  # noqa: E402
from EspnDataProvider import COUNTED_SEASON_TYPES, EspnDataProvider  # noqa: E402

MODEL_DIR = os.path.join(REPO_ROOT, "Models", "LightGBM_V2")
FEATURES_CSV = os.path.join(REPO_ROOT, "Data", "espn_cache", "features_v2.csv")
ODDS_CSV = os.path.join(REPO_ROOT, "Data", "espn_cache", "sbr_odds_2026.csv")
LABEL = "home_win"
META_COLUMNS = ["game_id", "date", "season", "season_type", "home_team", "away_team", LABEL]

# Starting point: the hyperparameters of the original SuperAdvanced LightGBM model
PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.032,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 45,
    "feature_fraction": 0.71,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l1": 0.22,
    "lambda_l2": 0.23,
    "seed": 42,
    "deterministic": True,
    "verbose": -1,
}

_STATE = {}


def _build_one(row):
    history = _STATE["history"]
    day = row["date"]
    stats = _STATE["stats"].on(day)
    if row["home_team"] not in stats.index or row["away_team"] not in stats.index:
        return None
    if stats.at[row["home_team"], "GP"] < 1 or stats.at[row["away_team"], "GP"] < 1:
        return None
    f = fb.game_features(row["home_team"], row["away_team"], day, history[history["date"] < day],
                         _STATE["elo"].as_of(day), stats, row["season_type"])
    f.update({k: row[k] for k in META_COLUMNS})
    return f


def build_dataset(through_day, workers=4):
    espn = EspnDataProvider()
    print("Updating ESPN results and box scores...")
    games = espn.update_games(fb.HISTORY_START, through_day)
    counted = games[games["completed"] & games["season_type"].isin(COUNTED_SEASON_TYPES)]
    boxes = espn.ensure_boxes(counted[counted["date"] >= fb.ESPN_STATS_FROM]["game_id"])
    history = fb.game_history(games[games["date"] <= f"{through_day:%Y-%m-%d}"])
    team_games = fb.team_game_rows(games, boxes)
    stats = fb.TeamStatsSource(team_games)
    for day in history["date"].unique():  # warm the cache before forking
        stats.on(day)
    _STATE.update(history=history, stats=stats, elo=fb.PointInTimeElo(history))
    rows = history.to_dict("records")
    print(f"Building features for {len(rows)} games with {workers} workers...")
    with Pool(workers) as pool:
        built = [r for r in pool.imap(_build_one, rows, chunksize=50) if r is not None]
    data = pd.DataFrame(built)
    data["date"] = pd.to_datetime(data["date"])
    data[LABEL] = data[LABEL].astype(int)
    data.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(data)} rows to {FEATURES_CSV}")
    return data


def _decimal(odds):
    odds = float(odds)
    return odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1


def fetch_closing_odds(games):
    """sportsbookreview moneylines for the given games (cached)."""
    from concurrent.futures import ThreadPoolExecutor
    from sbrscrape import Scoreboard

    if os.path.exists(ODDS_CSV):
        return pd.read_csv(ODDS_CSV)
    days = sorted(games["date"].dt.strftime("%Y-%m-%d").unique())

    def fetch(day):
        for _ in range(3):
            try:
                return [{"date": day, "home_team": g["home_team"], "away_team": g["away_team"],
                         "home_ml": json.dumps(g.get("home_ml", {})), "away_ml": json.dumps(g.get("away_ml", {}))}
                        for g in Scoreboard(sport="NBA", date=day).games or []
                        if g.get("source") == "sportsbookreview"]
            except Exception:
                pass
        return []

    with ThreadPoolExecutor(6) as pool:
        rows = [r for day_rows in pool.map(fetch, days) for r in day_rows]
    odds = pd.DataFrame(rows)
    odds.to_csv(ODDS_CSV, index=False)
    return odds


def betting_backtest(test, probs, book="fanduel", min_edge=0.03):
    """Model vs closing odds on the test season, and flat-stake results of betting
    every side with at least `min_edge` expected value."""
    odds = fetch_closing_odds(test)
    for col in ("home_team", "away_team"):
        odds[col] = odds[col].replace({"Los Angeles Clippers": "LA Clippers", "L.A. Clippers": "LA Clippers",
                                       "L.A. Lakers": "Los Angeles Lakers"})

    def line(r):
        h, a = json.loads(r.home_ml), json.loads(r.away_ml)
        for b in (book, "consensus"):
            if h.get(b) and a.get(b):
                return _decimal(h[b]), _decimal(a[b])
        return np.nan, np.nan

    odds[["dec_home", "dec_away"]] = odds.apply(lambda r: pd.Series(line(r)), axis=1)
    t = test[["date", "home_team", "away_team", LABEL]].copy()
    t["prob"] = probs
    t["date"] = t["date"].dt.strftime("%Y-%m-%d")
    m = t.merge(odds.dropna(subset=["dec_home"]).drop_duplicates(["date", "home_team", "away_team"]),
                on=["date", "home_team", "away_team"])
    if m.empty:
        return {}
    y, p = m[LABEL].values, m["prob"].values
    inv_h, inv_a = 1 / m["dec_home"].values, 1 / m["dec_away"].values
    market = inv_h / (inv_h + inv_a)
    ev_home, ev_away = p * m["dec_home"].values - 1, (1 - p) * m["dec_away"].values - 1
    bet = np.maximum(ev_home, ev_away) >= min_edge
    home_side = ev_home >= ev_away
    won = np.where(home_side, y == 1, y == 0)
    profit = np.where(won, np.where(home_side, m["dec_home"].values, m["dec_away"].values) - 1, -1)[bet]
    return {
        "games_with_odds": int(len(m)),
        "model": metrics(y, p),
        "bookmakers": metrics(y, market),
        "bets": int(bet.sum()),
        "bet_win_rate": round(float(won[bet].mean()), 4) if bet.any() else None,
        "flat_stake_roi": round(float(profit.mean()), 4) if bet.any() else None,
        "min_edge": min_edge,
    }


def feature_columns(data):
    cols = [c for c in data.columns if c not in META_COLUMNS]
    # Drop features that never vary (placeholders in the feature engine)
    return [c for c in cols if data[c].nunique(dropna=False) > 1]


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "games": int(len(y)),
        "accuracy": round(float(((p > 0.5) == y).mean()), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
    }


def train(data, cols, rounds=None, valid=None):
    dtrain = lgb.Dataset(data[cols], data[LABEL])
    if valid is None:
        return lgb.train(PARAMS, dtrain, num_boost_round=rounds)
    dvalid = lgb.Dataset(valid[cols], valid[LABEL], reference=dtrain)
    return lgb.train(PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features even if cached")
    parser.add_argument("--skip-final", action="store_true", help="Evaluate only, don't save a model")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    through = date.today() - timedelta(days=1)
    if args.rebuild or not os.path.exists(FEATURES_CSV):
        data = build_dataset(through, args.workers)
    else:
        data = pd.read_csv(FEATURES_CSV, parse_dates=["date"], dtype={"game_id": str})
        if data["date"].max().date() < through:
            data = build_dataset(through, args.workers)
    cols = feature_columns(data)
    print(f"{len(data)} games, {len(cols)} features, "
          f"{data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d}")

    # Honest evaluation: tune on 2024-25, test on 2025-26 and the current season
    tr, va = data[data["season"] <= 2024], data[data["season"] == 2025]
    te, cur = data[data["season"] == 2026], data[data["season"] >= 2027]
    model = train(tr, cols, valid=va)
    best = model.best_iteration
    report = {"trees": best, "validation_2024_25": metrics(va[LABEL].values, model.predict(va[cols], num_iteration=best))}
    # Refit on 2012-13..2024-25 to score the 2025-26 test season
    refit = train(pd.concat([tr, va]), cols, rounds=best)
    report["test_2025_26"] = metrics(te[LABEL].values, refit.predict(te[cols]))
    report["test_2025_26_always_home"] = metrics(te[LABEL].values, np.full(len(te), pd.concat([tr, va])[LABEL].mean()))
    if len(cur):
        report["current_season_so_far"] = metrics(cur[LABEL].values, refit.predict(cur[cols]))
    print(json.dumps(report, indent=2))
    te_pred = te[META_COLUMNS].copy()
    te_pred["prob_home"] = refit.predict(te[cols])
    try:
        report["betting_2025_26"] = betting_backtest(te, te_pred["prob_home"].values)
    except Exception as e:  # odds are optional
        print(f"[WARNING] Betting backtest skipped: {e}")
    te_pred.to_csv(os.path.join(REPO_ROOT, "Data", "espn_cache", "test_2025_26_predictions.csv"), index=False)

    if args.skip_final:
        return
    final = train(data, cols, rounds=best)
    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, "LightGBM_V2.txt"))
    with open(os.path.join(MODEL_DIR, "metadata.txt"), "w") as fh:
        fh.write(json.dumps({
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "games": int(len(data)),
            "first_game": f"{data['date'].min():%Y-%m-%d}",
            "last_game": f"{data['date'].max():%Y-%m-%d}",
            "features": cols,
            "params": PARAMS,
            "evaluation": report,
        }, indent=2))
    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
