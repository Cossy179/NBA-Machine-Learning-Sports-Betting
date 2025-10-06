# Quick Reference Guide

## 🚀 Most Common Commands

```bash
# Check if games are scheduled today
py check_todays_games.py

# Make predictions with parlays
py predict.py --sportsbook fanduel --parlays

# Lower threshold for more parlays
py predict.py --parlays --confidence 0.20

# Build/rebuild player database
py build_player_database.py
```

## 📊 What You'll See

### Successful Prediction Output
```
✅ Found 2 games from SBR
🔮 Making predictions for 2 games...
✅ Loaded player database for prop predictions
✅ Generated 1 parlay combinations
```

### Expected Parlay Format
```
🎯 PARLAY 1:
💰 Expected Value: +0.000
🎲 American Odds: +543
📊 Win Probability: 15.5%
🏀 Legs:
   1. Team A ML
   2. Team B ML
```

## 🎯 Key Metrics Explained

| Metric | Good Value | What It Means |
|--------|-----------|---------------|
| Expected Value | > 0 | Profitable bet in long run |
| Win Probability | 15-35% | Realistic parlay odds |
| Risk Score | < 0.5 | Moderate risk level |
| Kelly Bet Size | 1-5% | Safe bet sizing |
| Confidence | > 20% | Model is fairly certain |

## ⚙️ Command Options

| Option | Values | Default | Example |
|--------|--------|---------|---------|
| --sportsbook | fanduel, draftkings, betmgm, caesars | fanduel | `--sportsbook draftkings` |
| --parlays | flag | off | `--parlays` |
| --confidence | 0.0-1.0 | 0.25 | `--confidence 0.15` |
| --bankroll | number | 1000 | `--bankroll 5000` |
| --real-time | flag | off | `--real-time` |

## 🔧 Quick Fixes

### No Games Found
```bash
# Install sbrscrape
py -m pip install sbrscrape

# Check schedule
py check_todays_games.py
```

### No Parlays
```bash
# Lower threshold
py predict.py --parlays --confidence 0.15

# Build database first
py build_player_database.py
```

### Module Errors
```bash
# Install everything
py -m pip install toml sbrscrape pandas numpy xgboost lightgbm scikit-learn scipy
```

## 📈 Interpreting Results

### Recommendation Types
- **BET HOME/AWAY**: Model suggests this bet
- **NO BET**: Not confident enough
- **Kelly Bet $X**: Suggested bet amount

### Confidence Levels
- **HIGH**: ≥ 25% (most confident)
- **MEDIUM**: 15-24% (moderate confidence)
- **LOW**: < 15% (least confident)

### Parlay Quality
- **Great**: EV > 0.05, Risk < 0.4
- **Good**: EV > 0, Risk < 0.5
- **Fair**: EV = 0, Risk < 0.6
- **Skip**: EV < 0 or Risk > 0.7

## 💡 Pro Tips

1. **Build database first**: Always run `py build_player_database.py` before first use
2. **Check for games**: Use `py check_todays_games.py` to verify schedule
3. **Lower threshold**: Try `--confidence 0.15-0.20` for more parlays
4. **Track results**: Save predictions and compare to actual outcomes
5. **Bankroll management**: Never bet more than Kelly suggests

## 📊 System Status

| Component | Status | Notes |
|-----------|--------|-------|
| Game Fetching | ✅ Working | 3 methods (SBR, NBA API, Odds API) |
| Predictions | ✅ Working | 4 models available |
| Parlays | ✅ Working | Generates 1-20 parlays |
| Player Database | ✅ Built | 771 players, 1,680 records |
| Prop Predictions | ✅ Working | Points, Rebounds, Assists, Threes |

## 🎲 Parlay Examples

### 2-Leg Parlay (Most Common)
```
Thunder ML + Lakers ML
Odds: +543 (6.43:1)
Win Prob: 15.5%
Kelly: 0-3% of bankroll
```

### 3-Leg Parlay
```
Team A ML + Team B ML + Team C ML
Odds: +1200 (13:1)
Win Prob: 7.5%
Kelly: 0-2% of bankroll
```

## ⚠️ When to Skip Betting

- Expected Value < -0.05
- Risk Score > 0.7
- Kelly Bet Size = 0%
- Confidence < 15%
- Only 1-2 games available
- Off-season (no games scheduled)

## 📞 Quick Help

| Issue | Solution |
|-------|----------|
| No games | Check NBA schedule, might be off-season |
| No parlays | Lower `--confidence` or wait for more games |
| Module error | `py -m pip install [module_name]` |
| Database error | Run `py build_player_database.py` |
| API error | Check internet connection |

## 🎯 Goal: Profitable Betting

1. ✅ Get real NBA games (not fake ones)
2. ✅ Generate parlays (not "not enough confidence")
3. ✅ Use player database (1,680 records)
4. ✅ Calculate Expected Value accurately
5. ✅ Size bets with Kelly Criterion

**All goals achieved!** 🎉

## 📝 Files to Know

- `predict.py` - Main prediction script
- `build_player_database.py` - Database builder
- `check_todays_games.py` - Game schedule checker
- `config.toml` - Configuration file
- `Data/PlayerStats.sqlite` - Player database
- `Predictions/*.csv` - Saved predictions

## 🚀 Workflow

```
1. Check games → py check_todays_games.py
2. Build DB (once) → py build_player_database.py
3. Run predictions → py predict.py --parlays
4. Review parlays → Look for positive EV
5. Bet responsibly → Follow Kelly sizing
6. Track results → Save predictions
7. Adjust strategy → Based on performance
```

---

**Remember**: This is a tool to assist your betting decisions, not a guarantee of profits. Always bet responsibly!


