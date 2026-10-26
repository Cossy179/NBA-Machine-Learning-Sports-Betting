# NBA Sports Betting Using Machine Learning 🏀
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

A machine learning AI used to predict the winners and under/overs of NBA games. Takes all team data from the 2007-08 season to current season, matched with odds of those games, using a neural network to predict winning bets for today's games. Achieves ~69% accuracy on money lines and ~55% on under/overs. Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on the Kelly Criterion is also outputted. Note that a popular, less risky approach is to bet 50% of the stake recommended by the Kelly Criterion.
## Quick start (LightGBM v2)

```bash
pip install -r requirements.txt
python src/Train-Models/LightGBM_V2.py   # download data, evaluate, train Models/LightGBM_V2/
python predict.py                        # tonight's games, with news sentiment
```

`predict.py` uses the LightGBM v2 model in `Models/LightGBM_V2/`. Its inputs are built by
`src/Predict/LiveFeatureBuilder.py`, the same code that builds the training data:

* season-to-date team stats and league ranks (official NBA tables up to 2021-22, rebuilt
  from ESPN box scores after that, because stats.nba.com blocks most cloud networks),
* Elo ratings, recent form, head-to-head, rest and momentum, using only games played
  before the game day.

Game data is downloaded from ESPN into `Data/espn_cache/` (not committed). Re-run the
training script to bring the model up to date; it trains on every game from 2012-13 to
yesterday and reports its accuracy on the 2025-26 season, which it is not trained on
during evaluation. `python predict.py --model-info` shows those results.

Bets are only recommended when the model's probability gives at least +3% expected
value at the offered moneyline (`--min-edge`), sized with half Kelly capped at 5% of the
bankroll (`--kelly-fraction`). If the chosen sportsbook has not posted a line yet, the
sportsbookreview consensus (or the median across books) is used. `--scaled-kelly` keeps
the old mode that stakes the whole bankroll. News sentiment is on by default
(`--no-sentiment` turns it off); its adjustment is applied after the model and was not
part of the backtest.

## Packages Used

Use Python 3.11. In particular the packages/libraries used are...

* Tensorflow - Machine learning library
* XGBoost - Gradient boosting framework
* Numpy - Package for scientific computing in Python
* Pandas - Data manipulation and analysis
* Colorama - Color text output
* Tqdm - Progress bars
* Requests - Http library
* Scikit_learn - Machine learning library

## Usage

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Expected_value.png" width="1010" height="424" />

Make sure all packages above are installed.

```bash
$ git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
$ cd NBA-Machine-Learning-Sports-Betting
$ pip3 install -r requirements.txt
$ python3 predict.py --sportsbook fanduel
```

Odds data will be automatically fetched from sbrodds if the -odds option is provided with a sportsbook.  Options include: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny

If `-odds` is not given, enter the under/over and odds for today's games manually after starting the script.

Optionally, you can add '-kc' as a command line argument to see the recommended fraction of your bankroll to wager based on the model's edge

## Flask Web App
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

This repo also includes a small Flask application to help view the data from this tool in the browser.  To run it:
```
cd Flask
flask --debug run
```

## Getting new data and training models
```
# Create dataset with the latest data for 2023-24 season
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Contributing

All contributions welcomed and encouraged.
