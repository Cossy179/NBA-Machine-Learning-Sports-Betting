import sqlite3
import sys
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src/Utils to path for temporal weights
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from temporal_weights import calculate_temporal_weights

dataset = "dataset_2012-25_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Store dates before dropping
dates = pd.to_datetime(data['Date'])

margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)

data = data.values
data = data.astype(float)

# Calculate temporal weights
temporal_weights = calculate_temporal_weights(dates, recent_season_start=2021, decay_factor=0.7)

acc_results = []
for x in tqdm(range(300)):
    x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(
        data, margin, temporal_weights, test_size=.1
    )

    train = xgb.DMatrix(x_train, label=y_train, weight=weights_train)
    test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'multi:softprob',
        'num_class': 2
    }
    epochs = 750

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))

    acc = round(accuracy_score(y_test, y) * 100, 1)
    print(f"{acc}%")
    acc_results.append(acc)
    # only save results if they are the best so far
    if acc == max(acc_results):
        model.save_model('../../Models/XGBoost_{}%_ML-4.json'.format(acc))
