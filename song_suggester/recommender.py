from model import model_knn
import pandas as pd
import random
import numpy as np
from typing import List, Tuple

df = pd.read_csv('https://raw.githubusercontent.com/aguilargallardo/DS-Unit-2-Applied-Modeling/master/data/SpotifyFeatures.csv')
#df = pd.read_sql(query.statement, query.session.bind) #Left this generalized, need help to connect from mark

#remove string values for model fitting
not_features = ['artist_name', 'song_name']
features=[i for i in list(df.columns) if i not in not_features]
X = df[features]

model_knn.fit(X)
"""
function that automates pulling song recommendations, done by index. returns 10 recommendations based on the selection
Written by Greg
"""
def rec_10(df: pd.DataFrame, X_array: np.ndarray ,song_id: int) -> List[Tuple] :
    song = df.iloc[song_id]
    X_song = X.iloc[song_id]
    _, neighbors = model_knn.kneighbors(np.array([X.iloc[180]]))
    song_list = []
    for x in neighbors[0][2:]: 
        row = df.iloc[x].to_frame().T
        song_list.append(row['song_name'])
    rec_10 = random.sample(song_list, 10)
    return(rec_10)
