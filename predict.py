##JIANXIN ZHOU

import neural_network_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import preprocessing

model=load_model('my_model.h5')

np.random.seed(7)
np.set_printoptions(suppress=True)

df = pd.read_csv("Test.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[1,3,4,5,6,8])
df.fillna(0, inplace=True)
dataset = df.values
min_max_scaler = preprocessing.MinMaxScaler()
scale = min_max_scaler.fit_transform(dataset)
normalized_data =pd.DataFrame(scale)
predictionset = normalized_data.values

candidate = predictionset[:, 2:9]

predictions = model.predict(candidate)

print(predictions)
