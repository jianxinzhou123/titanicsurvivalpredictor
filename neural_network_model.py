from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import preprocessing

def main():
    np.random.seed(7)
    np.set_printoptions(suppress=True)

    df = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[2,4,5,6,7,9])
    df.fillna(0, inplace=True)
    raw_dataset = df.values

    df2 = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[1])
    df2.fillna(0, inplace=True)
    survival_dataset = df2.values

    min_max_scaler = preprocessing.MinMaxScaler()
    scale = min_max_scaler.fit_transform(raw_dataset)
    normalized_data =pd.DataFrame(scale)
    actualset = normalized_data.values

    candidate = actualset[:, 2:9]
    survival = survival_dataset[:, 0]

    model = Sequential()

    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(candidate, survival, epochs=200, batch_size=300)

    scores = model.evaluate(candidate, survival)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save('my_model.h5')
    del model
    model=load_model('my_model.h5')

if __name__ == '__main__':
    this = main()
