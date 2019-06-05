from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import preprocessing

def func(x):
    #do stuff here

def main():
    np.random.seed(7)
    np.set_printoptions(suppress=True)
    
    df = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[2,4,5,6,7,9])
    df.fillna(0, inplace=True)
    dataset = df.values
    
    df2 = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[1])
    df2.fillna(0, inplace=True)
    survival_dataset = df2.values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    scale = min_max_scaler.fit_transform(dataset)
    normalized_data =pd.DataFrame(scale)
    actualset = normalized_data.values
    
    candidate = actualset[:, 2:9]
    survival = survival_dataset[:, 0]
    
    model = Sequential()
    
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    # GA standard settings
    generation_num = 50
    population_size = 16
    elitism = True
    selection = 'rank'
    tournament_size = None # in case of tournament selection
    mut_type = 1
    mut_prob = 0.05
    cross_type = 1
    cross_prob = 0.95
    optim = 'min' # minimize or maximize a fitness value? May be 'min' or 'max'.
    interval = (0, 1000)
    
    sga = RealGA(func, optim=optim, elitism=elitism, selection=selection,
        mut_type=mut_type, mut_prob=mut_prob, 
        cross_type=cross_type, cross_prob=cross_prob)
                    
    sga.init_random_population(population_size, 1, interval)
    
if _name_ == '_main_':
    main()
