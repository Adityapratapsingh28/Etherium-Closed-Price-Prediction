import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

#Load the csv file
df = pd.read_csv("etherium.csv")
#print(df)

#Select dependent and independent variables

x = df[["open","high","low"]]
y = df["close"]

#Splitting the data into training and testing

X_train , X_test , Y_train , Y_test = train_test_split(x , y, test_size=0.3 , random_state = 0 )

#Fitting the model

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

#make pickle file for our model

pickle.dump(lin_reg , open("model.pkl" , "wb"))

model = pickle.load(open("model.pkl", "rb"))