from Perceptron import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
#read data from .csv file
data=pd.read_csv("iris.csv")
data.columns=["petal_length","petal_width","sepal_length","sepal_width","class"]
#training a perceptron
classes=data["class"]
data=data.drop(columns="class")
#splitting test and train data for iris
x_train,x_test,y_train,y_test=train_test_split(data,classes)
#training the weights for percpetron
p=Perceptron()
p.fit(x_train,y_train,0.5,10)
pred=p.predict(x_test)
print("accuracy: ",p.accuracy(pred,y_test))
print("weights: ",p.getweights())
