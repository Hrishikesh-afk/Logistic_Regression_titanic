%pip install seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Titanic-Dataset.csv")
df.head()

df.info()

df.describe()

df = df.drop(columns=["PassengerId", "Cabin", "Name", "Ticket"])
df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})
mode=df["Embarked"].mode()[0]
df["Embarked"]=df["Embarked"].fillna(mode)
median=df["Age"].median()
df["Age"]=df["Age"].fillna(median)
df.info()

X=df.drop("Survived",axis=1)
y=df["Survived"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)
classification=classification_report(y_test,y_pred)
print("Classification Report:\n",classification)
