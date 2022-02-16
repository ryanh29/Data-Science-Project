import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('advertising.csv')
print(df)
print(df.columns)

sns.pairplot(df, hue='Clicked on Ad')
plt.show()

print(df.describe())

sns.histplot(data=df,x='Age')
plt.show()

sns.jointplot(x='Age', y='Area Income', data=df)
plt.show()

sns.jointplot(x='Age', y='Daily Time Spent on Site', data=df, kind='kde')
plt.show()

sns.jointplot(x='Daily Internet Usage', y='Daily Time Spent on Site', data=df, kind='hex')
plt.show()

X = df[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

#check nulls
sns.heatmap(df.isnull())
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)
print(predictions)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))



