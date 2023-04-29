from google.colab import drive
import os 
import pickle
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
 
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/')
 
train = pd.read_csv("DSP_13.csv", sep = ";")
train = train.fillna(train.mean())
X = train.drop('zdrowie', axis = 1)
y = train['zdrowie']
 
X_train, X_test, y_train, y_test = train_test_split(train.drop('zdrowie', axis = 1),
                                                               train['zdrowie'], test_size=0.2,
                                                               random_state=101)
 
model = RandomForestClassifier(n_estimators = 10, max_depth = 3, random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
filename = 'model.sv'
pickle.dump(model, open(filename, 'wb'))