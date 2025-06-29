import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os
import sys
import joblib 

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)
data_dir = os.path.join('data','iris.csv')
csv_path = os.path.join(path,data_dir)
model_path = os.path.join(path,'model','model.pkl')
data = pd.read_csv(csv_path)

# Spilt the data
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Train the model
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)

# Save the model
with open(model_path,'wb') as file:
    joblib.dump(mod_dt,file)

# Test the model
prediction=mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction,y_test)
print(f"Accuracy: {accuracy:.4f}")