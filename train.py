import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os
import sys
import joblib 
import mlflow
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings('ignore')

mlflow_flag = 0

try:
    mlflow.set_tracking_uri("http://34.61.158.253:8200")
    print(f"MLFLOW tracking uri: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment("Poisoning Iris Data Experiment")
    print("Experiment set: Poisoning Iris Data Experiment")
    mlflow_flag = 1
except Exception as e:
    print("Could not connect to mlflow server")

# Define paths
pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)
data_dir = os.path.join('data','iris.csv')
csv_path = os.path.join(path,data_dir)
model_path = os.path.join(path,'model','model.pkl')

# Get poison percent
if len(sys.argv) > 1:
    POISON_PERCENT = int(sys.argv[1])  
else:
    POISON_PERCENT = 0 

# Load and spilt the data
data = pd.read_csv(csv_path)
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train['species'].values.copy()
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test['species'].values.copy()

# Posion the training labels
if POISON_PERCENT > 0:
    n_poison = int(len(y_train) * POISON_PERCENT / 100)
    indices = np.random.choice(len(y_train), n_poison, replace=False)
    unique_labels = np.unique(y_train)

    for i in indices:
        current_label = y_train[i]
        other_labels = [label for label in unique_labels if label != current_label]
        y_train[i] = np.random.choice(other_labels)

# Train the model
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)

# Test the model
prediction=mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)
f1_macro = metrics.f1_score(y_test, prediction, average='macro')
print(f"Label Poisoning: {POISON_PERCENT}%")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")

# Log to MLFLOW
if mlflow_flag:
    with mlflow.start_run():
        mlflow.log_params({"POISON_PERCENT": POISON_PERCENT})
        mlflow.log_metric("Accuracy", f"{accuracy:.4f}")
        mlflow.log_metric("Macro F1 Score", f" {f1_macro:.4f}")
        mlflow.set_tag("Training Info",f"Decision Tree on Iris Dataset with Label Poisoning of {POISON_PERCENT}%")
        signature = infer_signature(X_train,prediction)
        model_info = mlflow.sklearn.log_model(
            sk_model = mod_dt,
            name = "Iris Classifier",
            signature = signature,
            input_example = X_train,
            registered_model_name = f"DT_iris_model"
        )
        print("Run tracked in mlflow")

# Save the model
with open(model_path,'wb') as file:
    joblib.dump(mod_dt,file)