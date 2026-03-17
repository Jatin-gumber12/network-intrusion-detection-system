from sklearn.metrics import classification_report
import joblib
import pandas as pd

model = joblib.load("models/intrusion_model.pkl")

data = pd.read_csv("data/KDDTest+.txt")

X = data.drop("label", axis=1)
y = data["label"]

pred = model.predict(X)

print(classification_report(y, pred))