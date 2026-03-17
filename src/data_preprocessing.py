import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(path):

    data = pd.read_csv(path)

    categorical_cols = ['protocol_type','service','flag']

    encoder = LabelEncoder()

    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])

    X = data.drop("label", axis=1)
    y = data["label"]

    return X, y