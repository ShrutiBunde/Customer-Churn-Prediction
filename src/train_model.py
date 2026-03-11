import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

from preprocess import preprocess_data


# load dataset
df = pd.read_csv("data/churn.csv")

# preprocessing
df = preprocess_data(df)

# features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

# prediction
pred = rf.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# save model
pickle.dump(rf, open("../churn_model.pkl", "wb"))
pickle.dump(X.columns, open("model_columns.pkl", "wb"))