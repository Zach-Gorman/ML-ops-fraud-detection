import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import load_and_preprocess_data


os.makedirs("models", exist_ok=True)

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

if recall < 0.70:
    raise Exception("Recall below threshold")

joblib.dump(model, "models/model.pkl")
print("Model saved")