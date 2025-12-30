import json
import pickle
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

print("Training model...")
start_time = time.time()
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
training_time = time.time() - start_time

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ model.pkl saved")

metrics = {
    'accuracy': float(accuracy),
    'training_time_seconds': float(training_time),
    'model_params': 1000
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ metrics.json saved")

print("Done!")
