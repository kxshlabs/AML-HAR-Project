import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
print("Loading dataset...")

X_train = pd.read_csv("X_train.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv("y_train.txt", delim_whitespace=True, header=None)

X_test = pd.read_csv("X_test.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv("y_test.txt", delim_whitespace=True, header=None)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Convert to numpy
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

# Train Hidden Markov Model
print("\nTraining HMM Model...")

n_states = 6  # Number of activities
model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)

model.fit(X_train)

# Predict Hidden States
print("Predicting activities...")
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Transition Matrix Heatmap
plt.figure()
sns.heatmap(model.transmat_, annot=True)
plt.title("Transition Probability Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# Hidden State Sequence Plot
plt.figure()
plt.plot(y_pred[:200])
plt.title("Predicted Activity States (First 200 Samples)")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()

# Activity Distribution Plot
plt.figure()
sns.countplot(x=y_test)
plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.show()

print("\nProject Execution Completed.")