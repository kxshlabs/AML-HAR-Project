import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("===== Human Activity Recognition using HMM =====")

# Load Dataset
X_train = pd.read_csv("X_train.txt", sep=r"\s+", header=None)
y_train = pd.read_csv("y_train.txt", sep=r"\s+", header=None)
X_test = pd.read_csv("X_test.txt", sep=r"\s+", header=None)
y_test = pd.read_csv("y_test.txt", sep=r"\s+", header=None)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

# Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=15)

model = hmm.GaussianHMM(
    n_components=6,
    covariance_type="full",
    n_iter=150
)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# PCA Variance Plot
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.title("PCA Explained Variance")
plt.xlabel("Components")
plt.ylabel("Variance")
plt.show()

# Train HMM
print("Training HMM...")
model = hmm.GaussianHMM(n_components=6, covariance_type="diag", n_iter=50)
model.fit(X_train)

# Predict hidden states
train_states = model.predict(X_train)
test_states = model.predict(X_test)

# Map hidden states to actual labels
state_mapping = {}
for i in range(6):
    mask = (train_states == i)
    if np.sum(mask) > 0:
        state_mapping[i] = np.bincount(y_train[mask]).argmax()

# Convert predicted states to activity labels
y_pred = np.array([state_mapping.get(state, 0) for state in test_states])

# Performance Metrics
print("\n===== Performance Metrics =====")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

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

# Transition Matrix
plt.figure()
sns.heatmap(model.transmat_, annot=True)
plt.title("Transition Probability Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# Hidden State Sequence Plot
plt.figure()
plt.plot(test_states[:200])
plt.title("Hidden State Sequence")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()

# Activity Distribution
plt.figure()
sns.countplot(x=y_test)
plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.show()

# Log Likelihood
log_likelihood = model.score(X_train)
print("\nLog Likelihood:", log_likelihood)

print("\n===== Project Completed =====")