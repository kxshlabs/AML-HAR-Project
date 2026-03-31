import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("===== Human Activity Recognition using HMM =====")

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

X_train = pd.read_csv("X_train.txt", sep=r"\s+", header=None)
y_train = pd.read_csv("y_train.txt", sep=r"\s+", header=None)
X_test = pd.read_csv("X_test.txt", sep=r"\s+", header=None)
y_test = pd.read_csv("y_test.txt", sep=r"\s+", header=None)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

# -----------------------------
# Normalize Data
# -----------------------------
print("Normalizing data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# PCA (Dimensionality Reduction)
# -----------------------------
print("Applying PCA...")
pca = PCA(n_components=20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# PCA Variance Graph
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.title("PCA Explained Variance")
plt.xlabel("Principal Components")
plt.ylabel("Variance Ratio")
plt.show()

# -----------------------------
# Train HMM Model
# -----------------------------
print("Training Hidden Markov Model...")
model = hmm.GaussianHMM(n_components=6, covariance_type="diag", n_iter=50)
model.fit(X_train)

# -----------------------------
# Predict Activities
# -----------------------------
print("Predicting activities...")
y_pred = model.predict(X_test)

# -----------------------------
# Performance Metrics
# -----------------------------
print("\n===== Performance Metrics =====")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Transition Matrix Heatmap
# -----------------------------
plt.figure()
sns.heatmap(model.transmat_, annot=True)
plt.title("Transition Probability Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# -----------------------------
# Hidden State Sequence Plot
# -----------------------------
plt.figure()
plt.plot(y_pred[:200])
plt.title("Hidden State Sequence (First 200 Samples)")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()

# -----------------------------
# Activity Distribution
# -----------------------------
plt.figure()
sns.countplot(x=y_test)
plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Log Likelihood
# -----------------------------
log_likelihood = model.score(X_train)
print("\nLog Likelihood:", log_likelihood)

print("\n===== Project Completed =====")