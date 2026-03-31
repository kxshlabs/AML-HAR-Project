import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

np.random.seed(42)

print("Weather State Prediction using Hidden Markov Model")

# Load dataset
data = pd.read_csv("BeijingPM2.5.csv")

# Select features
data = data[['TEMP', 'DEWP', 'PRES', 'Iws']]
data = data.dropna()

# -----------------------------
# Balanced Weather Labels (Quantiles)
# -----------------------------
data['WeatherLabel'] = pd.qcut(data['TEMP'], 4, labels=False)

# Features
X = data[['TEMP', 'DEWP', 'PRES', 'Iws']].values
y = data['WeatherLabel'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split (time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Train HMM
model = hmm.GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=300,
    random_state=42
)
model.fit(X_train)

# Predict states
train_states = model.predict(X_train)
test_states = model.predict(X_test)

# Map states to labels
state_mapping = {}
for i in range(4):
    mask = (train_states == i)
    if np.sum(mask) > 0:
        state_mapping[i] = np.bincount(y_train[mask]).argmax()

y_pred = np.array([state_mapping[s] for s in test_states])

# Metrics
print("\n===== PERFORMANCE METRICS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# Transition Matrix
plt.figure()
sns.heatmap(model.transmat_, annot=True)
plt.title("Transition Matrix")
plt.show()

# Hidden State Sequence
plt.figure()
plt.plot(test_states[:200])
plt.title("Hidden State Sequence")
plt.show()

# State Distribution
plt.figure()
sns.countplot(x=test_states)
plt.title("Hidden State Distribution")
plt.show()

# Log Likelihood
print("\nLog Likelihood:", model.score(X_train))

print("\nProject Completed")