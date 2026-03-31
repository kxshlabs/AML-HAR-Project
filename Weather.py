import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

print("Weather Prediction using Hidden Markov Model")

# Load dataset
data = pd.read_csv("BeijingPM2.5.csv")

# Select features
data = data[['TEMP', 'DEWP', 'PRES', 'Iws']]
data = data.dropna()

# Feature Engineering (Moving Average)
data['TEMP_MA'] = data['TEMP'].rolling(3).mean()
data['DEWP_MA'] = data['DEWP'].rolling(3).mean()
data['PRES_MA'] = data['PRES'].rolling(3).mean()
data['Iws_MA'] = data['Iws'].rolling(3).mean()
data = data.dropna()

# Better Weather Labels
def weather_label(row):
    if row['TEMP'] < 5 and row['PRES'] > 1020:
        return 0  # Cold
    elif row['TEMP'] > 25 and row['DEWP'] > 10:
        return 1  # Hot Humid
    elif row['Iws'] > 30:
        return 2  # Windy
    else:
        return 3  # Mild

data['WeatherLabel'] = data.apply(weather_label, axis=1)

# Features
X = data[['TEMP','DEWP','PRES','Iws','TEMP_MA','DEWP_MA','PRES_MA','Iws_MA']].values
y = data['WeatherLabel'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Train HMM
model = hmm.GaussianHMM(
    n_components=5,
    covariance_type="full",
    n_iter=400
)
model.fit(X_train)

# Predict hidden states
train_states = model.predict(X_train)
test_states = model.predict(X_test)

# Map states to labels
state_mapping = {}
for i in range(5):
    mask = (train_states == i)
    if np.sum(mask) > 0:
        state_mapping[i] = np.bincount(y_train[mask]).argmax()

y_pred = np.array([state_mapping[s] for s in test_states])

# PERFORMANCE METRICS
print("\n===== PERFORMANCE METRICS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

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

# Hidden State Sequence
plt.figure()
plt.plot(test_states[:200])
plt.title("Hidden State Sequence")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()

# Temperature Time Series
plt.figure()
plt.plot(data['TEMP'][:200])
plt.title("Temperature Time Series")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()

# State Distribution
plt.figure()
sns.countplot(x=test_states)
plt.title("Hidden State Distribution")
plt.xlabel("State")
plt.ylabel("Count")
plt.show()

# Log Likelihood
print("\nLog Likelihood:", model.score(X_train))

print("\nProject Completed")