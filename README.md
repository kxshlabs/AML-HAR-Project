# 🏃 Human Activity Recognition using HMMs (AML-HAR-Project)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Hidden_Markov_Models-00f3ff?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

> A probabilistic sequential machine learning project for classifying Human Activity Recognition (HAR) sensor data using Hidden Markov Models (HMM).

---

## 📌 Overview

The **AML HAR Project** applies sequence-based probabilistic modeling to identify human activities (walking, sitting, standing, etc.) using tri-axial accelerometer and gyroscope data collected from smartphone inertial sensors. 

By utilizing **Hidden Markov Models (HMM)**, the system models the hidden temporal state transitions of physical movement across continuous sensor signals.

---

## 🧠 Model & Approach

```
Inertial Sensor Data (Accelerometer & Gyroscope)
                    │
                    ▼
          Feature Extraction & Windowing
                    │
                    ▼
       Hidden Markov Model (HMM) Decoder
                    │
                    ▼
     Predicted Activity State (Walking / Standing / etc.)
```

---

## 🗂️ Project Structure

```
AML HAR Project/
├── har_hmm.py         # Main script for loading dataset & evaluating HMM
├── X_train.txt        # Training features (Inertial sensor signals)
├── y_train.txt        # Training labels (Activity IDs)
├── X_test.txt         # Testing features
├── y_test.txt         # Testing labels
└── README.md          # Project documentation
```

---

## ⚙️ Setup & Execution

### 1. Clone the repository
```bash
git clone https://github.com/kxshlabs/AML-HAR-Project.git
cd AML-HAR-Project
```

### 2. Install requirements
```bash
pip install numpy scikit-learn hmmlearn
```

### 3. Run the model
```bash
python har_hmm.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Model** | Hidden Markov Models (`hmmlearn`) |
| **Data Processing** | NumPy, Scikit-Learn |
| **Domain** | Signal Processing / Human Activity Recognition |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
