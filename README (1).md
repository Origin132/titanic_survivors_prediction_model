# 🚢 Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster using **Logistic Regression**, based on demographic and ticket information.

---

## 📌 Project Overview

April 15, 1912. The RMS Titanic sank into the North Atlantic Ocean, taking 1,517 lives with it. Over a century later, this tragedy became one of the most studied datasets in machine learning history. This project builds a classification pipeline to predict passenger survival based on features such as class, sex, age, and fare.

| Item | Detail |
|---|---|
| **Model** | Logistic Regression |
| **Target Variable** | Survived (0 = No, 1 = Yes) |
| **Training Set** | 891 rows × 12 columns |
| **Test Set** | 418 rows × 11 columns |
| **Accuracy** | 81.56% |
| **F1 Score (Survived)** | 0.74 |

---

## 📂 Project Structure

```
titanic-survival-prediction/
│
├── titanic_train.csv               # Training dataset (with Survived column)
├── titanic_test.csv                # Test dataset (without Survived column)
├── Titanic_Model.ipynb             # Full notebook (EDA + training + evaluation)
├── titanic_survival_model.pkl      # Saved trained model
├── titanic_scaler.pkl              # Saved StandardScaler
├── predict.py                      # Script to make new predictions
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🧾 Dataset Features

| Column | Description | Used |
|---|---|---|
| `PassengerId` | Unique passenger identifier | ❌ Dropped |
| `Survived` | Target — 0 = No, 1 = Yes | ✅ Target |
| `Pclass` | Passenger class (1, 2, 3) | ✅ |
| `Name` | Passenger name | ❌ Dropped |
| `Sex` | Gender (encoded) | ✅ |
| `Age` | Age in years | ✅ |
| `SibSp` | Siblings/spouses aboard | ✅ |
| `Parch` | Parents/children aboard | ✅ |
| `Ticket` | Ticket number | ❌ Dropped |
| `Fare` | Passenger fare | ✅ |
| `Cabin` | Cabin number | ❌ Dropped |
| `Embarked` | Port of embarkation (encoded) | ✅ |

---

## ⚙️ Data Preprocessing

- Dropped useless columns: `Name`, `Ticket`, `Cabin`, `PassengerId`
- **Sex** encoded using Label Encoding (`male=1`, `female=0`)
- **Embarked** encoded using Label Encoding (`S=2`, `C=0`, `Q=1`)
- Missing `Age` values filled with median
- Missing `Fare` values filled with median
- Feature scaling applied using **StandardScaler**

---

## 📊 Model Evaluation

| Metric | Class 0 (Did Not Survive) | Class 1 (Survived) |
|---|---|---|
| Precision | 0.82 | 0.80 |
| Recall | 0.89 | 0.70 |
| F1 Score | 0.86 | 0.74 |
| **Accuracy** | **81.56%** | |

**Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 98 ✅ | 12 ❌ |
| **Actual 1** | 21 ❌ | 48 ✅ |

---

## 🔍 Key Insights from EDA

- **Sex** is the strongest single predictor (correlation: -0.54 with Survived)
- **Pclass** is the second strongest predictor (correlation: -0.34)
- **Fare** positively correlates with survival (+0.26) — wealth bought access to lifeboats
- **SibSp** and **Parch** show near-zero correlation with survival

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Origin132/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `Titanic_Model.ipynb` in Jupyter or Google Colab and run all cells.

### 4. Make a new prediction
```bash
python predict.py
```

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 👤 Author

**Sambo Bashir**
Aspiring AI/ML Engineer | Data Science Practitioner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/sambo-bashir-3a9648185/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Origin132)

Update author links in README
