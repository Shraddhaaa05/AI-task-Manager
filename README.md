# 🧠 AI Task Management System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-task-manager.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-8B5CF6?style=for-the-badge)](https://scikit-learn.org/stable/modules/feature_extraction.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

---

Engineers waste a lot of time manually triaging tickets. Every Jira task that comes in needs a priority, an assignee, a deadline flag, a status, and a project type — all assigned by hand from a plain-English description. It is repetitive, inconsistent, and slow.

This project automates that triage step using a Multi-Output machine learning model trained on real Jira data. Type a task description and get nine predictions instantly — all at the same time from a single model.

## 🔗 Live Demo

**Try it here → [ai-task-manager.streamlit.app](https://ai-task-manager1.streamlit.app/)**

Type any task description in plain English and the system predicts all nine attributes at once.

---

## ✨ Features

- **Nine simultaneous predictions** — priority, assignee, deadline flag, status, project type, and four additional task attributes all predicted from one input
- **TF-IDF text vectorisation** — converts plain-English task descriptions into a 5,000-feature numeric representation the model can work with
- **Multi-Output Logistic Regression** — one model predicts all nine labels at the same time using Scikit-learn's `MultiOutputClassifier`
- **Per-label evaluation** — accuracy, precision, recall, and F1 measured separately for each of the nine output labels, not just an average
- **Confusion matrix visualisation** — see exactly where the model is confident and where it struggles for each label
- **Real Jira dataset** — trained on actual project management data, not synthetic examples
- **Clean Streamlit interface** — paste a task, click predict, see all nine results instantly

---

## ⚙️ How It Works

The system follows a straightforward ML pipeline from raw text to nine predictions.

**Step 1 — Text input**
The user types a task description in plain English. For example: "Set up CI/CD pipeline for the staging environment and configure automated tests."

**Step 2 — Preprocessing** (`data/preprocessing.py`)
The raw text is lowercased, stripped of punctuation, and stop words are removed. This cleans the input before vectorisation.

**Step 3 — TF-IDF Vectorisation**
The cleaned text is converted into a numeric vector using TF-IDF with 5,000 features. TF-IDF gives higher weight to words that are important for this specific task description but are not generic across all descriptions. This is what lets the model distinguish "deploy to AWS" from "write unit tests" even though both are engineering tasks.

**Step 4 — Multi-Output Classification** (`model/predict.py`)
The TF-IDF vector is passed into a `MultiOutputClassifier` wrapping Logistic Regression. This runs one Logistic Regression classifier per label internally — so nine classifiers, all trained on the same feature set. Each one predicts its label independently, and all nine results are returned together.

**Step 5 — Output display**
The Streamlit UI displays all nine predictions with confidence indicators. The evaluation tab shows per-label accuracy, F1, and confusion matrices from the test set.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Scikit-learn | TF-IDF vectorisation, MultiOutputClassifier, Logistic Regression |
| Pandas | Data loading, cleaning, and preprocessing |
| NumPy | Numeric operations |
| Streamlit | Web interface |
| Joblib | Saving and loading the trained model |
| Matplotlib / Seaborn | Confusion matrix visualisation |

---

## 📊 Model Performance

Evaluated per label on a held-out test set (20% of the Jira dataset).

| Output Label | Accuracy | F1 Score |
|---|---|---|
| Priority | [add your value] | [add your value] |
| Assignee | [add your value] | [add your value] |
| Deadline Flag | [add your value] | [add your value] |
| Status | [add your value] | [add your value] |
| Project Type | [add your value] | [add your value] |
| Label 6 | [add your value] | [add your value] |
| Label 7 | [add your value] | [add your value] |
| Label 8 | [add your value] | [add your value] |
| Label 9 | [add your value] | [add your value] |

> Run `python model/train.py` after setup to see your actual numbers. Then fill them in above.

**Why this project matters technically:**
Most ML tutorials show single-label classification — one input, one output. Predicting nine labels simultaneously from the same text input is a real production ML pattern used in ticket triaging systems at companies like Atlassian and ServiceNow. This project demonstrates that multi-output architecture end to end.

---

## 📁 Project Structure

```
Ai-Task-Manager/
│
├── app.py                        # Streamlit entry point
│
├── model/
│   ├── train.py                  # Training script — run this first
│   ├── predict.py                # Inference — loads the saved model
│   └── task_classifier.pkl       # Saved trained model (generated by train.py)
│
├── data/
│   ├── jira_dataset.csv          # Training data (real Jira export)
│   └── preprocessing.py         # Text cleaning pipeline
│
├── evaluation/
│   └── metrics.py                # Per-label precision, recall, F1, confusion matrix
│
├── notebooks/
│   └── EDA_and_Training.ipynb    # Exploratory analysis and model training walkthrough
│
├── docs/
│   └── screenshots/
│       ├── input_interface.png   # Add your screenshot here
│       ├── prediction_output.png # Add your screenshot here
│       └── evaluation_metrics.png # Add your screenshot here
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

**Clone the repository**

```bash
git clone https://github.com/shraddha-gidde/Ai-Task-Manager.git
cd Ai-Task-Manager
```

**Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Before running the app, train the model on the dataset:

```bash
python model/train.py
```

This reads `data/jira_dataset.csv`, fits the TF-IDF vectoriser and MultiOutputClassifier, and saves `task_classifier.pkl`. It also prints per-label evaluation metrics so you can fill in the table above.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**How to use it:**
1. Type any task description in the input box
2. Click **Predict**
3. See all nine task attributes predicted instantly
4. Switch to the **Evaluation** tab to see per-label metrics and confusion matrices

---

## 🔮 Future Improvements

- [ ] Fill in actual per-label metric values from training output
- [ ] Add confusion matrix screenshots to the repository
- [ ] Try Random Forest and XGBoost and compare against Logistic Regression baseline
- [ ] Add bigrams to the TF-IDF vocabulary to capture phrases like "high priority" as a single token
- [ ] Add class weighting for any labels that are imbalanced in the dataset
- [ ] Support uploading a CSV of tasks for batch prediction

---

## 👩‍💻 Author

**Shraddha Gidde**
B.Tech — Artificial Intelligence and Data Science
MIT World Peace University, Pune

[![Portfolio](https://img.shields.io/badge/Portfolio-shraddha--gidde.netlify.app-2563EB?style=flat-square)](https://shraddha-gidde.netlify.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/shraddha-gidde-063506242)
[![GitHub](https://img.shields.io/badge/GitHub-shraddha--gidde-181717?style=flat-square&logo=github)](https://github.com/Shraddhaaa05)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
