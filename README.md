# ✅ AI-Powered Task Manager

> An intelligent task management system that automatically classifies and prioritizes tasks by deadline and workload using NLP and machine learning — built with a Streamlit interface for real-time use.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20TF--IDF-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 What It Does

Manual task prioritization is time-consuming and inconsistent. This system automates it by:

- Classifying tasks by type and urgency using NLP
- Prioritizing tasks intelligently based on deadlines and workload
- Providing a clean, interactive interface to manage and track tasks in real time

---

## ✨ Features

- 🧠 **Automatic Task Classification** — TF-IDF vectorization + Logistic Regression to categorize tasks
- 📅 **Smart Prioritization** — ranks tasks by deadline, complexity, and workload
- ⚡ **Low-Latency ML Pipeline** — optimized for fast inference without sacrificing accuracy
- 📊 **Multi-Output Prediction** — predicts both task category and priority level simultaneously
- 🖥️ **Streamlit Interface** — interactive UI to add, view, and manage tasks
- 💾 **Pre-trained Model** — includes saved `.pkl` model files for instant deployment

---

## 🏗️ How It Works

```
Task Input (title + description)
        ↓
  TF-IDF Vectorizer
        ↓
  Logistic Regression Model
        ↓
  ┌─────────────────────────┐
  │ Task Category           │  (e.g. Bug, Feature, Research)
  │ Priority Level          │  (e.g. High, Medium, Low)
  └─────────────────────────┘
        ↓
  Sorted Task List (by priority + deadline)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| NLP | TF-IDF Vectorization |
| ML | Scikit-learn, Logistic Regression, Multi-output Classification |
| UI | Streamlit |
| Dataset | Jira task dataset (CSV) |
| Model Persistence | Pickle (.pkl) |

---

## ⚙️ Getting Started

### Installation
```bash
# Clone the repo
git clone https://github.com/Shraddhaaa05/AI-task-manager-.git
cd AI-task-manager-

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
├── streamlit_app.py           # Streamlit UI
├── tarin_model.py             # Model training script
├── som_projectfile.ipynb      # Full ML notebook (EDA + training)
├── model.pkl                  # Trained classifier
├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
├── multi_output_labels.pkl    # Label encoders
├── jira_dataset.csv           # Training dataset
└── requirements.txt
```

---

## 📊 Dataset

Trained on a Jira-style task dataset containing task titles, descriptions, categories, and priority labels — simulating real-world project management workflows.

---

## 🔍 Results

- Achieved strong multi-output classification accuracy for task type and priority
- Optimized ML pipeline for low latency while maintaining performance
- Applied structured data handling and sorting logic for efficient task ranking

---

## 👩‍💻 Author

**Shraddha Gidde**
- 🔗 [LinkedIn](https://www.linkedin.com/in/shraddha-gidde-063506242/)
- 💻 [GitHub](https://github.com/Shraddhaaa05)
