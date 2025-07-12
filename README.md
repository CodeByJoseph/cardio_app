# Cardio_app

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Cardio_app** is a Python-based machine learning project for cardiovascular disease prediction.  
It uses a trained model on medical data and provides interactive pages for exploring the dataset, evaluating the model, and understanding the project.

> ⚠️ *This application is intended for educational/demo purposes only. It does not store user data or implement authentication.*

---

## 📁 Project Structure

Cardio_app/
│
├── main.py                 # Main app runner
├── train_model.py          # Script for training the ML model
├── cardio_train.csv        # Dataset used for training
├── best_model.pkl          # Trained model saved file
│
├── pages/                  # Pages for the app (if using multi-page framework)
│   ├── about.py            # About page
│   ├── data_browser.py     # Data exploration page
│   └── model_evaluation.py # Model evaluation and metrics page
│

---

## 🚀 Features

- Cleans and preprocesses cardiovascular health data
- Removes outliers (e.g. abnormal BMI or blood pressure values)
- Trains a machine learning model (e.g., **XGBoostClassifier**)
- Encodes and scales data via a **pipeline**
- Offers interactive pages to:
  - Browse and filter the dataset
  - Evaluate model metrics and visualize performance
  - Read about the project and context

---

## 🛠 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Cardio_app.git
cd Cardio_app

```

2. (Optional) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

---

# ▶️ Usage
1. To train the model from scratch:
```bash
python train_model.py
```

2. To run the app:
```bash
streamlit run main.py
```

## 📦 Model Versioning

The trained model (best_model.pkl) is included in the repository for convenience.
For larger projects or production use, consider:

Using Git Large File Storage (LFS)

External model registries (e.g., MLflow)

Cloud storage (e.g., S3, Google Cloud Storage)

---

## 📋 License

This project is licensed under the MIT License.
Let me know if you'd like a short `requirements.txt` snippet or want to auto-generate one from your environment (`pip freeze` filtered).






