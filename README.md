# Transparent-Fetal Net: Explainable Health Classification with Active Data Sampling

## Project Overview

This project presents an intelligent fetal health classification system using Cardiotocography (CTG) data integrated with Active Learning and Explainable Artificial Intelligence (XAI).

The system analyzes fetal health conditions and classifies them into:

- Normal
- Suspect
- Pathological

The project aims to improve fetal health monitoring by reducing dependency on large labeled datasets while providing transparent and interpretable predictions for clinical decision-making.

---

## Features

- CTG-based fetal health prediction
- Active Learning framework
- Explainable AI (XAI) integration
- Real-time prediction interface
- Clinical risk assessment
- SHAP-based feature importance analysis
- Dashboard for monitoring model performance
- Report generation and storage

---

## Technologies Used

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- HTML
- CSS
- JavaScript

---

## Dataset

The project uses the UCI Cardiotocography (CTG) dataset containing 21 clinical features for fetal health analysis.

Dataset Classes:
- Normal
- Suspect
- Pathological

---

## Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── Procfile
├── README.md
│
├── data/
│   └── fetal_health.csv
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── reports.html
│   └── ctg_analysis.html
│
├── static/
│   ├── styles.css
│   ├── script.js
│   └── images/
│
├── labeled_X.npy
├── labeled_y.npy
├── pool_X.npy
├── pool_y.npy
└── fetal_health_model.pkl
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sreeja-Pasunuti-dev/transparent-Fetal-net.git
```

Move into the project directory:

```bash
cd transparent-Fetal-net
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

---

## Deployment

The project is deployed using Render.

Live Project URL:
(Add your Render deployment link here)

---

## Machine Learning Workflow

1. Data Preprocessing
2. Active Learning Sample Selection
3. Model Training using XGBoost
4. Prediction Generation
5. Explainable AI Analysis using SHAP
6. Clinical Risk Assessment

---

## Explainable AI

The project uses SHAP (SHapley Additive Explanations) to provide transparent and interpretable predictions.

This helps:
- identify important clinical features
- improve trust in predictions
- support medical decision-making

---

## Future Enhancements

- Real-time CTG monitoring
- Deep learning integration
- Mobile responsive dashboard
- Cloud database integration
- PDF report generation
- User authentication system

---

## Author

Sreeja Pasunuti

---

## License

This project is developed for educational and research purposes.
