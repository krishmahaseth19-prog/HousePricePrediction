# House Price Prediction

<p align="center">
  <strong>Machine Learning-Based House Price Prediction System</strong>
  <br/>
  An end-to-end machine learning application with an interactive dashboard and REST API.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

---

## 1. Project Overview

House Price Prediction is an end-to-end machine learning project designed to estimate residential property prices based on relevant property characteristics.

The system uses an **Extra Trees Regressor** to generate price predictions. It combines data preprocessing, model training, performance evaluation, and deployment through FastAPI and Streamlit.

The application provides an interactive interface where users can enter property details and obtain estimated house prices.

## 2. Key Features

- **Machine Learning:** House price prediction using Extra Trees Regressor.
- **Data Preprocessing:** Data cleaning, feature engineering, and missing-value handling.
- **Model Evaluation:** Performance measurement using MAE, MSE, RMSE, and R².
- **REST API:** Prediction endpoints developed using FastAPI.
- **Interactive Dashboard:** User-friendly prediction interface built with Streamlit.
- **Data Visualization:** Interactive charts for exploring model performance and prediction results.
- **Model Persistence:** Saving trained models using Pickle and Joblib.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model | Extra Trees Regressor |
| API Development | FastAPI |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Model Serialization | Pickle, Joblib |
| Development Environment | VS Code, Git, GitHub |

## 4. Project Architecture

```text
HousePricePrediction/
│
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── schema.py
│
├── data/
│   └── data.csv
│
├── models/
│   └── (Generated model files)
│
├── outputs/
│   └── (Generated evaluation visualizations)
│
├── src/
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── save_model.py
│   ├── streamlit_app.py
│   └── train.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Model Performance

The project uses an **Extra Trees Regressor** configured with 700 estimators.

The following results represent the current baseline model evaluation:

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 77,440.97 |
| Mean Squared Error (MSE) | 14,442,493,078.53 |
| Root Mean Squared Error (RMSE) | 120,176.92 |
| R² Score | 0.7085 |

**Model Configuration:**
- Algorithm: Extra Trees Regressor
- Number of Estimators: 700
- Maximum Features: `sqrt`
- Random State: 42
- Parallel Processing: Enabled (`n_jobs=-1`)

## 6. Installation and Setup

Follow these steps to run the project locally.

### Step 1: Clone the repository

```bash
git clone https://github.com/krishmahaseth19-prog/HousePricePrediction.git
```

### Step 2: Navigate to the project directory

```bash
cd HousePricePrediction
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

## 7. Model Training

The trained model files are not included in the GitHub repository because of their size.

Generate the models locally by executing:

```bash
python main.py
```

This runs the project pipeline, including:

1. Loading the dataset.
2. Data preprocessing and feature engineering.
3. Model training.
4. Model evaluation.
5. Saving the trained model files.

Ensure that model generation completes successfully before starting the API.

## 8. Running the Application

The project uses FastAPI as the backend and Streamlit as the frontend.

### Start the FastAPI server

```bash
uvicorn api.app:app --reload
```

API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Start the Streamlit dashboard

Open a second terminal, activate the virtual environment, and execute:

```bash
streamlit run src/streamlit_app.py
```

Open the dashboard:

[http://localhost:8501](http://localhost:8501)

## 9. Model Evaluation and Visualization

The evaluation pipeline generates visualizations to help analyze model performance.

The project includes:

- Actual vs. Predicted Price Plot
- Residual Plot
- Prediction Error Distribution

These visualizations help assess prediction accuracy, error distribution, and model behavior.

## 10. Repository Notes

- The dataset is located in the `data/` directory.
- Trained `.pkl` and `.joblib` files are excluded from GitHub because they exceed GitHub's individual file size limit.
- Run `python main.py` to generate the trained models locally.
- Generated evaluation outputs are excluded from version control.

## 11. Future Improvements

- Experiment with additional regression algorithms.
- Perform hyperparameter optimization.
- Improve prediction accuracy through feature engineering.
- Add model comparison and selection.
- Enhance dashboard functionality and user experience.
- Deploy the application to a cloud platform.

## 12. Author

**Krishma Mahaseth**

GitHub: [krishmahaseth19-prog](https://github.com/krishmahaseth19-prog)

---

<p align="center">
  <strong>House Price Prediction</strong>
  <br/>
  Built with Python, Machine Learning, FastAPI, and Streamlit.
</p>