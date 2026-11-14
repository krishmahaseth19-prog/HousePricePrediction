# 🏠 House Price Prediction

<p align="center">
  <strong>Machine Learning-Based House Price Prediction System</strong><br/>
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

The system uses an **Extra Trees Regressor** to generate price predictions. It combines data preprocessing, model training, performance evaluation, and cloud deployment through FastAPI and Streamlit.

Users can enter property details through an interactive dashboard and receive estimated house prices.

## 2. Live Application

- **Streamlit Dashboard:** https://house-price-prediction-krishma.streamlit.app/
- **FastAPI Backend:** https://house-price-prediction-api-2q84.onrender.com/
- **API Documentation:** https://house-price-prediction-api-2q84.onrender.com/docs
- **GitHub Repository:** https://github.com/krishmahaseth19-prog/HousePricePrediction
- **Hugging Face Model:** https://huggingface.co/Krishmahaseth/house-price-prediction-model

## 3. Key Features

- **Machine Learning:** House price prediction using Extra Trees Regressor.
- **Data Preprocessing:** Data cleaning, feature engineering, and missing-value handling.
- **Model Evaluation:** Performance measurement using MAE, MSE, RMSE, and R².
- **REST API:** Prediction endpoints developed using FastAPI.
- **Interactive Dashboard:** User-friendly prediction interface built with Streamlit.
- **Data Visualization:** Interactive charts for exploring model performance and prediction results.
- **Model Persistence:** Saving trained models using Pickle and Joblib.
- **Cloud Deployment:** Streamlit frontend and FastAPI backend deployed online.
- **Model Hosting:** Trained model hosted on Hugging Face Hub.

## 4. Technology Stack

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
| Model Hosting | Hugging Face Hub |
| Deployment | Render, Streamlit Community Cloud |
| Development Tools | VS Code, Git, GitHub |

## 5. Project Architecture

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
│   └── Generated model files
│
├── outputs/
│   └── Generated evaluation visualizations
│
├── src/
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── save_model.py
│   ├── streamlit_app.py
│   └── train.py
│
├── main.py
├── prediction_errors.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Model Performance

The current local evaluation uses an **Extra Trees Regressor with 300 estimators**.

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 77,106.38 |
| Mean Squared Error (MSE) | 14,300,417,704.77 |
| Root Mean Squared Error (RMSE) | 119,584.35 |
| R² Score | 0.7114 |

### Model Configuration

- Algorithm: Extra Trees Regressor
- Number of Estimators: 300
- Maximum Features: `sqrt`
- Random State: 42
- Parallel Processing: `n_jobs=1`

The R² score of 0.7114 indicates that the model explains approximately 71.14% of the variance in the evaluation dataset.

## 7. Installation and Setup

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

## 8. Model Training

The trained model files are not stored in the GitHub repository because of their size.

Generate the models locally by executing:

```bash
python main.py
```

This runs the project pipeline:

1. Loading the dataset.
2. Data preprocessing and feature engineering.
3. Model training.
4. Model evaluation.
5. Saving the trained model files.
6. Generating evaluation results and visualizations.

## 9. Running the Application Locally

The application uses FastAPI as the backend and Streamlit as the frontend.

### Start the FastAPI server

```bash
uvicorn api.app:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Start the Streamlit dashboard

Open a second terminal, activate the virtual environment, and execute:

```bash
streamlit run src/streamlit_app.py
```

Open the dashboard:

http://localhost:8501

## 10. Model Evaluation and Visualization

The evaluation dashboard provides:

- Prediction Error Distribution
- Residual Error Plot
- Actual vs. Predicted Prices
- Model Evaluation Metrics
- Prediction Error Details

These visualizations help analyze prediction accuracy, error distribution, and model behavior.

## 11. Deployment Architecture

The project uses a cloud-based deployment architecture:

- **Frontend:** Streamlit Community Cloud
- **Backend:** FastAPI hosted on Render
- **Model Hosting:** Hugging Face Hub
- **Source Code:** GitHub

The Streamlit dashboard sends property information to the FastAPI prediction endpoint. The backend loads the trained model hosted on Hugging Face Hub and returns the predicted house price.

## 12. Future Improvements

- Experiment with additional regression algorithms.
- Perform hyperparameter optimization.
- Improve prediction accuracy through feature engineering.
- Add model comparison and selection.
- Enhance dashboard functionality and user experience.
- Add additional property-related features.

## 13. Author

**Krish Mahaseth**

GitHub: https://github.com/krishmahaseth19-prog

---

<p align="center">
  <strong>House Price Prediction</strong><br/>
  Built with Python, Machine Learning, FastAPI, and Streamlit.
</p>
