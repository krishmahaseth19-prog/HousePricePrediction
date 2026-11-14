
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, X_test, y_test):

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Results")
    print("------------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Create prediction errors
    errors = np.array(y_test) - np.array(y_pred)

    # Save prediction results
    results = pd.DataFrame({
        "actual": np.array(y_test),
        "predicted": np.array(y_pred),
        "error": errors
    })

    results.to_csv("prediction_errors.csv", index=False)

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # 1. Prediction Error Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")

    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/error_distribution.png")
    plt.close()

    # 2. Residual Error Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")

    plt.title("Residual Error Plot")
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Actual - Predicted ($)")
    plt.tight_layout()
    plt.savefig("outputs/residual_plot.png")
    plt.close()

    # 3. Actual vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)

    min_value = min(np.min(y_test), np.min(y_pred))
    max_value = max(np.max(y_test), np.max(y_pred))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r--"
    )

    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.close()

    print("\nAll three error analysis graphs saved in outputs/")

    return mae, mse, rmse, r2
