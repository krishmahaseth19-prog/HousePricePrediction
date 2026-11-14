
from src.preprocess import (
    load_data,
    preprocess_data,
    create_preprocessor
)

from src.train import train_model
from src.evaluate import evaluate_model

from src.save_model import (
    save_pickle,
    save_joblib
)


print("\nHOUSE PRICE PREDICTION PROJECT")
print("================================")

# 1. Load dataset
print("\nLoading dataset...")

df = load_data("data/data.csv")

# 2. Preprocess dataset
print("Preprocessing dataset...")

df = preprocess_data(df)

# 3. Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# 4. Create preprocessing pipeline
preprocessor = create_preprocessor(X)

# 5. Compare and train models
print("\nTraining models...")

model, X_test, y_test = train_model(
    X,
    y,
    preprocessor
)

# 6. Display selected model
print("\nSELECTED MODEL:")
print(model.named_steps["regressor"])

# 7. Evaluate selected model on untouched test data
evaluate_model(
    model,
    X_test,
    y_test
)

# 8. Save final model
save_pickle(model)
save_joblib(model)

print("\n================================")
print("PROJECT EXECUTION COMPLETED")
print("================================")
