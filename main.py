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


# Load data
df = load_data("data/data.csv")

# Preprocess data
df = preprocess_data(df)

# Features and target
X = df.drop("price", axis=1)

y = df["price"]

# Create transformer
preprocessor = create_preprocessor(X)

# Train model
model, X_test, y_test = train_model(
    X,
    y,
    preprocessor
)

# Evaluate model
evaluate_model(
    model,
    X_test,
    y_test
)

# Save model
save_pickle(model)

save_joblib(model)