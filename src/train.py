
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone


def train_model(X, y, preprocessor):

    print("\n========== EXTRA TREES MODEL TRAINING ==========")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Previously selected best Extra Trees parameters
    model = Pipeline([
        ("preprocessor", clone(preprocessor)),

        ("regressor", ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nTraining Extra Trees model...")
    print("Please wait...")

    # Train model
    model.fit(X_train, y_train)

    print("\nModel training completed successfully!")

    return model, X_test, y_test
