# Dataset: https://www.kaggle.com/datasets/hugomathien/soccer/

# From description:
# The bookies use 3 classes (Home Win, Draw, Away Win). They get it right about 53% of the time.
# Though it may sound high for such a random sport game, the home team wins about 46% of the time.
# So the base case (constantly predicting Home Win) has indeed 46% precision.

# =====

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG_SEED = 1
FILENAME = "global-outcomes.csv"

# Load your dataset
try:
    data = pd.read_csv(FILENAME)
except FileNotFoundError:
    from db_to_csv import create_csv  # local file (db_to_csv.py)

    create_csv(FILENAME)
    data = pd.read_csv(FILENAME)

# Convert the 'date' column to datetime format
data["date"] = pd.to_datetime(data["date"])

# Extract features from the date: year, month, and day of the week
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek

# Map 'outcome' to numerical values
data["outcome"] = data["outcome"].map({"H": -1, "D": 0, "A": 1})

# Select features (X)
X = data[
    ["home_team_api_id", "away_team_api_id", "year", "month", "day_of_week"]
]  # Features
y = data["outcome"]  # Target variable (the outcome)

# Split the data into training and testing sets (Example: 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RNG_SEED,
)

# Create a preprocessor for OneHotEncoding the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(),
            [
                "home_team_api_id",
                "away_team_api_id",
            ],  # OneHotEncode only categorical features
        ),
        (
            "num",
            "passthrough",  # Pass through numeric features
            ["year", "month", "day_of_week"],
        ),
    ]
)

# Build the pipeline
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),  # Preprocess features
        (
            "scaler",
            StandardScaler(with_mean=False),  # Scale features
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(100, 100), max_iter=600, random_state=RNG_SEED
            ),  # Neural network
        ),
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display a classification report
print(
    classification_report(y_test, y_pred, target_names=["Home Win", "Away Win", "Draw"])
)

# Output (RNG_SEED = 1)
# Accuracy: 0.42
#               precision    recall  f1-score   support

#     Home Win       0.53      0.54      0.53      2403
#     Away Win       0.27      0.29      0.28      1275
#         Draw       0.37      0.35      0.36      1518

#     accuracy                           0.42      5196
#    macro avg       0.39      0.39      0.39      5196
# weighted avg       0.42      0.42      0.42      5196
