import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG_SEED = 42

# Load the dataset
data = pd.read_csv("breast-cancer.csv")

# Map the target variable 'class' to binary values
data["class"] = data["class"].map(
    {"recurrence-events": 1, "false-recurrence-events": 0}
)

# Separate features and labels
X = data.drop(columns=["class"])  # Drop the target column to get features
y = data["class"]  # Target column contains the labels

# Identify categorical columns based on your dataset
categorical_columns = [
    "age",
    "mefalsepause",
    "tumor-size",
    "inv-falsedes",
    "falsede-caps",
    "breast",
    "breast-quad",
    "irradiat",
]

# Create a preprocessor for OneHotEncoding the categorical columns
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_columns)],
    remainder="passthrough",
)  # 'passthrough' keeps the rest of the columns (e.g., deg-malig)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG_SEED
)

# Build the pipeline
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),  # OneHotEncode categorical features
        (
            "scaler",
            StandardScaler(with_mean=False),
        ),  # Scale the features without centering (due to sparse matrix)
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(100, 100), max_iter=300, random_state=RNG_SEED
            ),
        ),  # Neural network
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
print(classification_report(y_test, y_pred))

# Output (RNG_SEED = 42):
# Accuracy: 0.76
#               precision    recall  f1-score   support

#            0       0.82      0.88      0.85        41
#            1       0.55      0.43      0.48        14

#     accuracy                           0.76        55
#    macro avg       0.68      0.65      0.66        55
# weighted avg       0.75      0.76      0.75        55


# ===== Chatgpt interpretation of results =====

# The classification report provides several metrics to evaluate the performance of your model. Let's break down each part of the report:

# ### Key Metrics

# 1. **Precision**: Precision is the ratio of correctly predicted positive observations to the total predicted positives. It tells you how many of the predictions made as "positive" (in your case, class `1`, which likely refers to "recurrence-events") were actually correct.

#    \[
#    \text{Precision} = \frac{TP}{TP + FP}
#    \]
#    - For class `0`: Precision = 0.82, meaning 82% of the time the model's predictions of class `0` (no-recurrence-events) were correct.
#    - For class `1`: Precision = 0.55, meaning 55% of the time the model's predictions of class `1` (recurrence-events) were correct.

# 2. **Recall**: Recall (also called **sensitivity** or **true positive rate**) is the ratio of correctly predicted positive observations to all actual positives. It answers: "Out of all the true instances of class `1` (recurrence-events), how many did the model correctly identify?"

#    \[
#    \text{Recall} = \frac{TP}{TP + FN}
#    \]
#    - For class `0`: Recall = 0.88, meaning the model correctly identified 88% of all actual class `0` instances.
#    - For class `1`: Recall = 0.43, meaning the model correctly identified only 43% of all actual class `1` instances (it missed 57%).

# 3. **F1-Score**: The F1-score is the harmonic mean of precision and recall. It is a balance between precision and recall and is useful when the dataset is imbalanced.

#    \[
#    \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
#    \]
#    - For class `0`: F1-score = 0.85, which indicates a good balance between precision and recall for predicting class `0`.
#    - For class `1`: F1-score = 0.48, which is relatively low, reflecting that the model struggles with accurately predicting class `1` (recurrence-events).

# 4. **Support**: This is the number of actual occurrences of each class in the test set. In your case:
#    - Support for class `0` (no-recurrence-events) = 41.
#    - Support for class `1` (recurrence-events) = 14.

#    The model had to make predictions for 41 samples of class `0` and 14 samples of class `1`.

# ### Overall Metrics:

# - **Accuracy**: Accuracy is the overall percentage of correct predictions (both classes).

#   \[
#   \text{Accuracy} = \frac{TP + TN}{Total}
#   \]
#   - Accuracy = 0.76, meaning the model correctly predicted 76% of the samples.

# - **Macro Average**: The **macro average** is the unweighted average of the precision, recall, and F1-score across all classes. It treats all classes equally, regardless of support. In your case, macro averages are:
#   - Precision = 0.68
#   - Recall = 0.65
#   - F1-score = 0.66

#   Since class `1` has fewer samples, this metric helps give an idea of performance that doesn't depend on class imbalance.

# - **Weighted Average**: The **weighted average** takes into account the support (number of samples) for each class. It adjusts the metrics (precision, recall, F1-score) based on how many samples are present in each class. In your case, the weighted averages are:
#   - Precision = 0.75
#   - Recall = 0.76
#   - F1-score = 0.75

#   These are more representative of the overall performance, taking into account the class imbalance.

# ---

# ### Interpretation:

# - **Class Imbalance**: The dataset is imbalanced (41 samples for class `0`, only 14 for class `1`), and the model performs better on the majority class (`0`) than the minority class (`1`). This is indicated by the much lower precision, recall, and F1-score for class `1` (recurrence-events).

# - **Precision vs. Recall**: For class `1` (recurrence-events), the model has a decent precision (55%) but low recall (43%). This means that while it doesn't misclassify too many non-recurrence-events as recurrence-events, it misses a lot of actual recurrence-events.

# - **F1-Score for Class 1**: The low F1-score (0.48) for class `1` indicates that the model struggles to find a good balance between precision and recall for this class, likely due to the small number of samples.
