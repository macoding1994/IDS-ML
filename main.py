import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

from config import SAMPLE_PATH

warnings.filterwarnings("ignore")


# Factory class to create models dynamically
class ModelFactory:
    def __init__(self, model_name):
        self.model_name = model_name
        self.models = {
            "DecisionTree": DecisionTreeClassifier(random_state=0),
            "RandomForest": RandomForestClassifier(random_state=0),
            "XGBoost": xgb.XGBClassifier(n_estimators=10),
            "ExtraTreesClassifier": ExtraTreesClassifier(),
        }

    def get_model(self):
        # Return the selected model instance
        if self.model_name in self.models:
            return self.models[self.model_name]
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")


# Function to train and evaluate the model
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    # Create the model using the factory
    factory = ModelFactory(model_name)
    model = factory.get_model()

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    # Print evaluation results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {fscore}")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, linewidths=0.5, cmap="Blues", fmt=".0f")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Main function for data processing and model execution
def main():
    # Load and preprocess the dataset
    df_balanced = pd.read_csv(SAMPLE_PATH)

    # Normalize numeric features
    numeric_features = df_balanced.select_dtypes(exclude=["object"]).columns
    df_balanced[numeric_features] = df_balanced[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    df = df_balanced.fillna(0)  # Replace missing values with 0

    # Encode labels into integers
    labelencoder = LabelEncoder()
    df["Label"] = labelencoder.fit_transform(df["Label"])
    # Label Mapping
    label_mapping = {index: label for index, label in enumerate(labelencoder.classes_)}
    print("Label Mapping (Integer -> Original):")
    for k, v in label_mapping.items():
        print(f"{k} -> {v}")

    # Split data into training and testing sets
    X = df.drop(["Label"], axis=1).values  # Features
    y = df["Label"].values  # Labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
    )

    # List of models to train
    models = ["DecisionTree", "RandomForest", "XGBoost", "ExtraTreesClassifier"]

    # Train and evaluate each model
    for model_name in models:
        train_and_evaluate(model_name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
