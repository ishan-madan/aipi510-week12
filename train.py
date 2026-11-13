from pathlib import Path
import json

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # load dataset
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    # split on proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # sclar features and train logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=dataset.target_names, output_dict=True
    )

    results = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "model": "StandardScaler + LogisticRegression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_samples": int(len(y)),
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=dataset.target_names
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Logistic Regression - Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"Test accuracy: {accuracy:.2%}")
    print("Confusion matrix:")
    print(matrix)
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=dataset.target_names))
    print(f"\nSaved metrics and plot to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
