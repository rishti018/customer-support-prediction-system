import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. Load dataset
FILE_PATH = "data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"

df = pd.read_csv(FILE_PATH)

print("Original dataset shape:", df.shape)

# 2. Select required columns
required_columns = ["subject", "body", "queue", "priority", "language"]

df = df[required_columns].copy()

# 3. Keep only English tickets
df = df[df["language"].astype(str).str.lower() == "en"].copy()

# 4. Handle missing values
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

df = df.dropna(subset=["queue", "priority"])

# 5. Combine subject and body
df["ticket_text"] = df["subject"] + " " + df["body"]

# Remove empty tickets
df = df[df["ticket_text"].str.strip() != ""]

print("Usable English tickets:", len(df))
print("Number of categories:", df["queue"].nunique())
print("Priority labels:", df["priority"].unique())

# 6. Prepare input and output
X = df["ticket_text"]

targets = {
    "category": df["queue"],
    "priority": df["priority"]
}

# 7. Define classification algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}

# 8. Define parameters for GridSearchCV
parameter_grid = {
    "Logistic Regression": {
        "classifier__C": [0.5, 1.0, 2.0]
    },

    "Naive Bayes": {
        "classifier__alpha": [0.5, 1.0, 2.0]
    },

    "Decision Tree": {
        "classifier__max_depth": [None, 20, 40]
    },

    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 30]
    },

    "Linear SVM": {
        "classifier__C": [0.5, 1.0, 2.0]
    }
}

# 9. Create models folder
os.makedirs("models", exist_ok=True)

# 10. Train and evaluate both prediction models
for target_name, y in targets.items():

    print("\n" + "=" * 60)
    print("TRAINING:", target_name.upper())
    print("=" * 60)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    best_score = 0
    best_model = None
    best_model_name = ""

    # Compare all algorithms
    for model_name, classifier in models.items():

        print("\nTraining:", model_name)

        # Create TF-IDF and classification pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2
            )),
            ("classifier", classifier)
        ])

        # Apply GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            parameter_grid[model_name],
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # Predict test data
        predictions = grid_search.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        print("Best Parameters:", grid_search.best_params_)
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print("Macro F1 Score:", round(macro_f1 * 100, 2), "%")

        # Select best model using macro F1
        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    # Display best model
    print("\nBest Model:", best_model_name)
    print("Best Macro F1 Score:", round(best_score * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        zero_division=0
    ))

    # Generate confusion matrix
    predictions = best_model.predict(X_test)
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=best_model.classes_
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_model.classes_,
        yticklabels=best_model.classes_
    )

    plt.title(f"Confusion Matrix - {target_name.title()}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(f"models/{target_name}_confusion_matrix.png")
    plt.close()

    print(f"Confusion matrix saved for {target_name}")

    # Save complete pipeline using Pickle
    model_path = f"models/{target_name}_model.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    print("Model saved at:", model_path)

print("\nAll models trained and saved successfully!")