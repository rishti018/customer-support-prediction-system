import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "Logistic Regression",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "Linear SVM"
]

# Category prediction results
category_accuracy = [57.50, 43.82, 49.11, 71.54, 67.04]
category_f1 = [60.27, 26.58, 47.48, 73.90, 69.08]

# Priority prediction results
priority_accuracy = [64.35, 55.97, 56.79, 75.98, 68.85]
priority_f1 = [63.46, 49.44, 55.29, 75.11, 67.93]

# Function to create comparison charts
def create_chart(values, title, filename):

    plt.figure(figsize=(10, 5))

    bars = plt.bar(models, values)

    plt.title(title)
    plt.ylabel("Score (%)")
    plt.xlabel("Machine Learning Algorithms")

    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")

    # Display values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

# Category charts
create_chart(
    category_accuracy,
    "Category Prediction - Accuracy Comparison",
    "models/category_accuracy_comparison.png"
)

create_chart(
    category_f1,
    "Category Prediction - Macro F1 Comparison",
    "models/category_f1_comparison.png"
)

# Priority charts
create_chart(
    priority_accuracy,
    "Priority Prediction - Accuracy Comparison",
    "models/priority_accuracy_comparison.png"
)

create_chart(
    priority_f1,
    "Priority Prediction - Macro F1 Comparison",
    "models/priority_f1_comparison.png"
)

print("All comparison charts generated successfully!")