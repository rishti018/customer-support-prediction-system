import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "svm": {
        "category": os.path.join(BASE_DIR, "models", "category_model.pkl"),
        "priority": os.path.join(BASE_DIR, "models", "priority_model.pkl")
    },
    "rf": {
        "category": os.path.join(BASE_DIR, "models_rf", "category_model.pkl"),
        "priority": os.path.join(BASE_DIR, "models_rf", "priority_model.pkl")
    }
}

# Store models after they are loaded
models = {}

def load_model(model_type):
    if model_type in models:
        return models[model_type]

    if model_type not in MODEL_PATHS:
        raise ValueError("Choose either 'svm' or 'rf'.")

    paths = MODEL_PATHS[model_type]

    if not os.path.exists(paths["category"]) or not os.path.exists(paths["priority"]):
        raise FileNotFoundError(
            f"Model files missing for {model_type.upper()}"
        )

    with open(paths["category"], "rb") as file:
        category_model = pickle.load(file)

    with open(paths["priority"], "rb") as file:
        priority_model = pickle.load(file)

    models[model_type] = {
        "category": category_model,
        "priority": priority_model
    }

    return models[model_type]


def predict_ticket(subject, body, model_type="svm"):
    selected_models = load_model(model_type)

    ticket_text = subject.strip() + " " + body.strip()

    category = selected_models["category"].predict([ticket_text])[0]
    priority = selected_models["priority"].predict([ticket_text])[0]

    return {
        "category": str(category),
        "priority": str(priority),
        "model_used": model_type
    }