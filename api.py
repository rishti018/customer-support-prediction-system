from flask import Flask, request, jsonify
from model_service import predict_ticket

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Support Ticket Prediction API is running"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "available_models": ["svm"],
        "loaded_models": list(__import__("model_service").models.keys())
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A valid JSON request body is required"}), 400

    subject = data.get("subject", "")
    body = data.get("body", "")
    model_type = data.get("model", "svm")

    if not isinstance(subject, str) or not isinstance(body, str):
        return jsonify({"error": "Subject and body must be text"}), 400

    if not isinstance(model_type, str):
        return jsonify({"error": "Model must be either 'svm' or 'rf'"}), 400

    model_type = model_type.lower().strip()

    if not subject.strip() or not body.strip():
        return jsonify({
            "error": "Subject and body cannot be empty"
        }), 400

    if model_type not in ["svm", "rf"]:
        return jsonify({
            "error": "Model must be either 'svm' or 'rf'"
        }), 400

    try:
        result = predict_ticket(subject, body, model_type)
        return jsonify(result), 200

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
