from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
from inference import inference
from process_results import process_results

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

CORS(app, origins=["http://localhost:3000"])


@app.route("/upload", methods=["POST"])
def upload():
    print("Uploading file...")
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    file.save(f"./uploads/{file.filename}")
    return jsonify({"status": "success"}), 200


@app.route("/run-inference", methods=["POST"])
def predict():
    body = request.get_json()

    success = inference(body["fileName"])
    return jsonify({"success": success}), 200 if success else 500


@app.route("/heatmap/<filename>", methods=["GET"])
def get_heatmap(filename):
    bytes_obj = process_results(filename)
    if bytes_obj is None:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    return (
        send_file(bytes_obj, mimetype="image/png"),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)
