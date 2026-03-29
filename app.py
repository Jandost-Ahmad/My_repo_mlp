"""
app.py  –  Flask backend for the CNN drawing classifier
Drop this file into the root of your project (next to myCnn3.py, run.py, etc.)
then run:  python app.py
"""

import os
import io
import base64
import numpy as np

# ── always resolve paths relative to this file's directory ───────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── torch / model imports ────────────────────────────────────────────────────
import torch
#from myCnn3 import ConvNet     #hier
from mlp import MLP             #hier
from data import (
    get_activation,
    save_activations, normalize_per_channel
)

# ── app setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)   # allow the HTML page to call this backend from any origin

OUTPUT_DIR = "outputs"
INPUT_DIR  = "input"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR,  exist_ok=True)

# ── load model once at startup ───────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model   = ConvNet().to(device)    #hier
model = MLP().to(device)           #hier

#MODEL_PATH = "EMNIST-balanced-CNN.pth"
MODEL_PATH = "mlp_emnist.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'. "
        "Please place the .pth file in the project root."
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Build EMNIST balanced mapping without needing any file.
# EMNIST balanced has 47 classes: digits 0-9, then uppercase A-Z
# then lowercase letters that differ from their uppercase: a b d e f g h i j k l m n p q r t u v x y z

def _build_emnist_mapping():  #Die Funktion habe total verändert nach Herr Schaefer
    path = r"data/EMNIST/raw/emnist-balanced-mapping.txt"
    if not os.path.exists(path):
        # Fallback keeps the app running even if the EMNIST raw mapping file is absent.
        print(f"Warning: mapping file not found at '{path}'. Using index labels instead.")
        return {i: str(i) for i in range(47)}

    mapping = {}
    with open(path) as f:
        for line in f:
            key, val = line.split()
            mapping[int(key)] = chr(int(val))

    return mapping

mapping = _build_emnist_mapping()
activations = {}
last_result  = {}          # ← stores the most recent prediction for the projector

# register forward hooks once
"""
hooks = {
    "conv1":       model.conv1,
    "convStride1": model.convStride1,
    "conv2":       model.conv2,
    "convStride2": model.convStride2,
    "fc1":         model.fc1,
    "fc2":         model.fc2,
    "fc3":         model.fc3,
}
"""      
hooks = {
    "fc1": model.fc[0],  # Linear 784→512    
    "fc2": model.fc[2],  # Linear 512→256
    "fc3": model.fc[4],  # Linear 256→47
}

for name, layer in hooks.items():
    layer.register_forward_hook(get_activation(activations, name))

print(f"Model loaded on {device}  ✓")


# ── helpers ──────────────────────────────────────────────────────────────────
def preprocess_b64(b64_string: str) -> torch.Tensor:
    """Decode a base64 PNG/JPEG, resize to 28×28, invert & normalise."""
    # strip data-url prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    raw  = base64.b64decode(b64_string)
    img  = Image.open(io.BytesIO(raw)).convert("L")          # grayscale
    img  = img.resize((28, 28), Image.LANCZOS)

    arr  = np.array(img, dtype=np.float32) / 255.0           # 0-1
    arr  = 1.0 - arr                                         # invert (white bg → black)
    #arr  = arr.reshape(1, 1, 28, 28)                        # (batch, C, H, W)
    arr  = arr.reshape(1, 28*28)                             # <--- für MLP flatten

    # save the 28×28 input for debugging / display
    #scaled = (arr[0, 0] * 255).astype(np.uint8)
    scaled = (arr[0] * 255).astype(np.uint8)          # Zum Speichern: wieder 28x28 für MLP
    Image.fromarray(scaled).save(os.path.join(INPUT_DIR, "input.png"))

    return torch.from_numpy(arr).float().to(device)


def list_output_images() -> list[dict]:
    """Return sorted list of output activation images with metadata."""
    images = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(".png"):
            images.append({
                "filename": fname,
                "label":    fname.replace(".png", "").replace("_", " "),
                "url":      f"/outputs/{fname}",
            })
    return images


# ── routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/viz")
def viz():
    return send_from_directory("templates", "viz.html")


@app.route("/input/<path:filename>")
def serve_input(filename):
    return send_from_directory(INPUT_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        tensor = preprocess_b64(data["image"])

        with torch.no_grad():
            output = model(tensor)                          # forward pass → fills activations
            pred   = int(torch.argmax(output, dim=1).item())

        char = mapping.get(pred, "?")

        # save activation visualisations
        """
        label_map = [
            ("1_conv1",       "conv1"),
            ("2_convStride1", "convStride1"),
            ("3_conv2",       "conv2"),
            ("4_convStride2", "convStride2"),
            ("5_fc1",         "fc1"),
            ("6_fc2",         "fc2"),
            ("7_fc3",         "fc3"),
        ]
        """
        import time

        probs = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()

        label_map = [("1_fc1", "fc1"),("2_fc2", "fc2"),("3_fc3", "fc3"),]  #für MLP
        for file_label, act_key in label_map:
            if act_key in activations:
                save_activations(activations[act_key], file_label, out_dir=OUTPUT_DIR)

        result = {
            "prediction":    char,
            "label_index":   pred,
            "probabilities": probs,
            "outputs":       list_output_images(),
            "ts":            time.time(),   # projector uses this to detect a NEW prediction
        }
        last_result.clear()
        last_result.update(result)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/latest")
def latest():
    """Projector polls this every second to detect a new prediction."""
    return jsonify(last_result)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/outputs")
def list_outputs():
    return jsonify(list_output_images())


# ── run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)