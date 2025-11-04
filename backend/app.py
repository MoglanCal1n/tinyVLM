from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, base64, os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)
CORS(app)

client = InferenceClient(token=HF_TOKEN)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend Flask running ✅"})

@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image received"}), 400

        image_bytes = base64.b64decode(image_data.split(",")[1])

        result = client.image_to_text(
            model="vikhyatk/moondream2",
            data=image_bytes
        )

        return jsonify({"caption": str(result)})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
