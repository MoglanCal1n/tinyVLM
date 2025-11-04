from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, base64
import torch
from transformers import AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda", # "cuda" on Nvidia GPUs
)

@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image received"}), 400

        # Decodează imaginea din base64
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes))

        # Optionally set sampling settings
        settings = {"temperature": 0.5, "max_tokens": 768, "top_p": 0.3}

        # Generate a short caption
        short_result = model.caption(
            image, 
            length="short", 
            settings=settings
        )

        return jsonify({"caption": str(short_result)})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
