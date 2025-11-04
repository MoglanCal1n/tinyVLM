from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found. Make sure to set it in your .env file.")
    # You might want to exit or raise an error here if the token is missing
else:
    print("✅ HF_TOKEN loaded successfully.")

app = Flask(__name__)
CORS(app)  # This allows your Netlify app to call this server

# Initialize the client with your token
try:
    client = InferenceClient(token=HF_TOKEN)
    print("✅ Hugging Face client initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Hugging Face client: {e}")
    client = None

@app.route("/process-image", methods=["POST"])
def process_image():
    if not client:
        return jsonify({"error": "Server is not configured with Hugging Face token."}), 500
        
    try:
        data = request.get_json()
        image_data = data.get("image")
        
        # --- Handle incoming prompt (optional enhancement) ---
        # Get a text prompt from the frontend, or use a default.
        prompt_text = data.get("prompt", "Describe what you see in this image.")

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Decode the Base64 string. It's in the format: "data:image/png;base64,..."
        # We need to split on the comma and take the second part.
        try:
            image_bytes = base64.b64decode(image_data.split(",")[1])
        except (IndexError, base64.binascii.Error) as e:
            print(f"❌ Error decoding Base64 string: {e}")
            return jsonify({"error": "Invalid image data format. Expected data URL."}), 400

        print(f"📸 Image received, sending to model with prompt: '{prompt_text}'")

        # --- THE FIX ---
        # 1. Pass the text to the `prompt` argument.
        # 2. Pass the image bytes (in a list) to the `images` argument.
        result = client.text_generation(
            model="vikhyatk/moondream2",
            prompt=prompt_text,
            images=[image_bytes]
        )

        print(f"✅ Model responded: {result}")
        # The result is a string, so we return it in a JSON object
        return jsonify({"caption": result})

    except Exception as e:
        # This will catch errors from the Hugging Face API
        print(f"❌ Error during model inference: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True) # Added debug=True for easier development
