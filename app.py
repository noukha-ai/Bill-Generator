import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from PIL import Image
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BillOCRParser:
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def extract_bill_data(self, image: Image.Image) -> Dict[str, Any]:
        prompt = """
You are an AI assistant analyzing a bill image.

Your tasks:
1. Extract the following fields in JSON format:
   - Bill No
   - Date
   - Total Amount
   - IsHandwritten: true if the bill appears handwritten, otherwise false

2. Based on the extracted data and the quality of the image:
   - Calculate a legitimacy_score from 0 to 100
   - Give reasons for the score in a list called legitimacy_reasons

Scoring rules:
- Start with 100.
- Subtract 20 points for each missing key field (Bill No, Date, Total Amount).
- Subtract 30 points if the bill is handwritten.
- Subtract 10 points if the image resolution appears to be low (less than 500x500 pixels).

Example output:
{
  "Bill No": "12345",
  "Date": "2024-05-01",
  "Total Amount": "Rs. 1245.00",
  "IsHandwritten": false,
  "legitimacy_score": 90,
  "legitimacy_reasons": ["Missing field: Date", "Image resolution is too low"]
}

Respond only with JSON.
        """
        try:
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON", "raw_response": response_text}
            else:
                return {"error": "No JSON found", "raw_response": response_text}
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}

# Flask App
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
parser = None

def get_parser():
    global parser
    if parser is None:
        parser = BillOCRParser()
    return parser

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/process-bill', methods=['POST'])
def process_bill():
    try:
        current_parser = get_parser()
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file name"}), 400

        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({"error": "Only JPG and PNG supported"}), 400

        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as temp_file:
            file.save(temp_file.name)
            image = Image.open(temp_file.name)

        result = current_parser.extract_bill_data(image)

        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        parser = get_parser()
    except Exception as e:
        logger.critical(f"Failed to init parser: {e}")
        raise

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
