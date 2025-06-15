import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from PIL import Image
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()

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

    def calculate_legitimacy_score(self, data: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
        score = 100
        reasons = []

        # Penalize for missing fields
        for field in ['Bill No', 'Date', 'Total Amount']:
            if not data.get(field):
                score -= 20
                reasons.append(f"Missing field: {field}")

        # Penalize if handwritten
        if data.get("IsHandwritten") == True:
            score -= 30
            reasons.append("Handwritten content detected")

        # Penalize for small or blurry images (rough estimate)
        width, height = image.size
        if width < 500 or height < 500:
            score -= 10
            reasons.append("Image resolution is too low (below 500x500)")

        return {
            "score": max(0, min(score, 100)),
            "reasons": reasons
        }

    def extract_bill_data(self, image: Image.Image) -> Dict[str, Any]:
        prompt = """
You are analyzing a bill image.
Extract the following fields in JSON:
- Bill No
- Date
- Total Amount
- IsHandwritten: true if the bill is handwritten, otherwise false.

If a field is missing, set it to null.
If handwriting is detected in any main field (Bill No, Date, Total), IsHandwritten = true.
Return JSON only.
        """
        try:
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                try:
                    bill_data = json.loads(json_content)
                    legitimacy = self.calculate_legitimacy_score(bill_data, image)
                    bill_data["legitimacy_score"] = legitimacy["score"]
                    bill_data["legitimacy_reasons"] = legitimacy["reasons"]
                    return bill_data
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON", "raw_response": response_text}
            else:
                return {"error": "No JSON found", "raw_response": response_text}
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}

# Flask App
app = Flask(__name__)
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
