from flask import Flask, request, jsonify
import PyPDF2
import io
import re
from function import *


app = Flask(__name__)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if not request.files:
        return jsonify({"error": "No files received"}), 400

    extracted_texts = ""

    for key in request.files:
        file = request.files[key]
        filename = file.filename.lower()

        if filename.endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                cleaned_text = re.sub(r'\s+', ' ', text).strip()
                extracted_texts += f"{key}: {cleaned_text}"
            except Exception as e:
                return jsonify({"error": f"Failed to process {key}: {str(e)}"}), 500
        else:
            extracted_texts += f"{key}: {request.form.get(key, ' ')}"

    event = request.form.get('event', 'default_event')
    print(extracted_texts)
    chromaAdd(extracted_texts, event)
    return jsonify({
        "message": "Files received and processed successfully.",
        "extracted_texts": extracted_texts,
        "event": event
    })

# POST API
@app.route('/prompt', methods=['POST'])
def post_data():
    data = request.json  # Get JSON data from request
    if not data or 'input' not in data:
        return jsonify({"error": "Invalid input"}), 400
    result = rdRAG(data['input'])  # Call your function
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
