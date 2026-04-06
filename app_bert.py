from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

MODEL_PATH = "model/BERT" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get('text', '').strip()
        
        if not review_text:
            return jsonify({'error': 'Lütfen bir yorum girin.'}), 400
            
        inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        
        neg_score = probabilities[0]
        pos_score = probabilities[1]
        
        if pos_score > 0.5:
            return jsonify({'sentiment': 'Pozitif', 'emoji': '😊', 'confidence': f"%{pos_score * 100:.2f}"})
        else:
            return jsonify({'sentiment': 'Negatif', 'emoji': '😡', 'confidence': f"%{neg_score * 100:.2f}"})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)