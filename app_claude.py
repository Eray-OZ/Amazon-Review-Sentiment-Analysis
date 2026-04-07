from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------------------------------------------------------
# 1. YENİ KİNAYE DEDEKTÖRÜ (Twitter çöpe atıldı, Reddit modeli geldi)
# ---------------------------------------------------------
print("1/2: Reddit Sarcasm Dedektörü Yükleniyor (Sokak dilini daha iyi anlar)...")
SARCASM_MODEL = "helinivan/english-sarcasm-detector"
sarc_tokenizer = AutoTokenizer.from_pretrained(SARCASM_MODEL)
sarc_model = AutoModelForSequenceClassification.from_pretrained(SARCASM_MODEL)

# ---------------------------------------------------------
# 2. SENİN EĞİTTİĞİN FİNAL MODEL
# ---------------------------------------------------------
MODEL_PATH = "model/final_model" 
print(f"2/2: Senin Eğittiğin RoBERTa Modeli Yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("Sistem Ateşlemeye Hazır! 🚀")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_text = data.get('text', '').strip()
        
        if not raw_text:
            return jsonify({'error': 'Lütfen bir yorum girin.'}), 400

        # --- ADIM 1: YENİ KİNAYE DEDEKTÖRÜ ---
        # PyTorch ile modelin direkt beynine (logits) bakıyoruz, pipeline saçmalığı bitti
        sarc_inputs = sarc_tokenizer(raw_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            sarc_outputs = sarc_model(**sarc_inputs)
        
        sarc_probs = torch.nn.functional.softmax(sarc_outputs.logits, dim=-1)[0].tolist()
        
        # Reddit modelinde: Index 0 -> Normal, Index 1 -> Kinaye
        sarcasm_score = sarc_probs[1]
        is_ironic = sarcasm_score > 0.70  # %70'ten eminsesi kinaye de

        # --- ADIM 2: KATI KARAR MEKANİZMASI ---
        if is_ironic:
            # EĞER CÜMLEDE KİNAYE VARSA SİSTEMİ NEGATİFE KİLİTLE
            return jsonify({
                'sentiment': 'Negatif (İroni Tespit Edildi)',
                'emoji': '🙄😡',
                'confidence': f"%{sarcasm_score * 100:.2f}",
                'is_ironic': True,
                'irony_confidence': f"%{sarcasm_score * 100:.2f}",
                'processed_text': raw_text
            })
            
        else:
            # KİNAYE YOKSA NORMAL YORUMDUR, SENİN MODELİNE GÖNDER
            inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
            
            neg_score = probs[0]
            pos_score = probs[1]
            
            sentiment = "Pozitif" if pos_score > 0.5 else "Negatif"
            confidence = max(pos_score, neg_score) * 100
            emoji = "😊" if sentiment == "Pozitif" else "😡"

            return jsonify({
                'sentiment': sentiment,
                'emoji': emoji,
                'confidence': f"%{confidence:.2f}",
                'is_ironic': False,
                'irony_confidence': "Tespit Edilmedi",
                'processed_text': raw_text 
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)