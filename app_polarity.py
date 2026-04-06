from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import keras
import string
import re

app = Flask(__name__)

original_set_vocab = keras.layers.StringLookup.set_vocabulary

def safe_set_vocabulary(self, vocabulary, *args, **kwargs):
    if vocabulary is not None:
        seen = set()
        clean_vocab = []
        for i, item in enumerate(vocabulary):
            val = item.decode('utf-8') if isinstance(item, bytes) else str(item)
            if val not in seen:
                clean_vocab.append(item)
                seen.add(val)
            else:
                dummy_val = f"{val}_dummy_{i}"
                clean_vocab.append(dummy_val.encode('utf-8') if isinstance(item, bytes) else dummy_val)
        
        max_limit = getattr(self, 'max_tokens', 20000)
        if max_limit is not None and len(clean_vocab) > max_limit:
            clean_vocab = clean_vocab[:max_limit]
        elif len(clean_vocab) > 20000: 
            clean_vocab = clean_vocab[:20000]
            
        vocabulary = clean_vocab
    return original_set_vocab(self, vocabulary, *args, **kwargs)

keras.layers.StringLookup.set_vocabulary = safe_set_vocabulary

@keras.saving.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')

# Yeni eğittiğin modelin adı
MODEL_PATH = "model/train5irony_epoch4_ysa.keras" 

print("YSA Hibrit Modeli yükleniyor, lütfen bekleyin...")
model = keras.models.load_model(MODEL_PATH)
print("Model başarıyla yüklendi!")

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
            
        # Numpy yerine TensorFlow Tensor formatı
        input_tensor = tf.constant([review_text], dtype=tf.string)
        prediction = model.predict(input_tensor)
        
        score = float(prediction[0][0])
        
        if score > 0.5:
            return jsonify({'sentiment': 'Pozitif', 'emoji': '😊', 'confidence': f"%{score * 100:.2f}"})
        else:
            return jsonify({'sentiment': 'Negatif', 'emoji': '😡', 'confidence': f"%{(1 - score) * 100:.2f}"})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)