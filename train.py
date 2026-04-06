import os
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TextVectorization
from tensorflow.keras.callbacks import Callback, EarlyStopping
from google.colab import drive

# 1. Drive Bağlantısı ve Klasör Ayarı
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Amazon_Sentiment_Models'
os.makedirs(save_dir, exist_ok=True)

# 2. Veri Yükleme ve Kesin Temizlik (Modelin Mac'te hata vermemesi için kritik adım)
print("Veri yükleniyor ve temizleniyor...")
df = pd.read_csv('/content/Amazon_Reviews.csv', engine='python', on_bad_lines='skip')
df = df.dropna(subset=['Review Text', 'Rating'])

def extract_rating(rating_str):
    try:
        match = re.search(r'\d+', str(rating_str))
        return int(match.group()) if match else None
    except:
        return None

df['Numeric_Rating'] = df['Rating'].apply(extract_rating)
df = df.dropna(subset=['Numeric_Rating'])
df = df[df['Numeric_Rating'] != 3]
df['sentiment'] = df['Numeric_Rating'].apply(lambda x: 1 if x > 3 else 0)

# Keras 3 Yükleme Hatasını Çözen Kısım: Boş metinleri ve string hatalarını yokediyoruz
df['Review Text'] = df['Review Text'].astype(str).str.strip()
df = df[df['Review Text'] != '']
df = df[~df['Review Text'].isin(['nan', 'None', 'null'])]

texts = df['Review Text'].values
labels = df['sentiment'].values

# 3. Metin Vektörizasyonu
print("Vektörizasyon yapılıyor...")
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=200
)
vectorize_layer.adapt(texts)

# 4. Model Kurulumu
print("Model oluşturuluyor...")
model = Sequential([
    Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    Embedding(input_dim=20000, output_dim=128, mask_zero=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Eğitim ve Kayıt (Erken Durdurma Aktif)
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("Eğitim başlıyor...")
history = model.fit(
    texts, labels,
    epochs=20, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stop]
)

# Eğitim bittikten sonra "EN İYİ" modeli Drive'a nihai olarak kaydet
final_model_path = os.path.join(save_dir, "en_iyi_amazon_modeli.keras")
model.save(final_model_path)
print(f"\n[BAŞARILI] En iyi model Drive'a kaydedildi: {final_model_path}")