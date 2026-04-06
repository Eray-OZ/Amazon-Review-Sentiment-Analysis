# YSA Dersi Projesi: CNN + Bi-LSTM Hibrit Mimarisi ile Duygu Analizi
# !pip install datasets

import os
import string
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, TextVectorization
from tensorflow.keras.callbacks import EarlyStopping
from datasets import load_dataset
from google.colab import drive

# 1. Drive Bağlantısı
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Amazon_YSA_Model'
os.makedirs(save_dir, exist_ok=True)

# 2. Devasa ve Kaliteli Veri Setinin Otomatik İndirilmesi
print("1. 'Amazon Polarity' Veri Seti İndiriliyor...")
# 3.6 milyonluk orijinal verinin 150.000'lik devasa bir parçasını Colab'e çekiyoruz
dataset = load_dataset("amazon_polarity", split="train[:150000]")
df = dataset.to_pandas()

# amazon_polarity veri setinde 'label' 0 (Negatif) ve 1 (Pozitif) olarak zaten hazırdır.
df['content'] = df['content'].astype(str).str.strip()
df = df[df['content'] != ''] # Boş satırları temizle

texts = df['content'].values
labels = df['label'].values
print(f"Eğitim için kullanılacak temizlenmiş yorum sayısı: {len(texts)}")

# 3. Metin Vektörizasyonu
print("2. Vektörizasyon işlemi yapılıyor...")
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=200 # Her yorum 200 kelimeye sabitlenir
)
vectorize_layer.adapt(texts)

# 4. YSA HİBRİT MODEL MİMARİSİ (CNN + Bİ-LSTM)
print("3. Hibrit Model Kuruluyor...")
model = Sequential([
    Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    Embedding(input_dim=20000, output_dim=128), # CNN katmanına girdiği için mask_zero kapalı
    
    # --- 1D-CNN Katmanı (Kalıp / Özellik Çıkarımı) ---
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # --- Çift Yönlü LSTM Katmanı (Zaman Serisi / Bağlam) ---
    Bidirectional(LSTM(64)),
    Dropout(0.4), # Nöronların %40'ını kapatarak ezberlemeyi önler (Regularization)
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # İkili sınıflandırma (0-1) çıktısı
])

# YSA'da iyi puan almak için özel Optimizer ayarı
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Eğitim ve Kayıt
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("4. Eğitim Başlıyor (150.000 veri ile eğitildiği için biraz zaman alabilir)...")
history = model.fit(
    texts, labels,
    epochs=10, 
    batch_size=128, # Daha büyük bir veri olduğu için batch_size'ı artırdık
    validation_split=0.1, # %10'unu doğrulama testine ayırdık
    callbacks=[early_stop]
)

final_model_path = os.path.join(save_dir, "ysa_hibrit_model.keras")
model.save(final_model_path)
print(f"\n[BAŞARILI] YSA Hibrit Modeli Drive'a kaydedildi: {final_model_path}")