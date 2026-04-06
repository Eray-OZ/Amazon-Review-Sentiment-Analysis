import os
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, TextVectorization
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive

# 1. Drive Bağlantısı ve Klasör Ayarı
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Amazon_Sentiment_Models'
os.makedirs(save_dir, exist_ok=True)

# 2. Veri Yükleme ve Kesin Temizlik
print("Veri yükleniyor ve temizleniyor...")
# Dosyayı Colab ana dizinine (Klasör simgesinin oraya) yüklediğinden emin ol!
df = pd.read_csv('/content/Amazon_Reviews.csv', engine='python', on_bad_lines='skip')
df = df.dropna(subset=['Review Text', 'Rating'])

# Puanları sayıya çevirme fonksiyonu
def extract_rating(rating_str):
    try:
        match = re.search(r'\d+', str(rating_str))
        return int(match.group()) if match else None
    except:
        return None

df['Numeric_Rating'] = df['Rating'].apply(extract_rating)
df = df.dropna(subset=['Numeric_Rating'])

# 3 Yıldızları (Nötr) atıp, 1-2'yi Negatif (0), 4-5'i Pozitif (1) yapıyoruz
df = df[df['Numeric_Rating'] != 3]
df['sentiment'] = df['Numeric_Rating'].apply(lambda x: 1 if x > 3 else 0)

# Keras kelime sözlüğü hatalarını önlemek için sıkı metin temizliği
df['Review Text'] = df['Review Text'].astype(str).str.strip()
df = df[df['Review Text'] != '']
df = df[~df['Review Text'].isin(['nan', 'None', 'null'])]

texts = df['Review Text'].values
labels = df['sentiment'].values
print(f"Eğitim için kullanılacak temizlenmiş yorum sayısı: {len(texts)}")

# 3. Metin Vektörizasyonu
print("Vektörizasyon yapılıyor...")
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, '<br />', ' ')
    # Noktalama işaretlerini sil
    return tf.strings.regex_replace(stripped, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=200
)
vectorize_layer.adapt(texts)

# 4. GELİŞTİRİLMİŞ MODEL KURULUMU (Ödev İçin Kendi Tasarımın)
print("Model oluşturuluyor (Bidirectional LSTM & Dropout)...")
model = Sequential([
    Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    Embedding(input_dim=20000, output_dim=128, mask_zero=True),
    
    # 1. Çift Yönlü LSTM Katmanı (return_sequences=True sonraki LSTM'e veri aktarmak için şart)
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3), # Aşırı öğrenmeyi (ezberlemeyi) önlemek için nöronların %30'unu rastgele kapatır
    
    # 2. Çift Yönlü LSTM Katmanı (Artık daha derin bir zekaya sahip)
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # İkili sınıflandırma (0 veya 1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary() # Modelin yapısını ödev raporuna eklemek istersen bu tablo işine çok yarayacaktır.

# 5. Eğitim ve Kayıt
# Model ezberlemeye (val_loss artmaya) başladığında eğitimi durdurur ve en iyi ağırlıkları geri yükler
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("\nEğitim başlıyor...")
history = model.fit(
    texts, labels,
    epochs=20, # Early stopping olduğu için 20'ye kadar rahatça çıkabiliriz
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stop]
)

# Eğitim bittikten sonra en iyi modeli Drive'a kaydet
final_model_path = os.path.join(save_dir, "gelismis_amazon_modeli.keras")
model.save(final_model_path)
print(f"\n[BAŞARILI] Geliştirilmiş model Drive'a kaydedildi: {final_model_path}")