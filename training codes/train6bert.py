#!pip install transformers datasets torch scikit-learn

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from google.colab import drive

# 1. Drive Bağlantısı
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/Kesin_Cozum_BERT'
os.makedirs(save_dir, exist_ok=True)

print("1. Veriler Çekiliyor ve Kinaye Aşısı Yapılıyor...")
# Amazon Verisi (40.000 adet)
amazon = load_dataset("amazon_polarity", split="train[:40000]")
df_amazon = amazon.to_pandas()[['content', 'label']]

# Kinaye/İroni Verisi (Cardiff Üniversitesi TweetEval)
irony = load_dataset("tweet_eval", "irony", split="train")
df_irony = irony.to_pandas()
df_sarcasm = df_irony[df_irony['label'] == 1].copy()
df_sarcasm['label'] = 0 # Kinayeyi "Negatif Şikayet" olarak öğretiyoruz
df_sarcasm = df_sarcasm.rename(columns={'text': 'content'})

# Büyük Birleşme ve Karıştırma
df_final = pd.concat([df_amazon, df_sarcasm], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

train_dataset = Dataset.from_pandas(df_final)

print("2. BERT Tokenizer Hazırlanıyor...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)

print("3. BERT Modeli İndiriliyor...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2, 
    per_device_train_batch_size=32,
    logging_dir='./logs',
    learning_rate=2e-5,
    save_strategy="no" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)

trainer.train()

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\n[BAŞARILI] Klasör Hazır: {save_dir}")