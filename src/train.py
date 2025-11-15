import pandas as pd
import re
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# --- 1. FUNÇÃO DE LIMPEZA (A mesma do outro script) ---
def clean_text_for_bert(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. FUNÇÃO PARA CALCULAR MÉTRICAS ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 3. CARREGAR E PREPARAR O DATASET ---
print("Carregando e preparando o dataset...")

# Caminho para o CSV
csv_path = os.path.join('..', 'data', 'TweetsWithTheme.csv')
df = pd.read_csv(csv_path)

# Encontra a coluna de texto
text_col = next((col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()), 'tweet_text')

# Limpa o texto
df['cleaned_text'] = df[text_col].apply(clean_text_for_bert)

# Mapeia os labels
# O modelo precisa de inteiros (0 e 1), não de strings
label_map = {'Positivo': 1, 'Negativo': 0}
df['label'] = df['sentiment'].map(label_map)

# Remove linhas sem label (caso existam)
df_clean = df[['cleaned_text', 'label']].dropna().copy()

print(f"Total de {len(df_clean)} linhas válidas para treino.")

# Dividir o dataset (90% treino, 10% validação)
df_train, df_val = train_test_split(df_clean, test_size=0.1, random_state=42, stratify=df_clean['label'])

# Converter para o formato 'Dataset' da Hugging Face
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)

print(f"Dataset dividido: {len(train_dataset)} para treino, {len(val_dataset)} para validação.")

# --- 4. CARREGAR TOKENIZER E MODELO ---
MODELO_BASE = "pysentimiento/bertweet-pt-sentiment"
NOME_MODELO_FINAL = "./meu-modelo-tunado" # Onde o modelo será salvo

print(f"Carregando tokenizer do modelo base: {MODELO_BASE}")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

# Função para tokenizar os dados
def tokenize_function(examples):
    # Usamos 'cleaned_text' como a coluna de texto
    return tokenizer(examples['cleaned_text'], padding="max_length", truncation=True)

print("Tokenizando datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

print(f"Carregando modelo base: {MODELO_BASE}")
print("Modificando a 'cabeça' do modelo para 2 labels (POS/NEG)...")

# AQUI ESTÁ A MÁGICA:
# Nós carregamos o modelo base, mas dizemos a ele que 
# a nova "cabeça" de classificação deve ter num_labels=2
model = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE, 
    num_labels=2,  # <-- Força uma nova cabeça de classificação binária
    id2label={0: "NEG", 1: "POS"}, 
    label2id={"NEG": 0, "POS": 1},
    ignore_mismatched_sizes=True  # <-- A SOLUÇÃO
)

# --- 5. CONFIGURAR O TREINAMENTO ---
print("Configurando o Trainer...")

# Configura os argumentos do treinamento
# --- 5. CONFIGURAR O TREINAMENTO ---
print("Configurando o Trainer...")

training_args = TrainingArguments(
    output_dir=NOME_MODELO_FINAL,      
    num_train_epochs=3,                
    per_device_train_batch_size=16,    
    per_device_eval_batch_size=64,     
    warmup_steps=500,                  
    weight_decay=0.01,                 
    logging_dir='./logs',              
    logging_steps=100,                 
    
    # --- CORREÇÃO PARA O ERRO DE MISMATCH ---
    # Vamos forçar a avaliação e o salvamento a usarem "steps"
    eval_strategy="steps",             # Mude de "epoch" para "steps"
    save_strategy="steps",             # Mude de "epoch" para "steps"
    eval_steps=500,                    # Avalie a cada 500 passos
    save_steps=500,                    # Salve a cada 500 passos
    # --- FIM DA CORREÇÃO ---

    load_best_model_at_end=True,       
    metric_for_best_model="accuracy",  
    report_to="none",                  
)

# Cria o objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# --- 6. TREINAR O MODELO ---
print("\n--- INICIANDO FINE-TUNING ---")
print("Isso pode levar um tempo...")

trainer.train()

print("\n--- TREINAMENTO CONCLUÍDO ---")

# Salva o modelo final
trainer.save_model(NOME_MODELO_FINAL)
print(f"Modelo final salvo em: {NOME_MODELO_FINAL}")

# Avalia o modelo final no conjunto de validação
print("\nAvaliação final do modelo treinado:")
eval_results = trainer.evaluate()
print(eval_results)