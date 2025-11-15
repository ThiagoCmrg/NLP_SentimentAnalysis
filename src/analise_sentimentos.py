import torch  # <-- Mova o torch para o TOPO
import sys    # <-- Mova o sys para o TOPO
import os
import pandas as pd
import re
import warnings
from transformers import pipeline # <-- Transformers DEPOIS do torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Ignora warnings do pipeline
warnings.filterwarnings('ignore')

# --- ETAPA 1: LIMPEZA DE DADOS ---

def clean_text_for_bert(text):
    """
    Limpeza minimalista para BERT. Remove apenas ruído (links, @mentions)
    e preserva sinais de sentimento (emojis, pontuação, hashtags).
    """
    if not isinstance(text, str):
        return ''
    
    # 1. Remove URLs (ruído)
    text = re.sub(r'http\S+|www\S+', '[URL]', text)
    # 2. Remove mentions (ruído)
    text = re.sub(r'@\w+', '[USER]', text)
    # 3. Normaliza espaços em branco
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Iniciando script de análise de sentimentos...")
print("Carregando e limpando dataset...")

# Carrega o CSV do diretório data
# IMPORTANTE: Este script espera que a pasta 'data' esteja um nível ACIMA
# da pasta onde você executa o script (por causa do '../')
csv_path = os.path.join('..', 'data', 'TweetsWithTheme.csv')
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {csv_path}")
    print("Verifique se o caminho para o 'TweetsWithTheme.csv' está correto.")
    exit()

print(f"Dataset carregado com {len(df)} linhas")

# Encontra a coluna de texto dinamicamente
text_col = next((col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()), None)

if text_col:
    print(f"Limpando coluna: {text_col}")
    # Aplica a limpeza correta
    df['cleaned_text'] = df[text_col].apply(clean_text_for_bert)
    print("Limpeza concluída!")
else:
    print("Erro: Coluna de texto (ex: 'tweet_text') não encontrada!")
    exit()


# --- ETAPA 2: FUNÇÃO DE ANÁLISE (BINÁRIA FORÇADA) ---

def analyze_sentiments_batch_forced_binary(texts, batch_size=32):
    sentiments = []
    confidences = []
    
    print(f"\nIniciando processamento de {len(texts)} textos com o modelo...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_texts = [text if text and len(text.strip()) > 0 else "neutro" for text in batch_texts]
        
        try:
            results = sentiment_pipeline(batch_texts, truncation=True, max_length=512)
            
            for result_list in results:
                scores = {item['label']: item['score'] for item in result_list}
                
                # LÓGICA DE DECISÃO BINÁRIA (POS vs NEG)
                if scores['POS'] > scores['NEG']:
                    sentiments.append('POS')
                    conf = scores['POS'] / (scores['POS'] + scores['NEG'])
                    confidences.append(conf)
                else:
                    sentiments.append('NEG')
                    conf = scores['NEG'] / (scores['POS'] + scores['NEG'])
                    confidences.append(conf)
                    
        except Exception as e:
            print(f"Erro no batch {i}: {e}")
            sentiments.extend(['NEG'] * len(batch_texts))
            confidences.extend([0.0] * len(batch_texts))
        
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(texts):
            print(f"Processadas {min(i + batch_size, len(texts))}/{len(texts)} linhas")
    
    return sentiments, confidences


# --- ETAPA 3: CARREGAR MODELO E EXECUTAR ANÁLISE ---

# --- ETAPA 3: CARREGAR MODELO E EXECUTAR ANÁLISE ---

print("\nVerificando dispositivo (GPU/CPU)...")

# Lógica explícita de seleção de dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
    print("O pipeline usará a GPU.")
else:
    device = torch.device("cpu")
    print("❌ Nenhuma GPU detectada. O pipeline usará a CPU (será lento).")

print("\nCarregando modelo BERTweet-pt...")
# Passamos o *objeto* device, não mais o número 0
sentiment_pipeline = pipeline("sentiment-analysis", 
                             model="pysentimiento/bertweet-pt-sentiment",
                             top_k=None,
                             device=device)  # <-- MUDANÇA IMPORTANTE
print("Modelo carregado!")

# Pega os textos da coluna limpa
texts = df['cleaned_text'].fillna('').tolist()

# Chama a função
sentiments, confidences = analyze_sentiments_batch_forced_binary(texts)

# Adiciona resultados ao DataFrame
df['sentiment_pred'] = sentiments
df['confidence_pred'] = confidences

print("\n✅ Análise de sentimentos (binária forçada) concluída!")


# --- ETAPA 4: AVALIAÇÃO E MÉTRICAS ---

print("\nIniciando avaliação (comparando com o gabarito)...")

# Mapeia seu gabarito (Positivo/Negativo) para os rótulos do modelo (POS/NEG)
mapa_labels = {
    'Positivo': 'POS',
    'Negativo': 'NEG'
}

# Usamos a coluna original 'sentiment' como gabarito
df['gabarito_ajustado'] = df['sentiment'].map(mapa_labels)

# Filtra linhas que não tinham gabarito (caso existam)
df_eval = df.dropna(subset=['gabarito_ajustado'])

print(f"Avaliando {len(df_eval)} de {len(df)} linhas (linhas com gabarito válido).")

# Separa as colunas para comparação
gabarito = df_eval['gabarito_ajustado']
previsoes = df_eval['sentiment_pred']

# Calcula Acurácia
acc = accuracy_score(gabarito, previsoes)
print(f"\n--- Resultados da Avaliação ---")
print(f"Acurácia Geral: {acc * 100:.2f}%")

# Imprime o Relatório de Classificação (Precision, Recall, F1-Score)
print("\nRelatório de Classificação Detalhado:")
print(classification_report(gabarito, previsoes, labels=['POS', 'NEG']))

print("\n--- Análise de Erros (Exemplos) ---")
# Criar coluna de acerto/erro para facilitar a análise
df_eval['acertou'] = (df_eval['gabarito_ajustado'] == df_eval['sentiment_pred'])

# 5 Exemplos que o modelo ERROU
erros = df_eval[df_eval['acertou'] == False]
print(f"\nO modelo errou em {len(erros)} tweets. 5 exemplos de erros:")
print(erros[[text_col, 'sentiment', 'sentiment_pred']].head(5).to_markdown(index=False))

# 5 Exemplos que o modelo ACERTOU
acertos = df_eval[df_eval['acertou'] == True]
print(f"\nO modelo acertou em {len(acertos)} tweets. 5 exemplos de acertos:")
print(acertos[[text_col, 'sentiment', 'sentiment_pred']].head(5).to_markdown(index=False))

print("\nScript finalizado.")