# Ficheiro: audio_benchmark.py
# Descrição: Benchmark de 4 embeddings de áudio, armazenamento em Milvus e análise de desempenho melhorada

import os
import time
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pymilvus import MilvusClient
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
from audio_db import extract_audio_metadata
import tracemalloc

# 1. Funções para extrair embeddings de áudio com 4 modelos diferentes

def extract_wav2vec2(audio_path, sample_rate=16000):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    return embedding

def extract_vggish(audio_path, sample_rate=16000):
    import tensorflow as tf
    import tensorflow_hub as hub
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
    embedding = vggish_model(audio).numpy()
    if embedding.ndim > 1:
        embedding = np.mean(embedding, axis=0)
    return embedding

def extract_openl3(audio_path, sample_rate=48000):
    import openl3
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
    if emb.shape[0] > 1:
        emb = np.mean(emb, axis=0)
    else:
        emb = emb.squeeze()
    return emb

def extract_yamnet(audio_path, sample_rate=16000):
    import tensorflow as tf
    import tensorflow_hub as hub
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)

# 2. Função para processar todos os ficheiros de áudio e medir tempos e recursos

def benchmark_embeddings(audio_dir, milvus_uri="http://localhost:19530", max_files=30, repeat=1):
    models = [
        ("wav2vec2", extract_wav2vec2, 768),
        ("vggish", extract_vggish, 128),
        ("openl3", extract_openl3, 512),
        ("yamnet", extract_yamnet, 1024)
    ]
    client = MilvusClient(uri=milvus_uri)
    results = []
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    audio_files = audio_files[:max_files]
    print(f"Encontrados {len(audio_files)} ficheiros de áudio.")

    for model_name, extract_fn, dim in models:
        collection_name = f"audio_{model_name}"
        if not client.has_collection(collection_name):
            client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                metric_type="COSINE",
                primary_field_name="id",
                primary_field_type="INT64"
            )
        print(f"\nBenchmark para {model_name}:")
        for idx, audio_path in enumerate(tqdm(audio_files, desc=f"Processar {model_name}")):
            for r in range(repeat):
                # Medir uso de memória antes
                mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                tracemalloc.start()
                t0 = time.time()
                embedding = extract_fn(audio_path)
                t1 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # current e peak estão em bytes
                mem_peak_mb = peak / 1024 / 1024
                mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                t2 = time.time()
                audio_size = os.path.getsize(audio_path) / 1024 / 1024  # em MB
                metadata = extract_audio_metadata(audio_path)
                audio_duration = metadata.get("duration", 0)
                client.insert(
                    collection_name=collection_name,
                    data=[{
                        "id": idx * repeat + r,
                        "vector": embedding.tolist(),
                        "filename": os.path.basename(audio_path),
                        "title": metadata.get("title"),
                        "duration": metadata.get("duration"),
                        "authors": metadata.get("authors"),
                        "channels": metadata.get("channels"),
                        "sample_rate": metadata.get("sample_rate"),
                        "album": metadata.get("album"),
                        "genre": metadata.get("genre"),
                    }]
                )
                t3 = time.time()
                results.append({
                    "modelo": model_name,
                    "ficheiro": os.path.basename(audio_path),
                    "tempo_extracao": t1 - t0,
                    "tempo_insercao": t3 - t2,
                    "memoria_peak_mb": mem_peak_mb,
                    "dimensao": dim,
                    "tamanho_audio_mb": audio_size,
                    "duracao_audio_s": audio_duration,
                    "repeat": r+1,
                    "embedding": embedding.tolist()  # <--- adiciona isto!
                })
    return pd.DataFrame(results)

# 3. Função para análise e apresentação dos resultados

def analisar_resultados(df):
    print("\nResumo dos tempos médios por modelo:")
    resumo = df.groupby("modelo")[["tempo_extracao", "tempo_insercao", "memoria_peak_mb"]].mean()
    print(resumo)
    print("\nNúmero de vetores inseridos por modelo:")
    print(df.groupby("modelo")["ficheiro"].count())

    # Gráfico de barras com seaborn
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="modelo", y="tempo_extracao", ci="sd")
    plt.title("Tempo de extração por modelo")
    plt.ylabel("Tempo de extração (s)")
    plt.tight_layout()
    plt.savefig("tempo_extracao_seaborn.png")
    print("Gráfico guardado em tempo_extracao_seaborn.png")

    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="modelo", y="tempo_insercao", ci="sd")
    plt.title("Tempo de inserção por modelo")
    plt.ylabel("Tempo de inserção (s)")
    plt.tight_layout()
    plt.savefig("tempo_insercao_seaborn.png")
    print("Gráfico guardado em tempo_insercao_seaborn.png")


    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="modelo", y="memoria_peak_mb", errorbar="sd")
    plt.title("Pico de memória RAM por modelo")
    plt.ylabel("Pico de memória RAM usada (MB)")
    plt.tight_layout()
    plt.savefig("memoria_pico_seaborn.png")
    print("Gráfico guardado em memoria_pico_seaborn.png")

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x="tamanho_audio_mb", y="tempo_extracao", hue="modelo")
    plt.title("Tempo de extração vs Tamanho do ficheiro de áudio")
    plt.xlabel("Tamanho do ficheiro (MB)")
    plt.ylabel("Tempo de extração (s)")
    plt.tight_layout()
    plt.savefig("scatter_tamanho_vs_tempo.png")
    print("Gráfico scatter guardado em scatter_tamanho_vs_tempo.png")

    # Seleciona apenas as colunas numéricas e o modelo
    cols = ["modelo", "tempo_extracao", "tempo_insercao", "memoria_peak_mb", "tamanho_audio_mb", "duracao_audio_s"]
    sns.pairplot(df[cols], hue="modelo", diag_kind="kde")
    plt.savefig("pairplot_benchmark.png")
    print("Pairplot guardado em pairplot_benchmark.png")

    # Output para ficheiro txt
    with open("resumo_benchmark.txt", "w") as f:
        f.write("Resumo dos tempos médios por modelo:\n")
        f.write(resumo.to_string())
        f.write("\n\nNúmero de vetores inseridos por modelo:\n")
        f.write(df.groupby("modelo")["ficheiro"].count().to_string())
        f.write("\n\nDescrição estatística detalhada:\n")
        f.write(df.describe().to_string())
    print("Resumo detalhado guardado em resumo_benchmark.txt")

def testar_pesquisas(client, df, top_k=5):
    """
    Testa pesquisas vetoriais para cada modelo, mede o tempo de pesquisa e guarda resultados e gráfico.
    """
    tempos_pesquisa = []
    print("\n--- Benchmark de Pesquisa Vetorial ---")
    for modelo in df['modelo'].unique():
        collection_name = f"audio_{modelo}"
        # Seleciona o primeiro vetor desse modelo para pesquisa
        row = df[df['modelo'] == modelo].iloc[0]
        # Recupera o vetor do Milvus (ou guarda o embedding durante o benchmark para usar aqui)
        # Aqui vamos supor que guardaste o embedding no benchmark, senão podes extrair de novo
        # Exemplo: query_vector = extrair_embedding(row['ficheiro'])
        # Mas para simplificar, vamos só mostrar o tempo de pesquisa
        try:
            # Pesquisa pelo filename (podes adaptar para guardar o embedding no benchmark)
            result = client.search(
                collection_name=collection_name,
                data=[row['embedding']],  # Usa o embedding real do ficheiro
                limit=top_k,
                output_fields=["filename", "title", "duration", "authors", "genre"]
            )
            tempo = result[1]['elapsed'] if isinstance(result, tuple) and 'elapsed' in result[1] else None
        except Exception as e:
            print(f"Erro ao pesquisar em {collection_name}: {e}")
            tempo = None
        tempos_pesquisa.append({"modelo": modelo, "tempo_pesquisa_s": tempo if tempo is not None else 0})

    tempos_df = pd.DataFrame(tempos_pesquisa)
    print(tempos_df)

    # Gráfico dos tempos de pesquisa
    plt.figure(figsize=(8,5))
    sns.barplot(data=tempos_df, x="modelo", y="tempo_pesquisa_s")
    plt.title("Tempo de pesquisa vetorial por modelo")
    plt.ylabel("Tempo de pesquisa (s)")
    plt.tight_layout()
    plt.savefig("tempo_pesquisa_vetorial.png")
    print("Gráfico de tempo de pesquisa guardado em tempo_pesquisa_vetorial.png")
    # Guarda também em txt
    with open("tempos_pesquisa.txt", "w") as f:
        f.write(tempos_df.to_string(index=False))
    print("Tempos de pesquisa guardados em tempos_pesquisa.txt")

# 4. Execução principal

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark de embeddings de áudio")
    parser.add_argument("--audio_dir", type=str, default="audio_data", help="Diretório com ficheiros de áudio")
    parser.add_argument("--max_files", type=int, default=30, help="Número máximo de ficheiros a processar")
    parser.add_argument("--repeat", type=int, default=1, help="Número de repetições por ficheiro/modelo")
    args = parser.parse_args()

    df = benchmark_embeddings(args.audio_dir, max_files=args.max_files, repeat=args.repeat)
    df.to_csv("resultados_benchmark.csv", index=False)
    print("\nResultados detalhados guardados em resultados_benchmark.csv")
    analisar_resultados(df)
    # Testar pesquisas vetoriais
    client = MilvusClient(uri="http://localhost:19530")
    testar_pesquisas(client, df)