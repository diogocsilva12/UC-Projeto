# ============================================================
# IMPORTS E CARREGAMENTO DE MODELOS
# ============================================================

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
import logging
import glob
import streamlit as st
import GPUtil

# Carregamento dos modelos de embeddings apenas uma vez (eficiente)
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ClapProcessor, ClapModel, ASTFeatureExtractor, ASTModel
import tensorflow_hub as hub
import openl3
import tensorflow as tf

# Configuração para evitar erros de compatibilidade com TensorFlow Hub
os.environ['TFHUB_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.cache', 'tfhub_modules')

# Modelos HuggingFace
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
ast_feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Modelos TensorFlow Hub - usando abordagem alternativa para carregamento
try:
    # Tenta usar modelo VGGish com URL alternativa
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
except (ValueError, tf.errors.NotFoundError, tf.errors.InternalError) as e:
    logging.warning(f"Erro ao carregar VGGish com URL padrão: {e}")
    try:
        # Tenta URL alternativa
        vggish_model = hub.load('https://tfhub.dev/google/vggish/1', tags=None)
    except Exception as e:
        logging.error(f"Não foi possível carregar o modelo VGGish: {e}")
        # Implementa um modelo VGGish dummy para permitir que o código continue funcionando
        class DummyVGGish:
            def __call__(self, audio):
                # Retorna um embedding simulado com a dimensão correta (128)
                return np.zeros((1, 128), dtype=np.float32)
        vggish_model = DummyVGGish()
        logging.warning("Usando modelo VGGish simulado (dummy) como fallback")

try:
    # Tenta carregar YAMNet com URL padrão
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
except (ValueError, tf.errors.NotFoundError, tf.errors.InternalError) as e:
    logging.warning(f"Erro ao carregar YAMNet com URL padrão: {e}")
    try:
        # Tenta URL alternativa
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1', tags=None)
    except Exception as e:
        logging.error(f"Não foi possível carregar o modelo YAMNet: {e}")
        # Implementa um modelo YAMNet dummy para permitir que o código continue funcionando
        class DummyYAMNet:
            def __call__(self, waveform):
                # Retorna scores, embeddings, spectrograma simulados
                return (
                    np.zeros((1, 521), dtype=np.float32),  # scores
                    np.zeros((1, 1024), dtype=np.float32),  # embeddings
                    np.zeros((1, 96, 64), dtype=np.float32)  # spectrogram
                )
        yamnet_model = DummyYAMNet()
        logging.warning("Usando modelo YAMNet simulado (dummy) como fallback")

# ============================================================
# FUNÇÕES DE EXTRAÇÃO DE EMBEDDINGS
# ============================================================

def extract_wav2vec2(audio_path, processor, model, sample_rate=16000):
    """Extrai embedding com wav2vec2."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    return embedding

def extract_clap(audio_path, processor, model, sample_rate=48000):
    """Extrai embedding com CLAP."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_audio_features(**inputs)
    return embeddings[0].numpy()

def extract_ast(audio_path, feature_extractor, model, sample_rate=16000):
    """Extrai embedding com AST."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

def extract_vggish(audio_path, model, sample_rate=16000):
    """Extrai embedding com VGGish."""
    import numpy as np
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    embedding = model(audio).numpy()
    if embedding.ndim > 1:
        embedding = np.mean(embedding, axis=0)
    return embedding

def extract_yamnet(audio_path, model, sample_rate=16000):
    """Extrai embedding com YAMNet."""
    import tensorflow as tf
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform)
    return np.mean(embeddings.numpy(), axis=0)

def extract_openl3(audio_path, sample_rate=48000):
    """Extrai embedding com OpenL3."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
    if emb.shape[0] > 1:
        emb = np.mean(emb, axis=0)
    else:
        emb = emb.squeeze()
    return emb

# ============================================================
# BENCHMARK: PROCESSAMENTO E MEDIÇÃO DE RECURSOS
# ============================================================

def benchmark_embeddings(audio_dir, milvus_uri="http://localhost:19530", max_files=30, repeat=1):
    """
    Processa todos os ficheiros de áudio, extrai embeddings com vários modelos,
    mede tempos, recursos e insere no Milvus.
    """
    models = [
        ("wav2vec2", lambda path: extract_wav2vec2(path, wav2vec2_processor, wav2vec2_model), 768),
        ("vggish", lambda path: extract_vggish(path, vggish_model), 128),
        ("openl3", extract_openl3, 512),
        ("yamnet", lambda path: extract_yamnet(path, yamnet_model), 1024),
        ("clap", lambda path: extract_clap(path, clap_processor, clap_model), 512),
        ("ast", lambda path: extract_ast(path, ast_feature_extractor, ast_model), 768)
    ]
    client = MilvusClient(uri=milvus_uri)
    results = []
    audio_files = []
    # Procura todos os ficheiros de áudio no diretório
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    # Limita o número de arquivos ao máximo especificado
    audio_files = audio_files[:max_files]
    # Armazena o número real de arquivos (importante para nomeação das coleções)
    actual_file_count = len(audio_files) 
    print(f"Encontrados {actual_file_count} ficheiros de áudio.")

    # Para cada modelo, processa todos os ficheiros
    for model_name, extract_fn, dim in models:
        # Usa o número REAL de arquivos no nome da coleção
        collection_name = f"audio_{model_name}_{actual_file_count}"
        # Cria coleção no Milvus se não existir
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
                # Mede recursos antes/depois da extração
                cpu_before = psutil.cpu_percent(interval=None)
                ram_before = psutil.virtual_memory().percent
                tracemalloc.start()
                t0 = time.time()
                try:
                    embedding = extract_fn(audio_path)
                except Exception as e:
                    logging.error(f"Erro ao extrair embedding de {audio_path}: {e}")
                    continue
                t1 = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_peak_mb = peak / 1024 / 1024
                cpu_after = psutil.cpu_percent(interval=None)
                ram_after = psutil.virtual_memory().percent
                mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                t2 = time.time()
                audio_size = os.path.getsize(audio_path) / 1024 / 1024  # em MB
                metadata = extract_audio_metadata(audio_path)
                audio_duration = metadata.get("duration", 0)
                gpus = GPUtil.getGPUs()
                gpu_usage = gpus[0].load * 100 if gpus else 0
                # Insere embedding e metadados no Milvus
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
                # Guarda resultados detalhados para análise posterior
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
                    "embedding": embedding.tolist(),
                    "cpu_percent_before": cpu_before,
                    "cpu_percent_after": cpu_after,
                    "ram_percent_before": ram_before,
                    "ram_percent_after": ram_after,
                    "gpu_usage": gpu_usage
                })
                logging.info(f"Processado {audio_path} ({model_name}), tempo extração: {t1-t0:.3f}s, CPU antes: {cpu_before}, CPU depois: {cpu_after}, RAM antes: {ram_before}, RAM depois: {ram_after}, GPU: {gpu_usage}")
    return pd.DataFrame(results)

# ============================================================
# ANÁLISE E VISUALIZAÇÃO DOS RESULTADOS
# ============================================================

def analisar_resultados(df, benchmark_folder, client):
    """
    Gera estatísticas, gráficos e relatórios a partir dos resultados do benchmark.
    """
    print("\nResumo dos tempos médios por modelo:")
    resumo = df.groupby("modelo")[["tempo_extracao", "tempo_insercao", "memoria_peak_mb"]].mean()
    print(resumo)
    print("\nNúmero de vetores inseridos por modelo:")
    print(df.groupby("modelo")["ficheiro"].count())

    # Gráfico de barras: tempo de extração por modelo
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="modelo", y="tempo_extracao", errorbar="sd")
    plt.title("Tempo de extração por modelo")
    plt.ylabel("Tempo de extração (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "tempo_extracao_seaborn.png"))
    print("Gráfico guardado em tempo_extracao_seaborn.png")

    # Gráfico de barras: tempo de inserção por modelo
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="modelo", y="tempo_insercao", errorbar="sd")
    plt.title("Tempo de inserção por modelo")
    plt.ylabel("Tempo de inserção (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "tempo_insercao_seaborn.png"))
    print("Gráfico guardado em tempo_insercao_seaborn.png")

    # Gráfico de barras: pico de memória por modelo
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="modelo", y="memoria_peak_mb", errorbar="sd")
    plt.title("Pico de memória RAM por modelo")
    plt.ylabel("Pico de memória RAM usada (MB)")
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "memoria_pico_seaborn.png"))
    print("Gráfico guardado em memoria_pico_seaborn.png")

    # Scatterplot tempo de extração vs tamanho do ficheiro (só se houver variabilidade)
    if df["tamanho_audio_mb"].std() > 0.01:
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df, x="tamanho_audio_mb", y="tempo_extracao", hue="modelo")
        sns.regplot(data=df, x="tamanho_audio_mb", y="tempo_extracao", scatter=False, color="black", line_kws={"linestyle":"dashed"})
        plt.title("Tempo de extração vs Tamanho do ficheiro de áudio")
        plt.xlabel("Tamanho do ficheiro (MB)")
        plt.ylabel("Tempo de extração (s)")
        plt.yscale('log')  # log-scale pode ajudar se os valores forem muito próximos
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_folder, "scatter_tamanho_vs_tempo.png"))
        print("Gráfico scatter guardado em scatter_tamanho_vs_tempo.png")
    else:
        print("Não há variabilidade suficiente em 'tamanho_audio_mb' para gerar scatterplot.")

    # Correlação entre duração do áudio e tempo de extração (só se houver variabilidade)
    if df["duracao_audio_s"].std() > 0.01:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="duracao_audio_s", y="tempo_extracao", hue="modelo")
        sns.regplot(data=df, x="duracao_audio_s", y="tempo_extracao", scatter=False, color="black", line_kws={"linestyle":"dashed"})
        plt.title("Correlação entre duração do áudio e tempo de extração")
        plt.xlabel("Duração do áudio (s)")
        plt.ylabel("Tempo de extração (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_folder, "correlacao_duracao_tempo.png"))
        print("Gráfico de correlação guardado em correlacao_duracao_tempo.png")
    else:
        print("Não há variabilidade suficiente em 'duracao_audio_s' para gerar gráfico de correlação.")

    # Boxplot dos tempos por modelo
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x="modelo", y="tempo_extracao")
    plt.title("Boxplot do tempo de extração por modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "boxplot_tempo_extracao.png"))
    print("Boxplot guardado em boxplot_tempo_extracao.png")

    # Scatterplot 3D (só se houver variabilidade suficiente)
    if df["tamanho_audio_mb"].std() > 0.01 and df["memoria_peak_mb"].std() > 0.01 and df["tempo_extracao"].std() > 0.01:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        for modelo in df['modelo'].unique():
            subset = df[df['modelo'] == modelo]
            ax.scatter(subset['tempo_extracao'], subset['memoria_peak_mb'], subset['tamanho_audio_mb'], label=modelo)
        ax.set_xlabel('Tempo de extração (s)')
        ax.set_ylabel('Memória pico (MB)')
        ax.set_zlabel('Tamanho ficheiro (MB)')
        plt.title("Scatterplot 3D: Tempo vs Memória vs Tamanho")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(benchmark_folder, "scatter3d_tempo_memoria_tamanho.png"))
        print("Scatterplot 3D guardado em scatter3d_tempo_memoria_tamanho.png")
    else:
        print("Não há variabilidade suficiente para gerar scatterplot 3D.")

    # Gráficos de evolução dos recursos (CPU, RAM, GPU)
    plt.figure(figsize=(12,6))
    for modelo in df['modelo'].unique():
        subset = df[df['modelo'] == modelo]
        plt.plot(subset.index, subset['cpu_percent_after'], label=f'CPU {modelo}')
    plt.title("Evolução do uso de CPU (%)")
    plt.xlabel("Processamento (ordem dos ficheiros)")
    plt.ylabel("CPU (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "evolucao_cpu.png"))
    print("Gráfico de evolução do uso de CPU guardado em evolucao_cpu.png")

    plt.figure(figsize=(12,6))
    for modelo in df['modelo'].unique():
        subset = df[df['modelo'] == modelo]
        plt.plot(subset.index, subset['ram_percent_after'], label=f'RAM {modelo}')
    plt.title("Evolução do uso de RAM (%)")
    plt.xlabel("Processamento (ordem dos ficheiros)")
    plt.ylabel("RAM (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "evolucao_ram.png"))
    print("Gráfico de evolução do uso de RAM guardado em evolucao_ram.png")

    plt.figure(figsize=(12,6))
    for modelo in df['modelo'].unique():
        subset = df[df['modelo'] == modelo]
        plt.plot(subset.index, subset['gpu_usage'], label=f'GPU {modelo}')
    plt.title("Evolução do uso de GPU (%)")
    plt.xlabel("Processamento (ordem dos ficheiros)")
    plt.ylabel("GPU (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "evolucao_gpu.png"))
    print("Gráfico de evolução do uso de GPU guardado em evolucao_gpu.png")

# ============================================================
# BENCHMARK DE PESQUISA VETORIAL
# ============================================================

def testar_pesquisas(client, df, benchmark_folder, top_k=5, actual_file_count=None):
    """
    Realiza pesquisas vetoriais no Milvus para cada modelo e guarda os scores.
    """
    tempos_pesquisa = []
    print("\n--- Benchmark de Pesquisa Vetorial ---")
    for modelo in df['modelo'].unique():
        # Usa o número REAL de arquivos no nome da coleção
        collection_name = f"audio_{modelo}_{actual_file_count}"
        row = df[df['modelo'] == modelo].iloc[0]
        try:
            result = client.search(
                collection_name=collection_name,
                data=[row['embedding']],
                limit=top_k,
                output_fields=["filename", "title", "duration", "authors", "genre"]
            )
            print(f"\nResultados para {modelo}:")
            if result and len(result[0]) > 0:
                for hit in result[0]:
                    print(f"Score: {hit.get('score', 'N/A')} | Ficheiro: {hit.get('filename')} | Título: {hit.get('title')}")
                tempo = result[0][0].get('score', None)
            else:
                print(f"Nenhum resultado encontrado para {collection_name}")
                tempo = None
        except Exception as e:
            print(f"Erro ao pesquisar em {collection_name}: {e}")
            tempo = None
        tempos_pesquisa.append({"modelo": modelo, "score_primeiro_resultado": tempo if tempo is not None else 0})

    tempos_df = pd.DataFrame(tempos_pesquisa)
    tempos_df.to_csv(os.path.join(benchmark_folder, "tempos_pesquisa.csv"), index=False)

    # Gráfico dos scores do primeiro resultado por modelo
    plt.figure(figsize=(8,5))
    sns.barplot(data=tempos_df, x="modelo", y="score_primeiro_resultado")
    plt.title("Score do 1º resultado da pesquisa vetorial por modelo")
    plt.ylabel("Score (distância)")
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_folder, "score_primeiro_resultado.png"))
    print("Gráfico dos scores guardado em score_primeiro_resultado.png")

# ============================================================
# UTILITÁRIO PARA CRIAR PASTA DE BENCHMARKS
# ============================================================

def get_benchmark_folder(num_files):
    """
    Cria (se necessário) e devolve o caminho da pasta para guardar os resultados do benchmark.
    """
    base_folder = f"benchmarks/benchmark_{num_files}"
    
    # Verifica se já existe uma pasta com esse nome
    if os.path.exists(base_folder):
        # Encontra um nome único adicionando um número sequencial
        i = 1
        while os.path.exists(f"{base_folder}_{i}"):
            i += 1
        folder = f"{base_folder}_{i}"
    else:
        folder = base_folder
        
    os.makedirs(folder, exist_ok=True)
    print(f"Resultados serão guardados em: {folder}")
    return folder

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark de embeddings de áudio")
    parser.add_argument("--audio_dir", type=str, default="audio_data", help="Diretório com ficheiros de áudio")
    parser.add_argument("--max_files", type=int, default=30, help="Número máximo de ficheiros a processar")
    parser.add_argument("--repeat", type=int, default=1, help="Número de repetições por ficheiro/modelo")
    args = parser.parse_args()

    # Executa benchmark principal
    df = benchmark_embeddings(args.audio_dir, max_files=args.max_files, repeat=args.repeat)
    
    # Obtém o número real de arquivos processados (pode ser diferente de max_files)
    actual_file_count = len(df['ficheiro'].unique())
    
    # Cria pasta para guardar resultados deste benchmark
    benchmark_folder = get_benchmark_folder(actual_file_count)

    # Configura logging para ficheiro
    logging.basicConfig(
        filename=os.path.join(benchmark_folder, "benchmark.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    df.to_csv(os.path.join(benchmark_folder, "resultados_benchmark.csv"), index=False)
    print("\nResultados detalhados guardados em resultados_benchmark.csv")

    # Cria cliente Milvus para análise/pesquisa
    client = MilvusClient(uri="http://localhost:19530")

    # Gera gráficos e estatísticas
    analisar_resultados(df, benchmark_folder, client)

    # Testa pesquisas vetoriais com o número real de arquivos
    testar_pesquisas(client, df, benchmark_folder, actual_file_count=actual_file_count)