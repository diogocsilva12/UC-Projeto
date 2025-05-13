# Dashboard Interativo de Benchmarks de Áudio (Jupyter Notebook)

import pandas as pd
import numpy as np
import glob
import os
import ast
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings("ignore")

# 1. Seleção e carregamento dos dados
folders = sorted(glob.glob("benchmarks/benchmark_*"))
if not folders:
    raise FileNotFoundError("Nenhum benchmark encontrado na pasta 'benchmarks'.")
display(Markdown(f"**Benchmarks disponíveis:** {folders}"))

benchmark_folder = folders[0]  # Altera o índice se quiseres outro benchmark
print(f"Benchmark selecionado: {benchmark_folder}")

csv_path = os.path.join(benchmark_folder, "resultados_benchmark.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("CSV de resultados não encontrado!")
df = pd.read_csv(csv_path)
display(Markdown("**Dados carregados com sucesso!**"))

# 2. Filtros dinâmicos
modelos = df['modelo'].unique().tolist()
display(Markdown(f"**Modelos disponíveis:** {modelos}"))
modelos_filtrados = modelos  # ou ['wav2vec2', 'vggish']
tempo_min, tempo_max = df['tempo_extracao'].min(), df['tempo_extracao'].max()
df_filt = df[df['modelo'].isin(modelos_filtrados) & df['tempo_extracao'].between(tempo_min, tempo_max)]
display(Markdown(f"**Filtros aplicados:** Modelos={modelos_filtrados}, Tempo=({tempo_min}, {tempo_max})"))

# 3. Sumário estatístico
display(Markdown("**Estatísticas principais:**"))
display(df_filt.describe())

# 4. Tabela de resultados
display(Markdown("""
**Descrição:**  
Esta tabela apresenta todos os resultados do benchmark, incluindo métricas de desempenho, recursos utilizados e informações dos ficheiros de áudio processados por cada modelo.  
Permite consultar, filtrar e exportar os dados para análise detalhada.
"""))
display(df_filt)
df_filt.to_csv("resultados_benchmark_filtrado.csv", index=False)
display(Markdown("CSV exportado: `resultados_benchmark_filtrado.csv`"))

# 5. Gráficos Interativos
metricas = ["tempo_extracao", "tempo_insercao", "memoria_peak_mb", "cpu_percent_after", "ram_percent_after", "gpu_usage"]
metrica_bar = metricas[0]
metrica_box = metricas[0]

# Gráfico de barras
fig = px.bar(df_filt, x="modelo", y=metrica_bar, color="modelo", barmode="group",
             title=f"{metrica_bar.replace('_',' ').capitalize()} por modelo")
fig.show()

# Boxplot
fig = px.box(df_filt, x="modelo", y=metrica_box, color="modelo",
             title=f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por modelo")
fig.show()

# Scatterplot personalizado
x_axis = "tamanho_audio_mb"
y_axis = "tempo_extracao"
fig = px.scatter(df_filt, x=x_axis, y=y_axis, color="modelo", title=f"{y_axis} vs {x_axis}")
fig.show()

# Scatterplot 3D
fig = px.scatter_3d(df_filt, x="tempo_extracao", y="memoria_peak_mb", z="tamanho_audio_mb", color="modelo",
                    title="3D: tempo_extracao vs memoria_peak_mb vs tamanho_audio_mb")
fig.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_filt, x="modelo", y="tempo_extracao")
plt.title("Distribuição dos tempos de extração por modelo (Violin Plot)")
plt.show()

# Pairplot
sns.pairplot(df_filt, hue="modelo", vars=["tempo_extracao", "memoria_peak_mb", "tamanho_audio_mb", "duracao_audio_s"])
plt.show()

# Heatmap de correlação
plt.figure(figsize=(8, 6))
corr = df_filt[["tempo_extracao", "memoria_peak_mb", "tamanho_audio_mb", "duracao_audio_s"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Mapa de Correlação entre Métricas")
plt.show()

# 6. Visualização t-SNE dos embeddings (opcional)
if "embedding" in df_filt.columns:
    from sklearn.manifold import TSNE
    display(Markdown("### Visualização t-SNE dos Embeddings"))
    modelo = modelos[0]  # ou outro modelo
    subset = df_filt[df_filt['modelo'] == modelo]
    if len(subset) > 1:
        embeddings = np.stack(subset['embedding'].apply(ast.literal_eval).values)
        n_samples = len(subset)
        perplexity = min(30, max(2, n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        emb_2d = tsne.fit_transform(embeddings)
        fig = px.scatter(x=emb_2d[:,0], y=emb_2d[:,1], text=subset['ficheiro'],
                         title=f"t-SNE dos embeddings ({modelo})")
        fig.show()
        display(Markdown("Se vês grupos bem separados, o modelo está a criar embeddings discriminativos."))
    else:
        display(Markdown("Precisas de pelo menos 2 embeddings para t-SNE."))

# 7. Pesquisa vetorial simulada (top-5 mais próximos)
if "embedding" in df_filt.columns:
    display(Markdown("### Pesquisa Vetorial Simulada (top-5 mais próximos)"))
    modelo_sel = modelos[0]  # ou outro modelo
    df_model = df_filt[df_filt['modelo'] == modelo_sel].reset_index(drop=True)
    idx = 0  # índice do ficheiro para query
    emb_query = np.array(ast.literal_eval(df_model.iloc[idx]['embedding']))
    embeddings = np.stack(df_model['embedding'].apply(ast.literal_eval).values)
    dists = np.linalg.norm(embeddings - emb_query, axis=1)
    k = 5
    topk_idx = np.argsort(dists)[:k]
    display(Markdown(f"**Top-{k} mais próximos para o ficheiro:** {df_model.iloc[idx]['ficheiro']}"))
    display(df_model.iloc[topk_idx][['ficheiro', 'modelo', 'tempo_extracao', 'memoria_peak_mb']])

# 8. Pesquisa vetorial real no Milvus (opcional, requer pymilvus e Milvus ativo)
# from pymilvus import MilvusClient
# client = MilvusClient(uri="http://localhost:19530")
# collection_name = "audio_wav2vec2"  # exemplo
# embedding_query = emb_query.tolist()
# result = client.search(
#     collection_name=collection_name,
#     data=[embedding_query],
#     limit=k,
#     output_fields=["id", "filename", "title", "duration", "authors", "genre"]
# )
# print(result)

# 9. Processamento de ficheiros de áudio externos (upload/processamento local)
display(Markdown("### Processamento de Ficheiros de Áudio Externos"))
import librosa

audio_path = "caminho/para/o/teu/audio.wav"  # Altera para o teu ficheiro
if os.path.exists(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    plt.figure(figsize=(12, 3))
    plt.plot(audio)
    plt.title("Forma de onda do áudio")
    plt.show()
    display(Markdown(f"**Duração:** {len(audio)/sr:.2f} segundos | **Sample rate:** {sr} Hz"))
else:
    display(Markdown("Ficheiro de áudio não encontrado. Altera o caminho para testar."))

# 10. Mostrar logs do benchmark (se existir)
log_path = os.path.join(benchmark_folder, "benchmark.log")
if os.path.exists(log_path):
    with open(log_path) as f:
        log_content = f.read()
    display(Markdown("#### Log do Benchmark"))
    print(log_content)
else:
    display(Markdown("Log file não encontrado."))

# 11. Descrição geral
display(Markdown("""
---
### Descrição Geral

Este dashboard foi desenvolvido para analisar, comparar e explorar o desempenho de diferentes modelos de embeddings de áudio em tarefas de benchmark.  
Permite visualizar estatísticas, gráficos interativos, recursos utilizados e realizar pesquisas vetoriais tanto simuladas como reais na base de dados Milvus.

**O que significa "k mais próximos"?**  
Quando realizas uma pesquisa vetorial (simulada ou real), o sistema procura, para um determinado ficheiro de áudio (query), os `k` ficheiros cujos embeddings são mais próximos do embedding da query, segundo uma métrica de distância (normalmente L2 ou similaridade de cosseno).  
**Na prática, isto significa que os "k mais próximos" correspondem às músicas/áudios mais parecidos com o ficheiro escolhido, de acordo com o modelo de embeddings selecionado.**  
Esta funcionalidade é útil para tarefas como recomendação, deteção de duplicados, agrupamento de músicas semelhantes, entre outras aplicações em processamento de áudio.

Explora os resultados, testa diferentes modelos e compreende como cada abordagem representa e diferencia os ficheiros de áudio!
"""))