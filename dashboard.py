import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

st.set_page_config(layout="wide")
st.title("Dashboard Interativo de Benchmarks de Áudio")

# 1. Escolher pasta do benchmark
folders = sorted(glob.glob("benchmarks/benchmark_*"))
benchmark_folder = st.sidebar.selectbox("Escolhe o benchmark", folders)
st.sidebar.write(f"Pasta selecionada: {benchmark_folder}")

# 2. Carregar dados
csv_path = os.path.join(benchmark_folder, "resultados_benchmark.csv")
if not os.path.exists(csv_path):
    st.error("CSV de resultados não encontrado!")
    st.stop()
df = pd.read_csv(csv_path)

# 3. Mostrar tabela de dados
st.header("Tabela de Resultados")
st.dataframe(df)
st.download_button("Download CSV", df.to_csv(index=False), file_name="resultados_benchmark.csv")

# 4. Mostrar todos os gráficos PNG gerados
st.header("Gráficos do Benchmark")
pngs = sorted([f for f in os.listdir(benchmark_folder) if f.endswith(".png")])
for png in pngs:
    st.subheader(png.replace(".png", "").replace("_", " ").capitalize())
    st.image(os.path.join(benchmark_folder, png), use_column_width=True)

# 5. Visualização t-SNE dos embeddings (opcional)
if "embedding" in df.columns:
    st.header("Visualização t-SNE dos Embeddings")
    modelo = st.selectbox("Escolhe o modelo para t-SNE", df['modelo'].unique())
    subset = df[df['modelo'] == modelo]
    if len(subset) > 1:
        from sklearn.manifold import TSNE
        embeddings = np.stack(subset['embedding'].apply(eval).values)
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots()
        ax.scatter(emb_2d[:,0], emb_2d[:,1])
        for i, fname in enumerate(subset['ficheiro']):
            ax.annotate(fname, (emb_2d[i,0], emb_2d[i,1]), fontsize=8)
        st.pyplot(fig)
    else:
        st.info("Precisas de pelo menos 2 embeddings para t-SNE.")

# 6. Mostrar logs
log_path = os.path.join(benchmark_folder, "benchmark.log")
if os.path.exists(log_path):
    st.header("Logs do Benchmark")
    with open(log_path) as f:
        st.text(f.read())
else:
    st.info("Log file não encontrado.")

# 7. Pesquisa vetorial interativa (opcional)
if "embedding" in df.columns:
    st.header("Pesquisa Vetorial Simulada (top-5 mais próximos)")
    idx = st.selectbox("Escolhe um áudio para pesquisar", range(len(df)), format_func=lambda i: df.iloc[i]['ficheiro'])
    emb_query = np.array(eval(df.iloc[idx]['embedding']))
    dists = np.linalg.norm(np.stack(df['embedding'].apply(eval).values) - emb_query, axis=1)
    top5_idx = np.argsort(dists)[:5]
    st.write("Top-5 mais próximos:")
    st.table(df.iloc[top5_idx][['ficheiro', 'modelo', 'tempo_extracao', 'memoria_peak_mb']])

st.success("Dashboard carregado! Explora todos os gráficos e dados do benchmark.")