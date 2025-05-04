import streamlit as st
import pandas as pd
import os
import glob
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymilvus import MilvusClient
import ast

st.set_page_config(layout="wide")
st.title("Dashboard Interativo de Benchmarks de Áudio")

st.markdown("""
Este dashboard permite analisar e comparar o desempenho de diferentes modelos de embeddings de áudio.
Aqui podes explorar os resultados, gráficos, recursos usados e até simular pesquisas vetoriais.
""")

@st.cache_resource
def get_milvus_client():
    return MilvusClient(uri="http://localhost:19530")

# 1. Comparação entre benchmarks (lado a lado)
folders = sorted(glob.glob("benchmarks/benchmark_*"))
col1, col2 = st.columns(2)
with col1:
    benchmark_folder = st.selectbox("Escolhe o benchmark (A)", folders, key="benchA")
with col2:
    benchmark_folder2 = st.selectbox("Comparar com benchmark (B) (opcional)", ["Nenhum"] + folders, key="benchB")

@st.cache_data
def load_csv(csv_path):
    return pd.read_csv(csv_path)

# 2. Carregar dados (com cache)
csv_path = os.path.join(benchmark_folder, "resultados_benchmark.csv")
if not os.path.exists(csv_path):
    st.error("CSV de resultados não encontrado!")
    st.stop()
df = load_csv(csv_path)

if benchmark_folder2 != "Nenhum":
    csv_path2 = os.path.join(benchmark_folder2, "resultados_benchmark.csv")
    if os.path.exists(csv_path2):
        df2 = load_csv(csv_path2)
    else:
        df2 = None
else:
    df2 = None

# 3. Filtros dinâmicos
with st.expander("Filtros avançados", expanded=False):
    modelos = st.multiselect("Filtrar modelos", df['modelo'].unique(), default=list(df['modelo'].unique()))
    tempo_min, tempo_max = st.slider("Filtrar tempo de extração (s)", float(df['tempo_extracao'].min()), float(df['tempo_extracao'].max()), (float(df['tempo_extracao'].min()), float(df['tempo_extracao'].max())))
    df = df[df['modelo'].isin(modelos) & df['tempo_extracao'].between(tempo_min, tempo_max)]
    if df2 is not None:
        modelos2 = st.multiselect("Filtrar modelos (B)", df2['modelo'].unique(), default=list(df2['modelo'].unique()), key="mod2")
        df2 = df2[df2['modelo'].isin(modelos2)]

# 4. Sumário estatístico
with st.expander("Sumário Estatístico", expanded=True):
    st.write("**Estatísticas principais (A):**")
    st.dataframe(df.describe())
    if df2 is not None:
        st.write("**Estatísticas principais (B):**")
        st.dataframe(df2.describe())

# 5. Mostrar tabela de dados
st.header("Tabela de Resultados")
st.dataframe(df)
st.download_button("Download CSV", df.to_csv(index=False), file_name="resultados_benchmark.csv")

# 6. Seleção de métricas e eixos para gráficos
st.header("Gráficos Interativos")
metricas = ["tempo_extracao", "tempo_insercao", "memoria_peak_mb", "cpu_percent_after", "ram_percent_after", "gpu_usage"]
metrica_bar = st.selectbox("Métrica para gráfico de barras", metricas, index=0)
metrica_box = st.selectbox("Métrica para boxplot", metricas, index=0)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Barras", "Boxplot", "Scatter Personalizado", 
    "Scatterplots", "Evolução Recursos", "3D", "Download Gráfico"
])

with tab1:
    st.subheader(f"{metrica_bar.replace('_',' ').capitalize()} por Modelo")
    fig = px.bar(df, x="modelo", y=metrica_bar, color="modelo", barmode="group",
                 title=f"{metrica_bar.replace('_',' ').capitalize()} por modelo")
    if df2 is not None:
        fig2 = px.bar(df2, x="modelo", y=metrica_bar, color="modelo", barmode="group",
                      title=f"{metrica_bar.replace('_',' ').capitalize()} por modelo (B)")
        fig.add_traces(fig2.data)
    st.plotly_chart(fig, use_container_width=True, key="barplot")
    st.help("Quanto menor, melhor para tempos e recursos.")

with tab2:
    st.subheader(f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por Modelo")
    fig = px.box(df, x="modelo", y=metrica_box, color="modelo",
                 title=f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por modelo")
    st.plotly_chart(fig, use_container_width=True, key="boxplotA")
    if df2 is not None:
        fig2 = px.box(df2, x="modelo", y=metrica_box, color="modelo",
                      title=f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por modelo (B)")
        st.plotly_chart(fig2, use_container_width=True, key="boxplotB")
    st.help("Boxplots mostram a distribuição e outliers.")

with tab3:
    st.subheader("Scatterplot Personalizado")
    x_axis = st.selectbox("Eixo X", df.columns, index=df.columns.get_loc("tamanho_audio_mb"))
    y_axis = st.selectbox("Eixo Y", df.columns, index=df.columns.get_loc("tempo_extracao"))
    fig = px.scatter(df, x=x_axis, y=y_axis, color="modelo", title=f"{y_axis} vs {x_axis}")
    st.plotly_chart(fig, use_container_width=True, key="scatter_custom")
    st.help("Escolhe qualquer par de variáveis para explorar relações.")

with tab4:
    st.subheader("Scatterplots Pré-definidos")
    scatter_type = st.selectbox("Tipo de scatter", [
        "Tempo de extração vs Tamanho do ficheiro",
        "Correlação duração vs tempo de extração"
    ])
    if scatter_type == "Tempo de extração vs Tamanho do ficheiro":
        fig = px.scatter(df, x="tamanho_audio_mb", y="tempo_extracao", color="modelo",
                         title="Tempo de extração vs Tamanho do ficheiro de áudio",
                         labels={"tamanho_audio_mb": "Tamanho do ficheiro (MB)", "tempo_extracao": "Tempo de extração (s)"})
        st.info("Procura relação entre tamanho do ficheiro e tempo de extração.")
        st.plotly_chart(fig, use_container_width=True, key="scatter1")
    else:
        fig = px.scatter(df, x="duracao_audio_s", y="tempo_extracao", color="modelo",
                         title="Correlação entre duração do áudio e tempo de extração",
                         labels={"duracao_audio_s": "Duração do áudio (s)", "tempo_extracao": "Tempo de extração (s)"})
        st.info("Procura relação entre duração do áudio e tempo de extração.")
        st.plotly_chart(fig, use_container_width=True, key="scatter2")

with tab5:
    st.subheader("Evolução dos Recursos")
    recurso = st.selectbox("Recurso", ["cpu_percent_after", "ram_percent_after", "gpu_usage"])
    fig = go.Figure()
    for modelo in df['modelo'].unique():
        subset = df[df['modelo'] == modelo]
        fig.add_trace(go.Scatter(x=subset.index, y=subset[recurso], mode='lines+markers', name=modelo))
    fig.update_layout(title=f"Evolução do uso de {recurso.replace('_', ' ').upper()}",
                      xaxis_title="Processamento (ordem dos ficheiros)",
                      yaxis_title=f"{recurso.replace('_', ' ').capitalize()} (%)")
    st.plotly_chart(fig, use_container_width=True, key="evolucao")
    st.help("Procura picos ou tendências de crescimento.")

with tab6:
    st.subheader("Scatterplot 3D Interativo")
    x3d = st.selectbox("Eixo X (3D)", df.columns, index=df.columns.get_loc("tempo_extracao"))
    y3d = st.selectbox("Eixo Y (3D)", df.columns, index=df.columns.get_loc("memoria_peak_mb"))
    z3d = st.selectbox("Eixo Z (3D)", df.columns, index=df.columns.get_loc("tamanho_audio_mb"))
    fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color="modelo", title=f"3D: {x3d} vs {y3d} vs {z3d}")
    st.plotly_chart(fig, use_container_width=True, key="scatter3d")
    st.help("Explora relações tridimensionais entre variáveis.")

with tab7:
    st.subheader("Download de Gráfico")
    tipo_grafico = st.selectbox("Escolhe o gráfico para download", ["Barra", "Boxplot", "Scatter 3D"])
    if tipo_grafico == "Barra":
        fig = px.bar(df, x="modelo", y=metrica_bar, color="modelo", barmode="group",
                     title=f"{metrica_bar.replace('_',' ').capitalize()} por modelo")
        st.plotly_chart(fig, use_container_width=True, key="download_bar")
    elif tipo_grafico == "Boxplot":
        fig = px.box(df, x="modelo", y=metrica_box, color="modelo",
                     title=f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por modelo")
        st.plotly_chart(fig, use_container_width=True, key="download_box")
    else:
        fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color="modelo", title=f"3D: {x3d} vs {y3d} vs {z3d}")
        st.plotly_chart(fig, use_container_width=True, key="download_3d")
    st.download_button("Download PNG", fig.to_image(format="png"), file_name="grafico.png")

# 7. Visualização t-SNE dos embeddings (opcional)
if "embedding" in df.columns:
    with st.expander("Visualização t-SNE dos Embeddings", expanded=False):
        modelo = st.selectbox("Modelo para t-SNE", df['modelo'].unique())
        subset = df[df['modelo'] == modelo]
        if len(subset) > 1:
            from sklearn.manifold import TSNE
            embeddings = np.stack(subset['embedding'].apply(eval).values)
            n_samples = len(subset)
            perplexity = min(30, max(2, n_samples - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            emb_2d = tsne.fit_transform(embeddings)
            fig = px.scatter(x=emb_2d[:,0], y=emb_2d[:,1], text=subset['ficheiro'],
                             title=f"t-SNE dos embeddings ({modelo})")
            st.plotly_chart(fig, use_container_width=True, key="tsne")
            st.info("Se vês grupos bem separados, o modelo está a criar embeddings discriminativos.")
        else:
            st.info("Precisas de pelo menos 2 embeddings para t-SNE.")

# 8. Pesquisa vetorial interativa (simulada)
if "embedding" in df.columns:
    with st.expander("Pesquisa Vetorial Simulada (top-5 mais próximos)", expanded=False):
        modelos_sim = df['modelo'].unique()
        modelo_sel = st.selectbox("Modelo para pesquisa simulada", modelos_sim)
        df_model = df[df['modelo'] == modelo_sel].reset_index(drop=True)
        idx = st.selectbox("Escolhe um áudio para pesquisar", range(len(df_model)), format_func=lambda i: df_model.iloc[i]['ficheiro'])
        emb_query = np.array(eval(df_model.iloc[idx]['embedding']))
        try:
            embeddings = np.stack(df_model['embedding'].apply(eval).values)
        except Exception as e:
            st.error(f"Erro ao processar embeddings: {e}")
            st.stop()
        dists = np.linalg.norm(embeddings - emb_query, axis=1)
        top5_idx = np.argsort(dists)[:5]
        st.write("Top-5 mais próximos:")
        st.dataframe(df_model.iloc[top5_idx][['ficheiro', 'modelo', 'tempo_extracao', 'memoria_peak_mb']])
        st.help("Se os resultados fizerem sentido, o embedding é bom!")

# 9. Pesquisa vetorial real no Milvus
with st.expander("Pesquisa Vetorial Real no Milvus", expanded=True):
    st.markdown("""
    Pesquisa diretamente na base de dados Milvus.  
    Seleciona o modelo (coleção), escolhe um ficheiro de áudio compatível como query e vê os k mais próximos (com distâncias reais).
    """)
    client = get_milvus_client()
    modelos_disponiveis = [c.replace("audio_", "") for c in client.list_collections() if c.startswith("audio_")]
    modelo_milvus = st.selectbox("Modelo (coleção Milvus)", modelos_disponiveis)
    collection_name = f"audio_{modelo_milvus}"

    # Define a dimensão esperada para cada modelo
    dimensoes_modelos = {
        "ast": 768,
        "clap": 512,
        "yamnet": 1024,
        "openl3": 512,
        "vggish": 128,
        "wav2vec": 768,
    }
    dim_esperada = dimensoes_modelos.get(modelo_milvus, None)
    if dim_esperada is None:
        st.error(f"Dimensão desconhecida para o modelo '{modelo_milvus}'.")
        st.stop()

    # Só mostra ficheiros do modelo e dimensão correta
    ficheiros = df[(df['modelo'] == modelo_milvus) & (df['embedding'].apply(lambda x: len(ast.literal_eval(x)) == dim_esperada))]['ficheiro'].tolist()
    if not ficheiros:
        st.warning(f"Não há ficheiros com embeddings de dimensão {dim_esperada} para o modelo '{modelo_milvus}'.")
        st.stop()
    ficheiro_query = st.selectbox("Escolhe o ficheiro de áudio para pesquisar", ficheiros)
    row_query = df[(df['modelo'] == modelo_milvus) & (df['ficheiro'] == ficheiro_query)].iloc[0]
    embedding_query = row_query['embedding']
    if isinstance(embedding_query, str):
        embedding_query = ast.literal_eval(embedding_query)

    if len(embedding_query) != dim_esperada:
        st.error(f"O embedding selecionado não tem dimensão {dim_esperada}!")
        st.stop()

    k = st.slider("Número de vizinhos mais próximos (k)", 1, 20, 5)

    if st.button("Pesquisar no Milvus"):
        result = client.search(
            collection_name=collection_name,
            data=[embedding_query],
            limit=k,
            output_fields=["id", "vector"]
        )
        hits = result[0] if result and len(result) > 0 else []
        if hits:
            st.success(f"Top-{k} mais próximos para '{ficheiro_query}':")
            res_df = pd.DataFrame([
                {
                    "score": h.get("score"),
                    "id": h.get("id"),
                    "vector": h.get("vector")
                } for h in hits
            ])
            st.dataframe(res_df)
            fig = px.bar(res_df, x="id", y="score", color="score", title="Distância dos k mais próximos")
            st.plotly_chart(fig, use_container_width=True, key="milvus_knn")
            st.info("Quanto menor o score, mais próximo está o vetor.")
        else:
            st.warning("Nenhum resultado encontrado no Milvus para esta query.")
            st.write("Dimensão do embedding:", len(embedding_query))
            st.write("Primeiros 5 valores:", embedding_query[:5])

# 10. Mostrar logs (colapsável)
log_path = os.path.join(benchmark_folder, "benchmark.log")
with st.expander("Logs do Benchmark", expanded=False):
    if os.path.exists(log_path):
        with open(log_path) as f:
            st.text(f.read())
    else:
        st.info("Log file não encontrado.")

st.success("Dashboard carregado! Explora todos os gráficos e dados do benchmark.")

