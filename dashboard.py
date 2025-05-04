import streamlit as st
import pandas as pd
import os
import glob
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Dashboard Interativo de Benchmarks de Áudio")

st.markdown("""
Este dashboard permite analisar e comparar o desempenho de diferentes modelos de embeddings de áudio.
Aqui podes explorar os resultados, gráficos, recursos usados e até simular pesquisas vetoriais.
""")

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

# 4. Gráficos interativos com Plotly
st.header("Gráficos Interativos")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Tempo de Extração", "Tempo de Inserção", "Memória Pico", 
    "Scatterplots", "Boxplot", "Evolução Recursos"
])

with tab1:
    st.subheader("Tempo de Extração por Modelo")
    fig = px.bar(df, x="modelo", y="tempo_extracao", color="modelo", barmode="group",
                 title="Tempo de extração por modelo")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quanto menor, melhor.")

with tab2:
    st.subheader("Tempo de Inserção por Modelo")
    fig = px.bar(df, x="modelo", y="tempo_insercao", color="modelo", barmode="group",
                 title="Tempo de inserção por modelo")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quanto menor, melhor.")

with tab3:
    st.subheader("Pico de Memória RAM por Modelo")
    fig = px.bar(df, x="modelo", y="memoria_peak_mb", color="modelo", barmode="group",
                 title="Pico de memória RAM por modelo")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Quanto menor, melhor.")

with tab4:
    st.subheader("Scatterplots")
    scatter_type = st.selectbox("Tipo de scatter", [
        "Tempo de extração vs Tamanho do ficheiro",
        "Correlação duração vs tempo de extração"
    ])
    if scatter_type == "Tempo de extração vs Tamanho do ficheiro":
        fig = px.scatter(df, x="tamanho_audio_mb", y="tempo_extracao", color="modelo",
                         title="Tempo de extração vs Tamanho do ficheiro de áudio",
                         labels={"tamanho_audio_mb": "Tamanho do ficheiro (MB)", "tempo_extracao": "Tempo de extração (s)"})
        st.info("Procura relação entre tamanho do ficheiro e tempo de extração.")
    else:
        fig = px.scatter(df, x="duracao_audio_s", y="tempo_extracao", color="modelo",
                         title="Correlação entre duração do áudio e tempo de extração",
                         labels={"duracao_audio_s": "Duração do áudio (s)", "tempo_extracao": "Tempo de extração (s)"})
        st.info("Procura relação entre duração do áudio e tempo de extração.")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Boxplot do Tempo de Extração por Modelo")
    fig = px.box(df, x="modelo", y="tempo_extracao", color="modelo",
                 title="Boxplot do tempo de extração por modelo")
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Evolução dos Recursos")
    recurso = st.selectbox("Recurso", ["cpu_percent_after", "ram_percent_after", "gpu_usage"])
    fig = go.Figure()
    for modelo in df['modelo'].unique():
        subset = df[df['modelo'] == modelo]
        fig.add_trace(go.Scatter(x=subset.index, y=subset[recurso], mode='lines+markers', name=modelo))
    fig.update_layout(title=f"Evolução do uso de {recurso.replace('_', ' ').upper()}",
                      xaxis_title="Processamento (ordem dos ficheiros)",
                      yaxis_title=f"{recurso.replace('_', ' ').capitalize()} (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Procura picos ou tendências de crescimento.")

