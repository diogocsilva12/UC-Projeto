import streamlit as st
import pandas as pd
import os
import glob
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymilvus import MilvusClient
import ast

#Dashboard no streamlit

st.set_page_config(layout="wide")
st.title("Dashboard Interativo de Benchmarks de Áudio")

st.markdown("""
Este dashboard permite analisar e comparar o desempenho de diferentes modelos de embeddings de áudio.
Aqui podes explorar os resultados, gráficos, recursos usados e até simular pesquisas vetoriais.
""")

@st.cache_resource
def get_milvus_client():
    return MilvusClient(uri="http://localhost:19530")

# Organização visual: seleção e carregamento em colunas
col1, col2 = st.columns([2, 3])
with col1:
    try:
        folders = sorted(glob.glob("benchmarks/benchmark_*"))
        if not folders:
            st.error("Nenhum benchmark encontrado na pasta 'benchmarks'.")
            st.stop()
        benchmark_folder = st.selectbox("Escolhe o benchmark", folders)
        st.success("Benchmark selecionado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao listar benchmarks: {e}")
        st.stop()

with col2:
    try:
        @st.cache_data
        def load_csv(csv_path):
            return pd.read_csv(csv_path)
        csv_path = os.path.join(benchmark_folder, "resultados_benchmark.csv")
        if not os.path.exists(csv_path):
            st.error("CSV de resultados não encontrado!")
            st.stop()
        df = load_csv(csv_path)
        st.success("Dados carregados com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

# Filtros dinâmicos
with st.expander("Filtros avançados", expanded=False):
    try:
        modelos = st.multiselect("Filtrar modelos", df['modelo'].unique(), default=list(df['modelo'].unique()), help="Seleciona os modelos a analisar.")
        tempo_min, tempo_max = st.slider(
            "Filtrar tempo de extração (s)",
            float(df['tempo_extracao'].min()), float(df['tempo_extracao'].max()),
            (float(df['tempo_extracao'].min()), float(df['tempo_extracao'].max()))
        )
        df = df[df['modelo'].isin(modelos) & df['tempo_extracao'].between(tempo_min, tempo_max)]
        st.success("Filtros aplicados com sucesso!")
    except Exception as e:
        st.error(f"Erro ao aplicar filtros: {e}")

# Sumário estatístico
with st.expander("Sumário Estatístico", expanded=True):
    try:
        st.write("**Estatísticas principais:**")
        st.dataframe(df.describe())
        st.success("Sumário estatístico apresentado.")
    except Exception as e:
        st.error(f"Erro ao mostrar sumário estatístico: {e}")

# Tabela de dados
st.header("Tabela de Resultados")
st.markdown("""
**Descrição:**  
Esta tabela apresenta todos os resultados do benchmark, incluindo métricas de desempenho, recursos utilizados e informações dos ficheiros de áudio processados por cada modelo.  
Permite consultar, filtrar e exportar os dados para análise detalhada.
""")
try:
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), file_name="resultados_benchmark.csv")
    st.success("Tabela de resultados carregada.")
except Exception as e:
    st.error(f"Erro ao mostrar tabela de resultados: {e}")

# Seleção de métricas e eixos para gráficos
st.header("Gráficos Interativos")
st.markdown("""
**Descrição:**  
Aqui podes escolher quais métricas analisar e visualizar nos diferentes tipos de gráficos disponíveis.  
Seleciona a métrica de interesse para cada tipo de gráfico e explora o desempenho dos modelos de embeddings de áudio de forma visual e comparativa.
""")
try:
    metricas = ["tempo_extracao", "tempo_insercao", "memoria_peak_mb", "cpu_percent_after", "ram_percent_after", "gpu_usage"]
    metrica_bar = st.selectbox("Métrica para gráfico de barras", metricas, index=0, help="Escolhe a métrica para o gráfico de barras.")
    metrica_box = st.selectbox("Métrica para boxplot", metricas, index=0, help="Escolhe a métrica para o boxplot.")
    st.success("Métricas selecionadas.")
except Exception as e:
    st.error(f"Erro ao selecionar métricas: {e}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Barras", "Boxplot", "Scatter Personalizado", 
    "Scatterplots", "Evolução Recursos", "3D", "Download Gráfico",
    "Violin Plot", "Pairplot", "Heatmap Correlação"
])

with tab1:
    st.subheader(f"{metrica_bar.replace('_',' ').capitalize()} por Modelo")
    try:
        fig = px.bar(df, x="modelo", y=metrica_bar, color="modelo", barmode="group",
                     title=f"{metrica_bar.replace('_',' ').capitalize()} por modelo")
        st.plotly_chart(fig, use_container_width=True, key="barplot")
        st.success("Gráfico de barras gerado com sucesso.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O gráfico de barras permite comparar rapidamente o valor médio de uma métrica (por exemplo, tempo de extração, uso de memória, etc.) entre diferentes modelos de embeddings de áudio.  
    Cada barra representa um modelo e a sua altura indica o valor da métrica selecionada.  
    Este tipo de gráfico é útil para identificar qual modelo é mais eficiente ou consome menos recursos.
    """)
    st.help("Quanto menor, melhor para tempos e recursos.")

with tab2:
    st.subheader(f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por Modelo")
    try:
        fig = px.box(df, x="modelo", y=metrica_box, color="modelo",
                     title=f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por modelo")
        st.plotly_chart(fig, use_container_width=True, key="boxplotA")
        st.success("Boxplot gerado com sucesso.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O boxplot mostra a distribuição dos valores de uma métrica para cada modelo, incluindo mediana, quartis e possíveis outliers.  
    Permite perceber a variabilidade dos resultados, identificar valores atípicos e comparar a consistência dos modelos.  
    Útil para analisar dispersão e estabilidade do desempenho.
    """)
    st.help("Boxplots mostram a distribuição e outliers.")

with tab3:
    st.subheader("Scatterplot Personalizado")
    try:
        x_axis = st.selectbox("Eixo X", df.columns, index=df.columns.get_loc("tamanho_audio_mb"), help="Escolhe a variável para o eixo X.")
        y_axis = st.selectbox("Eixo Y", df.columns, index=df.columns.get_loc("tempo_extracao"), help="Escolhe a variável para o eixo Y.")
        fig = px.scatter(df, x=x_axis, y=y_axis, color="modelo", title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True, key="scatter_custom")
        st.success("Scatterplot personalizado gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O scatterplot personalizado permite explorar a relação entre duas variáveis à escolha.  
    Cada ponto representa um ficheiro de áudio processado por um modelo, e as cores distinguem os modelos.  
    Este gráfico é útil para identificar correlações, tendências ou padrões entre diferentes métricas.
    """)
    st.help("Escolhe qualquer par de variáveis para explorar relações.")

with tab4:
    st.subheader("Scatterplots Pré-definidos")
    try:
        scatter_type = st.selectbox("Tipo de scatter", [
            "Tempo de extração vs Tamanho do ficheiro",
            "Correlação duração vs tempo de extração"
        ], help="Escolhe o tipo de scatterplot.")
        if scatter_type == "Tempo de extração vs Tamanho do ficheiro":
            fig = px.scatter(df, x="tamanho_audio_mb", y="tempo_extracao", color="modelo",
                             title="Tempo de extração vs Tamanho do ficheiro de áudio",
                             labels={"tamanho_audio_mb": "Tamanho do ficheiro (MB)", "tempo_extracao": "Tempo de extração (s)"})
            st.info("Procura relação entre tamanho do ficheiro e tempo de extração.")
            st.markdown("""
            **Descrição:**  
            Este scatterplot mostra como o tempo de extração varia em função do tamanho do ficheiro de áudio para cada modelo.  
            Ajuda a perceber se ficheiros maiores demoram mais tempo a ser processados e se há diferenças entre modelos.
            """)
            st.plotly_chart(fig, use_container_width=True, key="scatter1")
        else:
            fig = px.scatter(df, x="duracao_audio_s", y="tempo_extracao", color="modelo",
                             title="Correlação entre duração do áudio e tempo de extração",
                             labels={"duracao_audio_s": "Duração do áudio (s)", "tempo_extracao": "Tempo de extração (s)"})
            st.info("Procura relação entre duração do áudio e tempo de extração.")
            st.markdown("""
            **Descrição:**  
            Este scatterplot mostra a relação entre a duração do áudio e o tempo de extração do embedding.  
            Permite analisar se áudios mais longos levam mais tempo a ser processados e comparar o comportamento dos modelos.
            """)
            st.plotly_chart(fig, use_container_width=True, key="scatter2")
        st.success("Scatterplot pré-definido gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")

with tab5:
    st.subheader("Evolução dos Recursos")
    try:
        recurso = st.selectbox("Recurso", ["cpu_percent_after", "ram_percent_after", "gpu_usage"], help="Escolhe o recurso a visualizar.")
        fig = go.Figure()
        for modelo in df['modelo'].unique():
            subset = df[df['modelo'] == modelo]
            fig.add_trace(go.Scatter(x=subset.index, y=subset[recurso], mode='lines+markers', name=modelo))
        fig.update_layout(title=f"Evolução do uso de {recurso.replace('_', ' ').upper()}",
                          xaxis_title="Processamento (ordem dos ficheiros)",
                          yaxis_title=f"{recurso.replace('_', ' ').capitalize()} (%)")
        st.plotly_chart(fig, use_container_width=True, key="evolucao")
        st.success("Gráfico de evolução dos recursos gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown(f"""
    **Descrição:**  
    Este gráfico mostra a evolução temporal do uso de {recurso.replace('_', ' ')} durante o processamento dos ficheiros de áudio para cada modelo.  
    Permite identificar picos de consumo, tendências de crescimento e comparar a eficiência dos modelos ao longo do tempo.
    """)
    st.help("Procura picos ou tendências de crescimento.")

with tab6:
    st.subheader("Scatterplot 3D Interativo")
    try:
        x3d = st.selectbox("Eixo X (3D)", df.columns, index=df.columns.get_loc("tempo_extracao"), help="Escolhe a variável para o eixo X (3D).")
        y3d = st.selectbox("Eixo Y (3D)", df.columns, index=df.columns.get_loc("memoria_peak_mb"), help="Escolhe a variável para o eixo Y (3D).")
        z3d = st.selectbox("Eixo Z (3D)", df.columns, index=df.columns.get_loc("tamanho_audio_mb"), help="Escolhe a variável para o eixo Z (3D).")
        fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color="modelo", title=f"3D: {x3d} vs {y3d} vs {z3d}")
        st.plotly_chart(fig, use_container_width=True, key="scatter3d")
        st.success("Scatterplot 3D gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O scatterplot 3D permite visualizar simultaneamente três variáveis para cada ficheiro de áudio e modelo.  
    É útil para identificar padrões multidimensionais, agrupamentos ou outliers que não seriam visíveis em 2D.
    """)
    st.help("Explora relações tridimensionais entre variáveis.")

with tab7:
    st.subheader("Download de Gráfico")
    try:
        tipo_grafico = st.selectbox("Escolhe o gráfico para download", ["Barra", "Boxplot", "Scatter 3D"], help="Escolhe o tipo de gráfico para download.")
        if tipo_grafico == "Barra":
            fig = px.bar(df, x="modelo", y=metrica_bar, color="modelo", barmode="group",
                         title=f"{metrica_bar.replace('_',' ').capitalize()} por modelo")
            st.plotly_chart(fig, use_container_width=True, key="download_bar")
            st.markdown("""
            **Descrição:**  
            Permite gerar e descarregar um gráfico de barras para comparar rapidamente os modelos em relação a uma métrica selecionada.
            """)
        elif tipo_grafico == "Boxplot":
            fig = px.box(df, x="modelo", y=metrica_box, color="modelo",
                         title=f"Boxplot de {metrica_box.replace('_',' ').capitalize()} por modelo")
            st.plotly_chart(fig, use_container_width=True, key="download_box")
            st.markdown("""
            **Descrição:**  
            Permite gerar e descarregar um boxplot para analisar a distribuição dos valores de uma métrica por modelo.
            """)
        else:
            fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color="modelo", title=f"3D: {x3d} vs {y3d} vs {z3d}")
            st.plotly_chart(fig, use_container_width=True, key="download_3d")
            st.markdown("""
            **Descrição:**  
            Permite gerar e descarregar um gráfico 3D para explorar relações entre três variáveis ao mesmo tempo.
            """)
        st.download_button("Download PNG", fig.to_image(format="png"), file_name="grafico.png")
        st.success("Gráfico pronto para download.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")

with tab8:
    st.subheader("Violin Plot dos Tempos de Extração")
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        fig_violin, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x="modelo", y="tempo_extracao", ax=ax)
        ax.set_title("Distribuição dos tempos de extração por modelo (Violin Plot)")
        st.pyplot(fig_violin)
        st.success("Violin plot gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O violin plot mostra a distribuição da métrica (tempo de extração) para cada modelo, combinando características de boxplot e densidade.  
    Permite visualizar a forma da distribuição, simetrias, multimodalidade e outliers.
    """)
    st.help("O violin plot mostra a densidade e dispersão dos tempos de extração por modelo.")

with tab9:
    st.subheader("Pairplot das Métricas Principais")
    import seaborn as sns
    import matplotlib.pyplot as plt
    try:
        pairplot_fig = sns.pairplot(df, hue="modelo", vars=["tempo_extracao", "memoria_peak_mb", "tamanho_audio_mb", "duracao_audio_s"])
        st.pyplot(pairplot_fig)
        st.success("Pairplot gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O pairplot apresenta múltiplos scatterplots entre todas as combinações das principais métricas, separados por modelo.  
    É útil para explorar relações, correlações e possíveis agrupamentos entre variáveis.
    """)
    st.help("Pairplot permite explorar relações entre várias métricas ao mesmo tempo.")

with tab10:
    st.subheader("Mapa de Correlação entre Métricas")
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        corr = df[["tempo_extracao", "memoria_peak_mb", "tamanho_audio_mb", "duracao_audio_s"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Mapa de Correlação entre Métricas")
        st.pyplot(fig_corr)
        st.success("Heatmap de correlação gerado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
    st.markdown("""
    **Descrição:**  
    O heatmap de correlação mostra o grau de relação linear entre as principais métricas do benchmark.  
    Valores próximos de 1 ou -1 indicam forte correlação positiva ou negativa, respetivamente.  
    Ajuda a identificar dependências e redundâncias entre variáveis.
    """)
    st.help("O heatmap mostra a correlação entre as principais métricas num só gráfico.")

# 7. Visualização t-SNE dos embeddings (opcional)
if "embedding" in df.columns:
    with st.expander("Visualização t-SNE dos Embeddings", expanded=False):
        try:
            modelo = st.selectbox("Modelo para t-SNE", df['modelo'].unique(), help="Escolhe o modelo para visualizar embeddings em 2D.")
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
                st.success("t-SNE gerado.")
            else:
                st.info("Precisas de pelo menos 2 embeddings para t-SNE.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# 8. Pesquisa vetorial interativa (simulada)
if "embedding" in df.columns:
    with st.expander("Pesquisa Vetorial Simulada (top-5 mais próximos)", expanded=False):
        try:
            modelos_sim = df['modelo'].unique()
            modelo_sel = st.selectbox("Modelo para pesquisa simulada", modelos_sim, help="Escolhe o modelo para a pesquisa simulada.")
            df_model = df[df['modelo'] == modelo_sel].reset_index(drop=True)
            idx = st.selectbox("Escolhe um áudio para pesquisar", range(len(df_model)), format_func=lambda i: df_model.iloc[i]['ficheiro'])
            emb_query = np.array(eval(df_model.iloc[idx]['embedding']))
            embeddings = np.stack(df_model['embedding'].apply(eval).values)
            dists = np.linalg.norm(embeddings - emb_query, axis=1)
            top5_idx = np.argsort(dists)[:5]
            st.write("Top-5 mais próximos:")
            st.dataframe(df_model.iloc[top5_idx][['ficheiro', 'modelo', 'tempo_extracao', 'memoria_peak_mb']])
            st.success("Pesquisa simulada concluída.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
        st.help("Se os resultados fizerem sentido, o embedding é bom!")

# 9. Pesquisa vetorial real no Milvus
with st.expander("Pesquisa Vetorial Real no Milvus", expanded=True):
    st.markdown("""
    Pesquisa diretamente na base de dados Milvus.  
    Seleciona o modelo (coleção), escolhe um ficheiro de áudio compatível como query e vê os k mais próximos (com distâncias reais).
    """)
    try:
        client = get_milvus_client()
        modelos_disponiveis = [c.replace("audio_", "") for c in client.list_collections() if c.startswith("audio_")]
        modelo_milvus = st.selectbox("Modelo (coleção Milvus)", modelos_disponiveis, help="Escolhe o modelo (coleção) no Milvus.")
        collection_name = f"audio_{modelo_milvus}"

        # Define a dimensão esperada para cada modelo
        dimensoes_modelos = {
            "ast": 768,
            "clap": 512,
            "yamnet": 1024,
            "openl3": 512,
            "vggish": 128,
            "wav2vec": 768,
            "wav2vec2": 768,
        }
        dim_esperada = dimensoes_modelos.get(modelo_milvus, None)
        if dim_esperada is None:
            st.error(f"Dimensão desconhecida para o modelo '{modelo_milvus}'.")
            st.stop()

        ficheiros = df[(df['modelo'] == modelo_milvus) & (df['embedding'].apply(lambda x: len(ast.literal_eval(x)) == dim_esperada))]['ficheiro'].tolist()
        if not ficheiros:
            st.warning(f"Não há ficheiros com embeddings de dimensão {dim_esperada} para o modelo '{modelo_milvus}'.")
            st.stop()
        ficheiro_query = st.selectbox("Escolhe o ficheiro de áudio para pesquisar", ficheiros, help="Escolhe o ficheiro de áudio para a query.")
        row_query = df[(df['modelo'] == modelo_milvus) & (df['ficheiro'] == ficheiro_query)].iloc[0]
        embedding_query = row_query['embedding']
        if isinstance(embedding_query, str):
            embedding_query = ast.literal_eval(embedding_query)

        if len(embedding_query) != dim_esperada:
            st.error(f"O embedding selecionado não tem dimensão {dim_esperada}!")
            st.stop()

        k = st.slider("Número de vizinhos mais próximos (k)", 1, 20, 5, help="Número de vizinhos mais próximos a pesquisar.")

        client = get_milvus_client()
        num_vetores = client.get_collection_stats(collection_name)["row_count"]
        st.info(f"A coleção '{collection_name}' tem {num_vetores} vetores.")

        if num_vetores < k:
            st.warning(f"A coleção só tem {num_vetores} vetores. Só é possível devolver até esse número de resultados.")

        if st.button("Pesquisar no Milvus"):
            try:
                result = client.search(
                    collection_name=collection_name,
                    data=[embedding_query],
                    limit=k,
                    output_fields=["id", "vector"]
                )
                hits = result[0] if result and len(result) > 0 else []
                if hits:
                    st.success(f"Top-{len(hits)} mais próximos para '{ficheiro_query}':")
                    res_df = pd.DataFrame([
                        {
                            "distance": h.get("distance"),
                            "id": h.get("id"),
                            "vector": h.get("entity", {}).get("vector")
                        } for h in hits
                    ])
                    st.dataframe(res_df)
                    fig = px.bar(res_df, x="id", y="distance", color="distance", title="Distância dos k mais próximos")
                    st.plotly_chart(fig, use_container_width=True, key="milvus_knn")
                    st.info("Quanto menor a distância, mais próximo está o vetor.")
                else:
                    st.warning("Nenhum resultado encontrado no Milvus para esta query.")
                    st.write("Dimensão do embedding:", len(embedding_query))
                    st.write("Primeiros 5 valores:", embedding_query[:5])
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
    except Exception as e:
        st.error(f"Erro na configuração da pesquisa Milvus: {e}")

# Pesquisa Ad-hoc: Upload de Áudio
with st.expander("Pesquisa Ad-hoc: Upload de Áudio", expanded=False):
    st.write("Faz upload de um ficheiro de áudio para pesquisar no Milvus com o modelo selecionado.")
    try:
        uploaded_file = st.file_uploader("Seleciona um ficheiro de áudio", type=["wav", "mp3", "flac", "ogg"])
        if uploaded_file is not None:
            # Validação do tipo de ficheiro
            if not uploaded_file.name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                st.error("Tipo de ficheiro não suportado.")
                st.stop()
            # Validação do tamanho do ficheiro
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
                st.error("Ficheiro demasiado grande. Limite: 50MB.")
                st.stop()

            import tempfile
            import librosa
            import numpy as np

            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]).name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("Ficheiro carregado com sucesso!")

            modelo_upload = st.selectbox("Modelo para extrair embedding", [
                "wav2vec2", "vggish", "openl3", "yamnet", "clap", "ast"
            ], help="Escolhe o modelo para extrair o embedding do áudio.")
            embedding = None
            try:
                if modelo_upload == "wav2vec2":
                    from transformers import Wav2Vec2Processor, Wav2Vec2Model
                    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                    audio, sr = librosa.load(temp_path, sr=16000)
                    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                    import torch
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                elif modelo_upload == "clap":
                    from transformers import ClapProcessor, ClapModel
                    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
                    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
                    audio, sr = librosa.load(temp_path, sr=48000)
                    inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")
                    import torch
                    with torch.no_grad():
                        embeddings = model.get_audio_features(**inputs)
                    embedding = embeddings[0].numpy()
                elif modelo_upload == "ast":
                    from transformers import ASTFeatureExtractor, ASTModel
                    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                    audio, sr = librosa.load(temp_path, sr=16000)
                    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
                    import torch
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                elif modelo_upload == "vggish":
                    import tensorflow_hub as hub
                    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
                    audio, sr = librosa.load(temp_path, sr=16000)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    embedding = vggish_model(audio).numpy()
                    if embedding.ndim > 1:
                        embedding = np.mean(embedding, axis=0)
                elif modelo_upload == "yamnet":
                    import tensorflow_hub as hub
                    import tensorflow as tf
                    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
                    audio, sr = librosa.load(temp_path, sr=16000)
                    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
                    scores, embeddings, spectrogram = yamnet_model(waveform)
                    embedding = np.mean(embeddings.numpy(), axis=0)
                elif modelo_upload == "openl3":
                    import openl3
                    audio, sr = librosa.load(temp_path, sr=48000)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
                    if emb.shape[0] > 1:
                        embedding = np.mean(emb, axis=0)
                    else:
                        embedding = emb.squeeze()
                else:
                    st.error("Modelo não suportado.")
            except Exception as e:
                st.error(f"Erro ao extrair embedding: {e}")
                embedding = None

            if embedding is not None:
                st.success(f"Embedding extraído com sucesso! Dimensão: {len(embedding)}")
                client = get_milvus_client()
                collection_name = f"audio_{modelo_upload}"
                k = st.slider("Número de vizinhos mais próximos (k)", 1, 20, 5, key="k_upload")
                if st.button("Pesquisar no Milvus com este embedding"):
                    try:
                        result = client.search(
                            collection_name=collection_name,
                            data=[embedding.tolist()],
                            limit=k,
                            output_fields=["id", "filename", "title", "duration", "authors", "genre"]
                        )
                        hits = result[0] if result and len(result) > 0 else []
                        if hits:
                            st.success(f"Top-{k} mais próximos:")
                            res_df = pd.DataFrame([
                                {
                                    "distance": h.get("distance"),
                                    "id": h.get("id"),
                                    "filename": h.get("entity", {}).get("filename"),
                                    "title": h.get("entity", {}).get("title"),
                                    "duration": h.get("entity", {}).get("duration"),
                                    "authors": h.get("entity", {}).get("authors"),
                                    "genre": h.get("entity", {}).get("genre"),
                                } for h in hits
                            ])
                            st.dataframe(res_df)
                            fig = px.bar(res_df, x="id", y="distance", color="distance", title="Distância dos k mais próximos")
                            st.plotly_chart(fig, use_container_width=True, key="milvus_knn_upload")
                        else:
                            st.warning("Nenhum resultado encontrado no Milvus para este embedding.")
                    except Exception as e:
                        st.error(f"Erro ao pesquisar no Milvus: {e}")
    except Exception as e:
        st.error(f"Erro no upload ou pesquisa ad-hoc: {e}")

# 10. Mostrar logs (colapsável)
log_path = os.path.join(benchmark_folder, "benchmark.log")
with st.expander("Logs do Benchmark", expanded=False):
    try:
        if os.path.exists(log_path):
            with open(log_path) as f:
                st.text(f.read())
            st.success("Log carregado.")
        else:
            st.info("Log file não encontrado.")
    except Exception as e:
        st.error(f"Erro ao carregar log: {e}")

# 11. Descrição geral no final da página
st.markdown("""
---
### Descrição Geral

Este dashboard foi desenvolvido para analisar, comparar e explorar o desempenho de diferentes modelos de embeddings de áudio em tarefas de benchmark.  
Permite visualizar estatísticas, gráficos interativos, recursos utilizados e realizar pesquisas vetoriais tanto simuladas como reais na base de dados Milvus.

**O que significa "k mais próximos"?**  
Quando realizas uma pesquisa vetorial (simulada ou real), o sistema procura, para um determinado ficheiro de áudio (query), os `k` ficheiros cujos embeddings são mais próximos do embedding da query, segundo uma métrica de distância (normalmente L2 ou similaridade de cosseno).  
**Na prática, isto significa que os "k mais próximos" correspondem às músicas/áudios mais parecidos com o ficheiro escolhido, de acordo com o modelo de embeddings selecionado.**  
Esta funcionalidade é útil para tarefas como recomendação, deteção de duplicados, agrupamento de músicas semelhantes, entre outras aplicações em processamento de áudio.

Explora os resultados, testa diferentes modelos e compreende como cada abordagem representa e diferencia os ficheiros de áudio!
""")

st.success("Dashboard carregado! Explora todos os gráficos e dados do benchmark.")

