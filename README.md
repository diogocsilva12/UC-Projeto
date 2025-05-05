## Como usar o projeto

### 1. Levantar o Milvus e dependências

```bash
docker-compose up -d
```

Acede à interface Attu em: [http://localhost:8000](http://localhost:8000)


### 3. Correr o dashboard interativo

```bash
streamlit run src/dashboard.py
```

### 4. Testar ligação ao Milvus

```bash
python src/test_connection.py
```

---

### 5. Interface e Usabilidade

#### a) Dashboard: Tooltips, Upload e Erros Milvus

Adiciona estas melhorias ao início do teu `dashboard.py`:

````python
# filepath: [dashboard.py](http://_vscodecontentref_/1)
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

# Mostra mensagem de erro se Milvus não estiver disponível
def safe_get_milvus_client():
    try:
        return MilvusClient(uri="http://localhost:19530")
    except Exception as e:
        st.error(f"Não foi possível ligar ao Milvus: {e}")
        return None

@st.cache_resource
def get_milvus_client():
    return safe_get_milvus_client()