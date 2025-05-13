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

