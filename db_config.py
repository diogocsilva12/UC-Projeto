from pymilvus import MilvusClient

def get_client():
    """Retorna uma conexão com o Milvus"""
    client = MilvusClient(
        uri="http://localhost:19530",
        # Remova o token para esta versão do Milvus v2.5.6
        # token="root:Milvus"  # Remova esta linha
    )
    return client

def init_database():
    """Inicializa a base de dados e coleções necessárias"""
    client = get_client()
    
    # Verifique se a conexão está funcionando
    print("Conectado ao Milvus:", client.has_collection("test_collection"))
    
    # Criar coleção se não existir
    if not client.has_collection("audio_collection"):
        client.create_collection(
            collection_name="audio_collection",
            dimension=1536,
            metric_type="COSINE"
        )
        print("Coleção de áudio criada!")
    
    return client

# Inicializar base de dados e coleções
client = init_database()

# Inserir dados de exemplo
try:
    # Criar vetor de teste (normalmente viria de um modelo de embedding)
    test_vector = [0.1] * 1536  # Vetor de exemplo com 1536 dimensões
    
    # Inserir o vetor na coleção
    client.insert(
        collection_name="audio_collection",
        data=[
            {"id": "1", "vector": test_vector, "filename": "teste1.mp3"},
            {"id": "2", "vector": [0.2] * 1536, "filename": "teste2.mp3"}
        ]
    )
    print("Dados inseridos com sucesso!")
    
    # Contar registros na coleção
    count = client.get_collection_stats("audio_collection")
    print(f"Total de registros: {count['row_count']}")
    
    # Fazer uma busca simples
    results = client.search(
        collection_name="audio_collection",
        data=[test_vector],
        limit=5,
        output_fields=["id", "filename"]
    )
    print("Resultados da busca:", results)
    
except Exception as e:
    print(f"Erro ao manipular dados: {e}")