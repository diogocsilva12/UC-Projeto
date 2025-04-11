# Ficheiro: db_config.py
# Descrição: Configuração da ligação à base de dados vetorial Milvus

from pymilvus import MilvusClient

def get_client():
    """
    Cria e retorna um cliente de ligação ao Milvus.
    
    Returns:
        MilvusClient: Cliente para interação com o Milvus.
    """
    return MilvusClient(uri="http://localhost:19530")

def init_database():
    """
    Inicializa a base de dados e cria as coleções necessárias, se não existirem.
    
    Returns:
        MilvusClient: Cliente conectado ao Milvus.
    """
    client = get_client()
    
    # Verifica se a ligação está a funcionar
    print("Ligado ao Milvus:", client.has_collection("test_collection"))
    
    # Cria a coleção para os vetores de áudio se não existir
    if not client.has_collection("audio_collection"):
        client.create_collection(
            collection_name="audio_collection",
            dimension=1536,  # Dimensão do vetor de embeddings
            metric_type="COSINE"  # Métrica de similaridade (cosseno)
        )
        print("Coleção de áudio criada!")
    
    return client


# Código de teste (deve ser movido para um ficheiro de teste separado)
if __name__ == "__main__":
    client = init_database()
    
    # Inserir dados de exemplo
    try:
        # Criar vetor de teste
        test_vector = [0.1] * 1536
        
        # Inserir o vetor na coleção
        client.insert(
            collection_name="audio_collection",
            data=[
                {"id": "1", "vector": test_vector, "filename": "teste1.mp3"},
                {"id": "2", "vector": [0.2] * 1536, "filename": "teste2.mp3"}
            ]
        )
        print("Dados inseridos com sucesso!")
        
        # Contar registos na coleção
        count = client.get_collection_stats("audio_collection")
        print(f"Total de registos: {count['row_count']}")
        
        # Fazer uma pesquisa simples
        results = client.search(
            collection_name="audio_collection",
            data=[test_vector],
            limit=5,
            output_fields=["id", "filename"]
        )
        print("Resultados da pesquisa:", results)
        
    except Exception as e:
        print(f"Erro ao manipular dados: {e}")