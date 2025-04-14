# Ficheiro: db_config.py
# Descrição: Configuração da ligação à base de dados vetorial Milvus

from pymilvus import MilvusClient
import pymilvus

def get_client():
    """
    Cria e retorna um cliente de ligação ao Milvus.
    
    Returns:
        MilvusClient: Cliente para interação com o Milvus.
    """
    return MilvusClient(uri="http://localhost:19530")

def init_database():
    """
    Inicializa a base de dados e cria a coleção de áudio.
    
    Returns:
        MilvusClient: Cliente conectado ao Milvus.
    """
    client = get_client()
    
    # Verifica se a ligação está a funcionar
    connection_ok = client.list_collections() is not None
    print("Ligado ao Milvus:", connection_ok)
    
    # Verifica se a coleção existe
    if client.has_collection("audio_collection"):
        # Se existir, vamos eliminá-la para recriá-la com a dimensão correta
        print("A eliminar coleção anterior para recriação...")
        client.drop_collection("audio_collection")
    
    # Cria a coleção para os vetores de áudio
    print("A criar nova coleção com dimensão 768...")
    client.create_collection(
        collection_name="audio_collection",
        dimension=768,  # Dimensão do vetor de embeddings para Wav2Vec2
        metric_type="COSINE",  # Métrica de similaridade (cosseno)
        # Definir explicitamente que o campo ID é inteiro
        primary_field_name="id",
        primary_field_type=pymilvus.DataType.INT64
    )
    print("Coleção de áudio criada!")
    
    # Criar alguns dados de teste
    test_vectors = [
        {"id": 1, "vector": [0.1] * 768, "filename": "teste1.mp3"},
        {"id": 2, "vector": [0.2] * 768, "filename": "teste2.mp3"},
        {"id": 3, "vector": [0.3] * 768, "filename": "teste3.mp3"}
    ]
    
    # Inserir dados de teste
    client.insert(
        collection_name="audio_collection",
        data=test_vectors
    )
    print("Dados de teste inseridos!")
    
    # Mostrar contagem de registos
    count = client.get_collection_stats("audio_collection")
    print(f"Total de registos: {count['row_count']}")
    
    return client

# Executar se o script for executado diretamente
if __name__ == "__main__":
    init_database()
    print("Base de dados inicializada com sucesso.")