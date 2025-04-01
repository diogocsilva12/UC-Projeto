from db_config import get_client

def add_audio_vector(audio_id, vector, metadata=None):
    """Adiciona um vetor de áudio à base de dados
    
    Args:
        audio_id: Identificador único do áudio
        vector: Lista com o vetor de embeddings
        metadata: Dicionário com metadados (nome do arquivo, duração, etc)
    """
    client = get_client()
    client.use_database("audio_vec_database")
    
    data = {
        "id": audio_id,
        "vector": vector
    }
    
    # Adiciona metadados se fornecidos
    if metadata:
        for key, value in metadata.items():
            data[key] = value
    
    # Insere na coleção
    client.insert(
        collection_name="audio_collection", 
        data=[data]
    )
    return True

def search_similar_audio(query_vector, limit=5):
    """Busca áudios similares baseado no vetor de consulta
    
    Args:
        query_vector: Vetor de embeddings para buscar similares
        limit: Número máximo de resultados
    
    Returns:
        Lista de resultados similares com seus IDs e scores
    """
    client = get_client()
    client.use_database("audio_vec_database")
    
    results = client.search(
        collection_name="audio_collection",
        data=[query_vector],
        limit=limit,
        output_fields=["*"]  # Retorna todos os campos
    )
    
    return results[0]  # Retorna o primeiro grupo de resultados