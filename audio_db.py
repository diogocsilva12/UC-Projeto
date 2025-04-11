# Ficheiro: audio_db.py
# Descrição: Funções para manipular vetores de áudio na base de dados

from db_config import get_client

def add_audio_vector(audio_id, vector, metadata=None):
    """
    Adiciona um vetor de áudio à base de dados.
    
    Args:
        audio_id: Identificador único do áudio
        vector: Lista com o vetor de embeddings
        metadata: Dicionário com metadados (nome do ficheiro, duração, etc.)
        
    Returns:
        bool: True se a inserção foi bem-sucedida
    """
    client = get_client()
    
    # Prepara os dados para inserção
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
    """
    Pesquisa áudios similares baseado no vetor de consulta.
    
    Args:
        query_vector: Vetor de embeddings para pesquisar similares
        limit: Número máximo de resultados
    
    Returns:
        Lista de resultados similares com os seus IDs e pontuações
    """
    client = get_client()
    
    # Realiza a pesquisa por similaridade
    results = client.search(
        collection_name="audio_collection",
        data=[query_vector],
        limit=limit,
        output_fields=["*"]  # Retorna todos os campos
    )
    
    return results[0]  # Retorna o primeiro grupo de resultados