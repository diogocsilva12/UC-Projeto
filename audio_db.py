# Ficheiro: audio_db.py
# Descrição: Funções básicas para manipular a base de dados de áudio

from db_config import get_client

def add_audio_vector(audio_id, vector, metadata=None):
    """
    Adiciona um vetor de áudio à base de dados.
    
    Args:
        audio_id: Identificador único do áudio (deve ser um número inteiro)
        vector: Lista com o vetor de embeddings
        metadata: Dicionário com metadados (nome do ficheiro, duração, etc.)
        
    Returns:
        bool: True se a inserção foi bem-sucedida
    """
    client = get_client()
    
    # Verificar se o ID é um número inteiro
    if not isinstance(audio_id, int):
        audio_id = int(audio_id)  # Tenta converter para inteiro
    
    # Verificar a dimensão do vetor
    collection_info = client.describe_collection("audio_collection")
    expected_dim = collection_info.get("dimension", 768)
    
    # Ajustar dimensão se necessário
    if len(vector) != expected_dim:
        print(f"AVISO: A dimensão do vetor ({len(vector)}) não corresponde à dimensão esperada ({expected_dim})")
        if len(vector) < expected_dim:
            # Preencher com zeros se for menor
            vector = vector + [0] * (expected_dim - len(vector))
        else:
            # Truncar se for maior
            vector = vector[:expected_dim]
        print(f"Vetor ajustado para dimensão {len(vector)}")
    
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
    print(f"Vetor com ID {audio_id} inserido com sucesso!")
    return True