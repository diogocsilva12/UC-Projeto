# Ficheiro: main.py
# Descrição: Ponto de entrada principal da aplicação

from db_config import init_database
from audio_db import add_audio_vector

def main():
    """
    Função principal que demonstra o uso básico da base de dados.
    """
    # Inicializa a base de dados (cria a coleção e insere dados de teste)
    client = init_database()
    
    # Listar todas as coleções
    collections = client.list_collections()
    print(f"Coleções disponíveis: {collections}")
    
    # Mostrar estatísticas da coleção de áudio
    if "audio_collection" in collections:
        stats = client.get_collection_stats("audio_collection")
        print(f"Estatísticas da coleção de áudio: {stats}")
    
    # Adicionar mais um vetor de teste
    print("\nAdicionar novo vetor de teste:")
    add_audio_vector(
        audio_id=1001,
        vector=[0.5] * 768,
        metadata={"filename": "exemplo_novo.mp3", "duration": 120}
    )
    
    print("\nTeste concluído com sucesso!")

if __name__ == "__main__":
    main()