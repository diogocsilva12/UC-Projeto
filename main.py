# Ficheiro: main.py
# Descrição: Ponto de entrada principal da aplicação

from db_config import init_database
from audio_db import add_audio_vector, search_similar_audio

def main():
    """
    Função principal que demonstra o uso da base de dados para 
    armazenar e pesquisar vetores de áudio.
    """
    # Inicializa a base de dados
    init_database()
    
    # Exemplo: Adicionar um vetor de áudio (normalmente seria gerado por um modelo de embedding)
    sample_vector = [0.1, 0.2, 0.3] * 512  # Exemplo simplificado
    add_audio_vector(
        audio_id="audio1",
        vector=sample_vector,
        metadata={"filename": "exemplo.mp3", "duration": 120}
    )
    
    # Exemplo: Pesquisar áudios similares
    results = search_similar_audio(sample_vector)
    print("Resultados similares:")
    for item in results:
        print(f"ID: {item['id']}, Pontuação: {item['score']}")

if __name__ == "__main__":
    main()