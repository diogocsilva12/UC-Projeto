from db_config import init_database
from audio_db import add_audio_vector, search_similar_audio

def main():
    # Inicializa a base de dados
    client = init_database()
    
    # Exemplo: Adicionar um vetor de áudio (você precisará gerar embeddings reais)
    sample_vector = [0.1, 0.2, 0.3] * 512  # Exemplo simplificado
    add_audio_vector(
        audio_id="audio1",
        vector=sample_vector,
        metadata={"filename": "exemplo.mp3", "duration": 120}
    )
    
    # Exemplo: Buscar áudios similares
    results = search_similar_audio(sample_vector)
    print("Resultados similares:")
    for item in results:
        print(f"ID: {item['id']}, Score: {item['score']}")

if __name__ == "__main__":
    main()