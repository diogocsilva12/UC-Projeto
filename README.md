# UC-Projeto
 
Notas Importantes
1. Inicialização: Para iniciar o projeto, executar:
    ```bash
    docker-compose up -d
    ```

2. Interface Web: Aceder a http://localhost:8000 para ver a interface Attu

3. Testes:

- Teste de ligação: python test_connection.py
- Teste da aplicação principal: python main.py

4. Embeddings de Áudio: Para utilizar, substituir os vetores de exemplo por embeddings reais gerados a partir de modelos como Wav2Vec2, VGGish ou OpenL3. r (não implementada). Devemos escolher o Wav2Vec 2, pois é o mais utilizado para áudio.

5. Dimensões: Temos que certificar que escolhemos a dimensão correta para os embeddings. O Wav2Vec 2 tem 768 dimensões, enquanto o VGGish tem 128. O OpenL3 tem 6144 dimensões. Para o nosso projeto, vamos usar o Wav2Vec 2.

6. Persistência: O Milvus armazena os dados em volumes persistentes, que são criados automaticamente pelo Docker. Não apague o dir "volumes/" após parar os serviços.


# Estrutura do Projeto
```plaintext
UC-Projeto/
├── docker-compose.yml      # Configuração dos serviços Docker
├── db_config.py            # Configuração da ligação à base de dados Milvus
├── audio_db.py             # Funções para manipular vetores de áudio na BD
├── main.py                 # Ponto de entrada principal da aplicação
├── test_connection.py      # Script para testar a ligação ao Milvus
├── app.py                  # (Vazio) Reservado para interface de aplicação
├── audio_data/             # Diretório para guardar ficheiros de áudio
└── volumes/    
´´´            # Dados persistentes do Milvus (criados pelo Docker)