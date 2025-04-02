from pymilvus import MilvusClient

# Tenta conectar
client = MilvusClient(uri="http://localhost:19530")

# Verifica conexão
try:
    # Tenta listar as coleções
    collections = client.list_collections()
    print("Conexão bem sucedida!")
    print("Coleções existentes:", collections)
    
    # Tenta criar uma coleção de teste
    test_name = "test_connection"
    if not test_name in collections:
        client.create_collection(
            collection_name=test_name,
            dimension=128,
            metric_type="L2"
        )
        print(f"Coleção {test_name} criada com sucesso!")
    
    print("Teste completo - tudo OK!")
except Exception as e:
    print(f"Erro na conexão: {e}")
    print("\nVerifique se:")
    print("1. O docker-compose está rodando (docker-compose ps)")
    print("2. A porta 19530 está acessível")
    print("3. Não há firewall bloqueando a conexão")