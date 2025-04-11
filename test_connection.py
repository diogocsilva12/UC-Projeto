# Ficheiro: test_connection.py
# Descrição: Script para testar a ligação ao Milvus

from pymilvus import MilvusClient

def test_milvus_connection():
    """
    Testa a ligação ao servidor Milvus e cria uma coleção de teste.
    """
    # Tenta ligar ao Milvus
    client = MilvusClient(uri="http://localhost:19530")
    
    try:
        # Lista as coleções existentes
        collections = client.list_collections()
        print("Ligação bem-sucedida!")
        print("Coleções existentes:", collections)
        
        # Cria uma coleção de teste
        test_name = "test_connection"
        if test_name not in collections:
            client.create_collection(
                collection_name=test_name,
                dimension=128,
                metric_type="L2"
            )
            print(f"Coleção {test_name} criada com sucesso!")
        
        print("Teste completo - tudo OK!")
        return True
        
    except Exception as e:
        print(f"Erro na ligação: {e}")
        print("\nVerifique se:")
        print("1. O docker-compose está a correr (docker-compose ps)")
        print("2. A porta 19530 está acessível")
        print("3. Não há firewall a bloquear a ligação")
        return False

if __name__ == "__main__":
    test_milvus_connection()