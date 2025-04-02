import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymilvus import MilvusClient
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

# ===== EMBEDDING MODELS =====

def extract_wav2vec_embeddings(audio_file, sample_rate=16000):
    """Extract embeddings using Facebook's Wav2Vec2"""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Process audio
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling for a fixed-size vector
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    return embedding  # 768-dimensional vector

def extract_vggish_embeddings(audio_file, sample_rate=16000):
    """Extract embeddings using Google's VGGish"""
    import tensorflow as tf
    import tensorflow_hub as hub
    
    # Load VGGish model
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Convert to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Get embeddings
    embedding = vggish_model(audio).numpy()
    
    # If we get multiple segments, average them
    if embedding.ndim > 1:
        embedding = np.mean(embedding, axis=0)
    
    return embedding  # 128-dimensional vector

def extract_openl3_embeddings(audio_file, sample_rate=48000):
    """Extract embeddings using OpenL3"""
    import openl3
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Convert to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Extract embedding
    emb, _ = openl3.get_audio_embedding(audio, sr, 
                                       content_type="music",
                                       embedding_size=512)
    
    # Average across time segments if needed
    if emb.shape[0] > 1:
        emb = np.mean(emb, axis=0)
    else:
        emb = emb.squeeze()
        
    return emb  # 512-dimensional vector

# ===== BENCHMARKING CLASS =====

class AudioEmbeddingBenchmark:
    def __init__(self, uri="http://localhost:19530"):
        self.client = MilvusClient(uri=uri)
        self.results = []
        self.models = {
            "wav2vec2": {
                "extract_fn": extract_wav2vec_embeddings,
                "dim": 768,
                "metric": "COSINE"
            },
            "vggish": {
                "extract_fn": extract_vggish_embeddings,
                "dim": 128,
                "metric": "COSINE"
            },
            "openl3": {
                "extract_fn": extract_openl3_embeddings,
                "dim": 512,
                "metric": "COSINE"
            }
        }
    
    def prepare_test_data(self, audio_dir, limit=None):
        """Process audio files in directory to generate test data"""
        audio_files = []
        
        # Better way to search for files recursively
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(root, file))
        
        if limit and len(audio_files) > limit:
            audio_files = audio_files[:limit]
        
        print(f"Found {len(audio_files)} audio files for testing")
        return audio_files
    
    def extract_embeddings(self, model_name, audio_files):
        """Extract embeddings using the specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")
        
        model = self.models[model_name]
        extract_fn = model["extract_fn"]
        
        embeddings = []
        for audio_file in tqdm(audio_files, desc=f"Extracting {model_name} embeddings"):
            try:
                embedding = extract_fn(audio_file)
                embeddings.append({
                    "filepath": audio_file,
                    "filename": os.path.basename(audio_file),
                    "vector": embedding
                })
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        
        return embeddings
    
    def setup_collection(self, model_name):
        """Create collection for a model if it doesn't exist"""
        collection_name = f"benchmark_{model_name}"
        model = self.models[model_name]
        
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            
        self.client.create_collection(
            collection_name=collection_name,
            dimension=model["dim"],
            metric_type=model["metric"]
        )
        
        return collection_name
    
    def insert_embeddings(self, collection_name, embeddings):
        """Insert embeddings into Milvus collection"""
        data = []
        for i, emb in enumerate(embeddings):
            data.append({
                "id": str(i),
                "vector": emb["vector"].tolist(),
                "filename": emb["filename"]
            })
        
        start_time = time.time()
        self.client.insert(collection_name, data)
        insert_time = time.time() - start_time
        
        return insert_time
    
    def run_search_test(self, model_name, collection_name, query_embeddings, k=10):
        """Benchmark search performance"""
        search_times = []
        
        for emb in tqdm(query_embeddings[:10], desc=f"Testing {model_name} search"):
            start_time = time.time()
            
            self.client.search(
                collection_name=collection_name,
                data=[emb["vector"].tolist()],
                limit=k,
                output_fields=["filename"]
            )
            
            search_time = time.time() - start_time
            search_times.append(search_time)
            
        avg_search_time = sum(search_times) / len(search_times) if search_times else 0
        return avg_search_time
    
    def benchmark_model(self, model_name, train_files, query_files):
        """Run full benchmark for a model"""
        print(f"\n===== Benchmarking {model_name} =====")
        
        # Extract embeddings
        print("Extracting training embeddings...")
        train_embeddings = self.extract_embeddings(model_name, train_files)
        
        print("Extracting query embeddings...")
        query_embeddings = self.extract_embeddings(model_name, query_files)
        
        # Setup collection
        collection_name = self.setup_collection(model_name)
        
        # Insert embeddings
        print("Inserting embeddings...")
        insert_time = self.insert_embeddings(collection_name, train_embeddings)
        print(f"Insertion time: {insert_time:.2f}s for {len(train_embeddings)} vectors")
        
        # Test memory usage
        mem_per_vector = self.models[model_name]["dim"] * 4 / 1024  # KB per vector (32-bit floats)
        total_mem = mem_per_vector * len(train_embeddings)
        print(f"Approximate memory usage: {total_mem:.2f} KB ({mem_per_vector:.2f} KB per vector)")
        
        # Run search test
        print("Testing search performance...")
        search_time = self.run_search_test(model_name, collection_name, query_embeddings)
        print(f"Average search time: {search_time*1000:.2f}ms per query")
        
        # Store results
        self.results.append({
            "model": model_name,
            "dimension": self.models[model_name]["dim"],
            "insert_time": insert_time,
            "search_time_ms": search_time * 1000,
            "memory_kb_per_vector": mem_per_vector
        })
        
        return insert_time, search_time
    
    def plot_results(self):
        """Plot benchmark results"""
        if not self.results:
            print("No benchmark results to plot")
            return
            
        df = pd.DataFrame(self.results)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot dimensions
        axs[0, 0].bar(df['model'], df['dimension'], color='green')
        axs[0, 0].set_title('Vector Dimensions')
        axs[0, 0].set_ylabel('Dimensions')
        
        # Plot insertion time
        axs[0, 1].bar(df['model'], df['insert_time'], color='blue')
        axs[0, 1].set_title('Insert Time')
        axs[0, 1].set_ylabel('Time (seconds)')
        
        # Plot search time
        axs[1, 0].bar(df['model'], df['search_time_ms'], color='red')
        axs[1, 0].set_title('Search Time')
        axs[1, 0].set_ylabel('Time (milliseconds)')
        
        # Plot memory usage
        axs[1, 1].bar(df['model'], df['memory_kb_per_vector'], color='purple')
        axs[1, 1].set_title('Memory Usage per Vector')
        axs[1, 1].set_ylabel('KB per vector')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        
        # Save results to CSV
        df.to_csv('benchmark_results.csv', index=False)
        print("Results saved to benchmark_results.csv and benchmark_results.png")
        
        # Print summary
        print("\n===== BENCHMARK SUMMARY =====")
        print(df.to_string(index=False))
        
        # Determine best model based on combined metrics
        # Normalize scores (lower is better for all but dimension which we ignore)
        df['norm_insert'] = 1 - (df['insert_time'] / df['insert_time'].max())
        df['norm_search'] = 1 - (df['search_time_ms'] / df['search_time_ms'].max())
        df['norm_memory'] = 1 - (df['memory_kb_per_vector'] / df['memory_kb_per_vector'].max())
        
        # Combined score (weighted)
        df['combined_score'] = (
            df['norm_search'] * 0.5 +   # Search performance is most important
            df['norm_memory'] * 0.3 +   # Memory efficiency is next
            df['norm_insert'] * 0.2     # Insert time is least important
        )
        
        best_model = df.loc[df['combined_score'].idxmax(), 'model']
        print(f"\nRECOMMENDED MODEL: {best_model}")
        
        # Compare models side by side
        print("\nModel Comparison:")
        fastest_search = df.loc[df['search_time_ms'].idxmin(), 'model']
        smallest_vectors = df.loc[df['memory_kb_per_vector'].idxmin(), 'model']
        
        print(f"- Fastest search: {fastest_search}")
        print(f"- Most memory efficient: {smallest_vectors}")
        print(f"- Best overall: {best_model}")


# Example usage
if __name__ == "__main__":
    # Get paths from environment variables
    AUDIO_DIR = os.environ.get("AUDIO_DIR", "/app/audio_data")
    MILVUS_URI = os.environ.get("MILVUS_URI", "http://standalone:19530")
    
    # Initialize benchmark with URI from environment
    benchmark = AudioEmbeddingBenchmark(uri=MILVUS_URI)
    
    # Prepare data - use a small subset for testing to save time
    all_files = benchmark.prepare_test_data(AUDIO_DIR, limit=30)
    
    # Split into train and test sets
    train_size = int(0.8 * len(all_files))
    train_files = all_files[:train_size]
    query_files = all_files[train_size:]
    
    # Run benchmarks for all models
    for model_name in ["wav2vec2", "vggish", "openl3"]:
        try:
            benchmark.benchmark_model(model_name, train_files, query_files)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
    
    # Plot and analyze results
    benchmark.plot_results()

