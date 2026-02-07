import os
import sys
import time

# 1. Add the project root to the system path so we can import 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from src.database.vector_store import load_vector_db

def evaluate_retrieval(test_queries):
    """
    Runs a set of queries against the Vector Database and measures:
    1. Retrieval Speed (Latency)
    2. Average Confidence Score (Normalized for L2 Distance)
    """
    print("🧪 Starting Retrieval Evaluation System...\n")
    
    try:
        db = load_vector_db()
    except Exception as e:
        print(f"❌ Error: Could not load database. Run main.py first. Details: {e}")
        return

    total_score = 0
    total_time = 0
    
    print(f"{'QUERY':<50} | {'CONFIDENCE':<12} | {'SOURCE (Page)':<20}")
    print("-" * 90)

    for query in test_queries:
        start_time = time.time()
        
        # Get top 3 results with scores
        # Note: ChromaDB returns L2 DISTANCE by default (Lower is Better)
        results = db.similarity_search_with_score(query, k=3)
        
        end_time = time.time()
        latency = end_time - start_time
        total_time += latency

        if not results:
            print(f"{query[:47]:<50} ... | 0.0%         | NO RESULTS")
            continue

        best_doc, best_score = results[0]
        
        # --- 🔧 THE FIX: L2 Normalization Logic ---
        # Formula: Score = 1 / (1 + distance)
        # Distance 0.0 -> 100%
        # Distance 1.0 -> 50%
        # Distance 1.5 -> 40%
        # This prevents negative numbers.
        confidence = round((1 / (1 + best_score)) * 100, 2)
        
        # Get metadata
        source = best_doc.metadata.get('source', 'Unknown').split('/')[-1]
        page = best_doc.metadata.get('page', '??')

        print(f"{query[:47]:<50} ... | {confidence}%      | {source} (p.{page})")
        
        total_score += confidence

    # --- Summary Metrics ---
    avg_score = round(total_score / len(test_queries), 2)
    avg_latency = round(total_time / len(test_queries), 4)

    print("-" * 90)
    print("\n📊 FINAL PERFORMANCE METRICS")
    print("=" * 30)
    print(f"✅ Average Confidence Score: {avg_score}/100")
    print(f"⚡ Average Retrieval Time:   {avg_latency} seconds")
    print("=" * 30)

    # Interpretation
    if avg_score > 50:
        print("🌟 RATING: EXCELLENT (High relevance)")
    elif avg_score > 30:
        print("👍 RATING: GOOD (Acceptable relevance)")
    else:
        print("⚠️ RATING: POOR (Check chunk size or embeddings)")

if __name__ == "__main__":
    # Test Queries tailored to your specific content
    sample_queries = [
        "When was the Gandhi-Irwin Pact signed?",
        "What is the meaning of Satyagraha?",
        "Who was General Dyer?",
        "Why was the Simon Commission boycotted?",
        "Describe the Jallianwala Bagh incident.",
        "What did the Inland Emigration Act of 1859 do?"
    ]
    
    evaluate_retrieval(sample_queries)