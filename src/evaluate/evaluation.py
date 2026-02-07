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
    2. Average Confidence Score (Relevance)
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
        results = db.similarity_search_with_score(query, k=3)
        
        end_time = time.time()
        latency = end_time - start_time
        total_time += latency

        # Calculate Confidence of the #1 best result
        # Score is distance (lower is better), so we invert it for "Confidence"
        best_doc, best_score = results[0]
        confidence = round((1 - best_score) * 100, 2)
        
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
    if avg_score > 75:
        print("🌟 RATING: EXCELLENT (High relevance)")
    elif avg_score > 50:
        print("👍 RATING: GOOD (Acceptable relevance)")
    else:
        print("⚠️ RATING: POOR (Check chunk size or embeddings)")

if __name__ == "__main__":
    # Test Queries (Mix of specific dates, concepts, and people)
    # These are tailored to the History Chapter you uploaded
    sample_queries = [
        "When was the Gandhi-Irwin Pact signed?",
        "What is the meaning of Satyagraha?",
        "Who was General Dyer?",
        "Why was the Simon Commission boycotted?",
        "Describe the Jallianwala Bagh incident.",
        "What did the Inland Emigration Act of 1859 do?"
    ]
    
    evaluate_retrieval(sample_queries)