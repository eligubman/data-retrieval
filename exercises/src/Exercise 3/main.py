import os
import pandas as pd
from data_loader import DataLoader
from chunker import Chunker
from retriever import Retriever
from rag import RAG

def main():
    # 1. Load Data
    print("--- Step 1: Loading Data ---")
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    loader = DataLoader(base_dir)
    documents = loader.load_all()
    
    if not documents:
        print("No documents loaded. Exiting.")
        return

    # 2. Chunking
    print("\n--- Step 2: Chunking ---")
    chunker = Chunker()
    
    print("Creating Fixed Size Chunks...")
    fixed_chunks = chunker.fixed_size_split(documents)
    
    print("Creating Recursive Chunks...")
    recursive_chunks = chunker.recursive_split(documents)

    # 3. Initialize Retrievers
    print("\n--- Step 3: Initializing Retrievers ---")
    print("Initializing Retriever for Fixed Chunks...")
    retriever_fixed = Retriever(fixed_chunks, cache_dir="cache_fixed")
    
    print("Initializing Retriever for Recursive Chunks...")
    retriever_recursive = Retriever(recursive_chunks, cache_dir="cache_recursive")

    # 4. Initialize RAG
    print("\n--- Step 4: Initializing RAG ---")
    rag = RAG()

    # 5. Define Questions and Parameters
    questions = [
        # Factual Questions
        {"type": "Factual", "text": "What was the main topic of the debate on July 3rd, 2023 in the US Congress?"},
        {"type": "Factual", "text": "Who spoke about the 'District of Columbia National Guard Commanding General Residency Act'?"},
        {"type": "Factual", "text": "What concerns were raised about Thames Water in the UK debates?"},
        {"type": "Factual", "text": "Which UK MP paid tribute to Winnie Ewing?"},
        
        # Conceptual Questions
        {"type": "Conceptual", "text": "Compare the concerns regarding cost of living in the US and UK based on the debates."},
        {"type": "Conceptual", "text": "How do the approaches to environmental issues differ between the discussed US and UK contexts?"},
        {"type": "Conceptual", "text": "What are the recurring themes regarding public service and infrastructure in both countries?"},
        {"type": "Conceptual", "text": "Analyze the sentiment towards government spending in the provided texts."}
    ]
    
    # k values to test
    k_values = [3, 5, 10] # Example k values, prompt says k1, k2, k3
    
    results = []

    # 6. Run Evaluation
    print("\n--- Step 5: Running Evaluation ---")
    
    # We need to run:
    # 1. Fixed + BM25
    # 2. Recursive + BM25
    # 3. Fixed + Hybrid
    # 4. Recursive + Hybrid
    
    configs = [
        ("Fixed", "BM25", retriever_fixed),
        ("Recursive", "BM25", retriever_recursive),
        ("Fixed", "Hybrid", retriever_fixed),
        ("Recursive", "Hybrid", retriever_recursive)
    ]

    for q_idx, question in enumerate(questions):
        print(f"\nProcessing Question {q_idx + 1}: {question['text']}")
        
        for chunk_method, retrieval_method, retriever in configs:
            for k in k_values:
                print(f"  Running: {chunk_method} | {retrieval_method} | k={k}")
                
                # Retrieve
                if retrieval_method == "BM25":
                    retrieved_chunks, scores = retriever.retrieve_bm25(question['text'], k=k)
                else: # Hybrid
                    retrieved_chunks, scores = retriever.retrieve_hybrid(question['text'], k=k)
                
                # Generate Answer
                answer = rag.answer(question['text'], retrieved_chunks)
                
                # Store Result
                results.append({
                    "Question Type": question['type'],
                    "Question": question['text'],
                    "Chunking Method": chunk_method,
                    "Retrieval Method": retrieval_method,
                    "k": k,
                    "Answer": answer,
                    "Retrieved Sources": [c['filename'] for c in retrieved_chunks]
                })

    # 7. Save Results
    print("\n--- Step 6: Saving Results ---")
    df = pd.DataFrame(results)
    output_file = "rag_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
