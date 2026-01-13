import os
import pandas as pd
from data_loader import DataLoader
from chunker import Chunker
from retriever import Retriever
from rag import RAG

def main():
    # 1. טעינת נתונים (שימוש במבנה הקיים)
    print("--- Step 1: Loading Data ---")
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    loader = DataLoader(base_dir)
    documents = loader.load_all()
    
    if not documents:
        print("No documents loaded. Exiting.")
        return

    # 2. פיצול לטקסט (שתי השיטות שנדרשו בתרגיל)
    print("\n--- Step 2: Chunking ---")
    chunker = Chunker()
    fixed_chunks = chunker.fixed_size_split(documents)
    recursive_chunks = chunker.recursive_split(documents)

    # 3. אתחול מאחזרים (BM25 ושיטה נוספת - Hybrid)
    print("\n--- Step 3: Initializing Retrievers ---")
    retriever_fixed = Retriever(fixed_chunks, cache_dir="cache_fixed")
    retriever_recursive = Retriever(recursive_chunks, cache_dir="cache_recursive")

    # 4. אתחול ה-LLM
    print("\n--- Step 4: Initializing RAG ---")
    rag = RAG()

    # 5. הגדרת השאלות הטמפורליות (לפי סעיף 41-67 בהנחיות)
    questions = [
        # א. שאילתת נקודה-בזמן (Point-in-time) [cite: 43-45]
        {"type": "Point-in-time", "text": "What was the specific budget allocated to security in 2024?"},
        
        # ב. שאילתות סטטוס עדכני (Recency) [cite: 47-57]
        {"type": "Recency", "text": "What is the current official position regarding the State of Israel?"},
        {"type": "Recency", "text": "What is the current official position regarding Hamas/Gaza?"},
        {"type": "Recency", "text": "Was the official position in the last quarter of 2023 supportive of the State of Israel?"},
        {"type": "Recency", "text": "Was the official position in the last quarter of 2023 supportive of Hamas/Gaza?"},
        {"type": "Recency", "text": "Has the official position in the last quarter of 2023 changed relative to the official position in the last quarter of 2025?"},
        
        # ג. שאילתת אבולוציה (Evolution) [cite: 59-61]
        {"type": "Evolution", "text": "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza develop/change between his first and last speech?"},
        
        # ד. שאילתת קונפליקט (Ambiguity) [cite: 63-65]
        {"type": "Conflict", "text": "Who is the Minister of Defense/Secretary of Defense?"},
        
        # ה. 4 שאלות נוספות לחיזוק הניתוח 
        {"type": "Conflict", "text": "Who is the current Prime Minister of the United Kingdom?"},
        {"type": "Point-in-time", "text": "What were the primary economic concerns discussed specifically in June 2023?"},
        {"type": "Evolution", "text": "How has the discussion about cost of living evolved from the earliest documents to the most recent ones?"},
        {"type": "Recency", "text": "What is the latest update regarding Thames Water's financial situation mentioned in the debates?"}
    ]
    
    # ערכי K לבדיקה
    k_values = [3, 5, 10]
    
    results = []

    # 6. הרצת הערכה
    print("\n--- Step 5: Running Temporal Evaluation (Blind Baseline) ---")
    
    configs = [
        ("Fixed", "BM25", retriever_fixed),
        ("Recursive", "Hybrid", retriever_recursive) # שימוש בשיטה הנוספת (Hybrid) כנדרש [cite: 70]
    ]

    for q_idx, question in enumerate(questions):
        print(f"\nProcessing Question {q_idx + 1}: {question['text']}")
        
        for chunk_method, retrieval_method, retriever in configs:
            for k in k_values:
                print(f"  Running: {chunk_method} | {retrieval_method} | k={k}")
                
                # אחזור
                if retrieval_method == "BM25":
                    retrieved_chunks, _ = retriever.retrieve_bm25(question['text'], k=k)
                else: # Hybrid
                    retrieved_chunks, _ = retriever.retrieve_hybrid(question['text'], k=k)
                
                # יצירת תשובה (המערכת כרגע "עיוורת" לזמן במטא-דאטה)
                answer = rag.answer(question['text'], retrieved_chunks)
                
                # שמירת תוצאה
                results.append({
                    "Question Type": question['type'],
                    "Question": question['text'],
                    "Chunking Method": chunk_method,
                    "Retrieval Method": retrieval_method,
                    "k": k,
                    "Answer": answer,
                    "Retrieved Sources": [c['filename'] for c in retrieved_chunks]
                })

    # 7. שמירה ל-CSV
    print("\n--- Step 6: Saving Temporal Results ---")
    df = pd.DataFrame(results)
    output_file = "temporal_failure_report.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()