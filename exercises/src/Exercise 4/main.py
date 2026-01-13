import os
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader
from chunker import Chunker
from retriever import Retriever
from rag import RAG

def main():
    # --- שלב 2: אינדוקס והכנת נתונים --- [cite: 14]
    print("--- Stage 2: Temporal Indexing ---")
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    loader = DataLoader(base_dir)
    documents = loader.load_all()
    
    if not documents:
        print("No documents found. Exiting.")
        return

    chunker = Chunker()
    # שימוש בשתי שיטות חלוקה כפי שנדרש [cite: 69]
    fixed_chunks = chunker.fixed_size_split(documents)
    recursive_chunks = chunker.recursive_split(documents)

    # יצירת היסטוגרמת התפלגות (תוצר שלב 2) [cite: 21]
    plot_histogram(recursive_chunks)

    # --- אתחול רכיבים ---
    # נשתמש ב-Recursive Chunks כבסיס להשוואה העיקרית 
    retriever = Retriever(recursive_chunks) 
    rag = RAG() # מוגדר לעבוד עם Ollama 3

    # --- שלב 1, 3 ו-4: הגדרת שאלות והרצת הערכה --- [cite: 41]
    questions = [
        # א. נקודה בזמן (Point-in-time) [cite: 43]
        {"type": "Point-in-time", "text": "What was the specific budget allocated to security in 2024?"},
        {"type": "Point-in-time", "text": "What was the primary economic concern mentioned in the debate on July 10, 2023?"},
        
        # ב. סטטוס עדכני (Recency) [cite: 47]
        {"type": "Recency", "text": "What is the current official position regarding the State of Israel?"},
        {"type": "Recency", "text": "What is the current official position regarding Hamas/Gaza?"},
        {"type": "Recency", "text": "Has the official position in the last quarter of 2023 changed relative to the official position in the last quarter of 2025?"},
        {"type": "Recency", "text": "What is the most recent statement regarding climate change policy?"},
        
        # ג. אבולוציה (Evolution) [cite: 59]
        {"type": "Evolution", "text": "How did the Prime Minister/President's rhetoric regarding the war between Israel and Hamas/Gaza develop between his first and last speech?"},
        {"type": "Evolution", "text": "How has the discussion about the cost of living evolved from 2023 to 2025?"},
        
        # ד. קונפליקט (Conflict) [cite: 63]
        {"type": "Conflict", "text": "Who is the Minister of Defense/Secretary of Defense?"},
        {"type": "Conflict", "text": "Who was the Prime Minister in early 2023 compared to late 2024?"}
    ]

    results = []
    k_values = [5] #[3, 5, 10]
    strategies = ["Baseline", "HardFilter", "SoftDecay"] # אסטרטגיות שלב 3 [cite: 25]

    print("\n--- Running Evaluation ---")
    for q in questions:
        print(f"Processing: {q['text']}")
        for strategy in strategies:
            for k in k_values:
                # לוגיקה מיוחדת לשאלות אבולוציה (שלב 4) [cite: 37]
                if q['type'] == "Evolution" and strategy != "Baseline":
                    # אחזור כפול: תקופה מוקדמת ותקופה מאוחרת [cite: 38]
                    early_top_k, late_top_k = retriever.retrieve_evolutionary(q['text'], k=k)
                    answer = rag.evolutionary_answer(q['text'], early_top_k, late_top_k)
                    retrieved_docs = [c['filename'] for c in early_top_k + late_top_k]
                else:
                    # אחזור רגיל או מבוסס זמן (שלב 3) [cite: 26, 29]
                    chunks = retriever.retrieve(q['text'], k=k, strategy=strategy)
                    answer = rag.answer(q['text'], chunks)
                    retrieved_docs = [c['filename'] for c in chunks]

                results.append({
                    "Question Type": q['type'],
                    "Question": q['text'],
                    "Strategy": strategy,
                    "k": k,
                    "Answer": answer,
                    "Sources": retrieved_docs
                })

    # --- שמירת תוצאות (תוצר שלב 3) --- [cite: 34]
    df_results = pd.DataFrame(results)
    df_results.to_csv("temporal_rag_results.csv", index=False, encoding='utf-8-sig')
    print("\nEvaluation complete. Results saved to 'temporal_rag_results.csv'.")

def plot_histogram(chunks):
    """מייצר היסטוגרמה להוכחת אינדוקס המטא-דאטה [cite: 21]"""
    df = pd.DataFrame(chunks)
    df['year'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.year
    df_clean = df.dropna(subset=['year'])
    
    plt.figure(figsize=(10, 5))
    df_clean['year'].value_counts().sort_index().plot(kind='bar', color='teal')
    plt.title("Chunk Distribution by Year (Temporal Indexing)")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.savefig("temporal_distribution_histogram.png")
    plt.close()

if __name__ == "__main__":
    main()