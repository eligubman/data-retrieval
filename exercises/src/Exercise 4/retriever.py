import re
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi
import nltk
# Add these imports for the second method (Dense Retrieval)
from sentence_transformers import SentenceTransformer, util
import torch

class Retriever:
    def __init__(self, chunks, alpha=0.3, lambd=0.5):
        self.chunks = chunks
        self.alpha = alpha
        self.lambd = lambd
        
        print("--- Initializing Retrievers ---")
        
        # Method 1: BM25 (Sparse)
        print("1. Indexing BM25...")
        self.tokenized_corpus = [nltk.word_tokenize(c['text'].lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Method 2: Dense Retrieval (Semantic) - This is the "Second Method" required
        print("2. Indexing Dense Vectors (SentenceTransformer)...")
        # We use a lightweight model to keep it fast on CPU
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract texts for embedding
        texts = [c['text'] for c in self.chunks]
        
        # Compute embeddings (This might take a moment)
        self.corpus_embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        print("Indexing Complete.")

    def retrieve(self, query, k=5, strategy="Baseline", retrieval_method="hybrid"):
        """
        retrieval_method can be: 'bm25', 'dense', or 'hybrid'
        """
        results = []
        target_year = int(self._extract_year(query) or 2026)

        # --- Step 1: Calculate Semantic Scores ---
        
        # Score A: BM25
        query_tokens = nltk.word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(query_tokens)
        # Normalize BM25 scores (0 to 1)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        norm_bm25 = bm25_scores / max_bm25

        # Score B: Dense (Vector)
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        # Calculate Cosine Similarity
        dense_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0].cpu().numpy()

        # Combine Scores (Hybrid is usually best, but you can stick to BM25 for speed if needed)
        # We will use 50% BM25 + 50% Dense for the "Semantic Score"
        
        for i, chunk in enumerate(self.chunks):
            # Calculate base semantic similarity
            if retrieval_method == 'bm25':
                sim = norm_bm25[i]
            elif retrieval_method == 'dense':
                sim = dense_scores[i]
            else: # Hybrid
                sim = (0.5 * norm_bm25[i]) + (0.5 * dense_scores[i])

            ts = chunk.get('timestamp', 'Unknown')
            
            # --- Step 2: Apply Temporal Strategy ---
            
            if strategy == "HardFilter":
                year = self._extract_year(query)
                if year and (ts == "Unknown" or not str(ts).startswith(year)):
                    continue
                score = sim
            
            elif strategy == "SoftDecay":
                if ts == "Unknown" or not ts:
                    time_score = 0.1 
                else:
                    try:
                        doc_year = int(str(ts).split('-')[0])
                        delta_t = abs(target_year - doc_year)
                        time_score = 1 / (1 + delta_t * self.lambd)
                    except (ValueError, IndexError):
                        time_score = 0.1
                
                # Formula: Score = (1-alpha)*Sim + alpha*TimeScore
                score = (1 - self.alpha) * sim + self.alpha * time_score
            
            else: # Baseline (No Time)
                score = sim

            res = chunk.copy()
            res['score'] = score
            results.append(res)

        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]

    def retrieve_evolutionary(self, query, k=5):
        # Sort by time
        sorted_chunks = sorted(self.chunks, key=lambda x: str(x.get('timestamp', '')))
        
        if not sorted_chunks:
            return [], []

        mid = len(sorted_chunks) // 2
        
        # Simple approach: Take the chronologically first K and last K chunks
        # This ensures we get the earliest and latest data available in the corpus
        early_top_k = sorted_chunks[:k]
        late_top_k = sorted_chunks[-k:]

        return early_top_k, late_top_k

    def _extract_year(self, text):
        match = re.search(r'\b(20\d{2}|19\d{2})\b', text)
        return match.group(1) if match else None