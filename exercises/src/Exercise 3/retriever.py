import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import pickle

class Retriever:
    def __init__(self, chunks, embedding_model_name='all-MiniLM-L6-v2', cache_dir="cache"):
        self.chunks = chunks
        self.corpus_texts = [chunk['chunk_content'] for chunk in chunks]
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize BM25
        print("Initializing BM25...")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        tokenized_corpus = [self.tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize Dense Retriever
        print(f"Initializing Dense Retriever with {embedding_model_name}...")
        self.encoder = SentenceTransformer(embedding_model_name)
        
        # Check for cached embeddings
        cache_path = os.path.join(cache_dir, f"embeddings_{len(chunks)}_{embedding_model_name}.pkl")
        if os.path.exists(cache_path):
            print("Loading embeddings from cache...")
            with open(cache_path, 'rb') as f:
                self.corpus_embeddings = pickle.load(f)
        else:
            print("Encoding corpus (this may take a while)...")
            self.corpus_embeddings = self.encoder.encode(self.corpus_texts, show_progress_bar=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.corpus_embeddings, f)
        
    def tokenize(self, text):
        return nltk.word_tokenize(text.lower())

    def retrieve_bm25(self, query, k=5):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_n], scores[top_n]

    def retrieve_dense(self, query, k=5):
        query_embedding = self.encoder.encode([query])
        scores = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        top_n = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_n], scores[top_n]

    def retrieve_hybrid(self, query, k=5, alpha=0.5):
        # BM25 scores
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Dense scores
        query_embedding = self.encoder.encode([query])
        dense_scores = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        
        # Normalize scores (Min-Max normalization)
        def normalize(scores):
            if np.max(scores) == np.min(scores):
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)
            
        norm_bm25 = normalize(bm25_scores)
        norm_dense = normalize(dense_scores)
        
        # Combine
        hybrid_scores = alpha * norm_dense + (1 - alpha) * norm_bm25
        
        top_n = np.argsort(hybrid_scores)[::-1][:k]
        return [self.chunks[i] for i in top_n], hybrid_scores[top_n]
