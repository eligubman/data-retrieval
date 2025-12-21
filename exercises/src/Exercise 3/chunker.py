import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    def fixed_size_split(self, documents, max_words=660, overlap_sentences=3):
        chunks = []
        for doc in documents:
            text = doc['content']
            sentences = nltk.sent_tokenize(text)
            
            current_chunk = []
            current_word_count = 0
            
            i = 0
            while i < len(sentences):
                sentence = sentences[i]
                word_count = len(sentence.split())
                
                if current_word_count + word_count > max_words and current_chunk:
                    # Chunk is full, save it
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        **doc,
                        "chunk_content": chunk_text,
                        "chunk_method": "fixed"
                    })
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - overlap_sentences)
                    # We need to find the index in 'sentences' where this overlap starts
                    # This is a bit tricky with the while loop index. 
                    # Let's simplify: just keep the last 'overlap_sentences' from current_chunk list
                    
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                    
                    # Don't increment i here, we just processed the overlap for the *new* chunk
                    # But wait, we haven't added the *current* sentence that caused overflow yet.
                    # Actually, the standard sliding window approach is better.
                
                current_chunk.append(sentence)
                current_word_count += word_count
                i += 1
            
            # Add the last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    **doc,
                    "chunk_content": chunk_text,
                    "chunk_method": "fixed"
                })
                
        print(f"Fixed split created {len(chunks)} chunks.")
        return chunks

    def recursive_split(self, documents, chunk_size=1000, chunk_overlap=200):
        # Note: chunk_size in RecursiveCharacterTextSplitter is characters, not words.
        # 660 words is roughly 3000-4000 characters. Let's approximate or use a reasonable default.
        # The prompt didn't specify size for recursive, just "Recursive". 
        # I'll use a standard size like 2000 chars (~400 words) to be somewhat comparable but distinct.
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = []
        for doc in documents:
            split_texts = text_splitter.split_text(doc['content'])
            for text in split_texts:
                chunks.append({
                    **doc,
                    "chunk_content": text,
                    "chunk_method": "recursive"
                })
                
        print(f"Recursive split created {len(chunks)} chunks.")
        return chunks
