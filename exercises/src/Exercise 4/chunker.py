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
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,            # שדה ה-$text הנדרש
                        "timestamp": doc.get("timestamp"), # שדה ה-$timestamp הנדרש
                        "filename": doc.get("filename"),
                        "source": doc.get("source"),
                        "chunk_method": "fixed"
                    })
                    
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_word_count += word_count
                i += 1
            
            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "timestamp": doc.get("timestamp"),
                    "filename": doc.get("filename"),
                    "source": doc.get("source"),
                    "chunk_method": "fixed"
                })
                
        print(f"Fixed split created {len(chunks)} chunks.")
        return chunks

    def recursive_split(self, documents, chunk_size=2000, chunk_overlap=200):
        # שימוש בשיטה הנוספת כפי שנדרש בתרגיל [cite: 69]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = []
        for doc in documents:
            split_texts = text_splitter.split_text(doc['content'])
            for text in split_texts:
                chunks.append({
                    "text": text,                      # שדה ה-$text הנדרש
                    "timestamp": doc.get("timestamp"), # שדה ה-$timestamp הנדרש
                    "filename": doc.get("filename"),
                    "source": doc.get("source"),
                    "chunk_method": "recursive"
                })
                
        print(f"Recursive split created {len(chunks)} chunks.")
        return chunks