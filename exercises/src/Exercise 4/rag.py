import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

current_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=current_dir / '.env')

class RAG:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(f"API Key not found in {current_dir / '.env'}")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "google/gemini-2.0-flash-exp:free" 

    def answer(self, question, retrieved_chunks):
        # FIX 1: Changed 'chunk_content' to 'text' to match your Chunker
        context = "\n\n".join(
            [f"Source: {c['filename']}\nDate: {c.get('timestamp', 'Unknown')}\nContent: {c['text']}" for c in retrieved_chunks]
        )

        combined_prompt = (
            f"INSTRUCTIONS: Answer the question based ONLY on the provided context. "
            f"If the answer is not in the context, say: 'I cannot answer this based on the provided context.' "
            f"Pay attention to dates in filenames/metadata.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )
        
        return self._send_request(combined_prompt)

    # FIX 2: Added this missing method required by your main.py
    def evolutionary_answer(self, question, early_chunks, late_chunks):
        early_context = "\n".join([f"- {c['text']} ({c.get('timestamp', 'Unknown')})" for c in early_chunks])
        late_context = "\n".join([f"- {c['text']} ({c.get('timestamp', 'Unknown')})" for c in late_chunks])
        
        prompt = (
            f"INSTRUCTIONS: Analyze how the topic described in the question has EVOLVED over time.\n"
            f"Compare the 'Early Period' context with the 'Late Period' context.\n\n"
            f"EARLY PERIOD CONTEXT:\n{early_context}\n\n"
            f"LATE PERIOD CONTEXT:\n{late_context}\n\n"
            f"QUESTION: {question}"
        )
        
        return self._send_request(prompt)

    def _send_request(self, prompt):
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Temporal_RAG_Testing"
                    }
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if "429" in str(e) and attempt < 2:
                    time.sleep(20)
                    continue
                return f"Error: {e}"