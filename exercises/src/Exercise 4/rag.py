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
        # שימוש במודל פחות עמוס למניעת שגיאות 429
        self.model = "google/gemini-2.0-flash-exp:free" 

    def answer(self, question, retrieved_chunks):
        context = "\n\n".join(
            [f"Source: {c['filename']}\nContent: {c['chunk_content']}" for c in retrieved_chunks]
        )

        # הנחיה מאוחדת לתוך הודעת המשתמש כדי לעקוף את שגיאת ה-system prompt
        combined_prompt = (
            f"INSTRUCTIONS: Answer the question based ONLY on the provided context. "
            f"If the answer is not in the context, say: 'I cannot answer this based on the provided context.' "
            f"Pay attention to dates in filenames (YYYY-MM-DD).\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        # כאן השינוי: שמים הכל ב-user
                        {"role": "user", "content": combined_prompt}
                    ],
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Temporal_RAG_Testing"
                    }
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(20)
                    continue
                return f"Error: {e}"
