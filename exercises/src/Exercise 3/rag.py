import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (exercises/)
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class RAG:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print(f"Warning: OPENROUTER_API_KEY not found in environment variables or {env_path}. Please set it in .env file.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "openai/gpt-oss-20b:free"

    def answer(self, question, retrieved_chunks):
        """
        question: string
        retrieved_chunks: list of dicts with keys filename + chunk_content
        """
        context = "\n\n".join(
            [f"Source: {c['filename']}\nContent: {c['chunk_content']}" for c in retrieved_chunks]
        )

        system_prompt = """You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say exactly:
"I cannot answer this based on the provided context."
"""
        
        user_message = f"""Context:
{context}

Question:
{question}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_message
                    }
                ],
                extra_body={"reasoning": {"enabled": True}}
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"
