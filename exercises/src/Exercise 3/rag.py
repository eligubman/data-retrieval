'''import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class RAG:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
        
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question based only on the following context.
            If the answer is not in the context, say "I cannot answer this based on the provided context."
            
            Context:
            {context}
            
            Question: {question}
            """
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def answer(self, question, retrieved_chunks):
        context = "\n\n".join([f"Source: {chunk['filename']}\nContent: {chunk['chunk_content']}" for chunk in retrieved_chunks])
        try:
            response = self.chain.invoke({"context": context, "question": question})
            return response
        except Exception as e:
            return f"Error generating answer: {e}"


'''

import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RAG:
    def __init__(self, model_name="llama3"):
        """
        Initialize a local RAG pipeline using an Ollama model.
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1  # low temperature = factual answers
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question based ONLY on the following context.
                If the answer is not in the context, say exactly:
                "I cannot answer this based on the provided context."

                Context:
                {context}

                Question:
                {question}
            """
            )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def answer(self, question, retrieved_chunks):
        """
        question: string
        retrieved_chunks: list of dicts with keys filename + chunk_content
        """

        context = "\n\n".join(
            [f"Source: {c['filename']}\nContent: {c['chunk_content']}" for c in retrieved_chunks]
        )

        try:
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            return response
        except Exception as e:
            return f"Error generating answer: {e}"
