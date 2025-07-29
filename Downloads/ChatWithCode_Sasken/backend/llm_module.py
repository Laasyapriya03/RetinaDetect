# backend/llm_module.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key="AIzaSyDzr3v40GCQ4W-jBXDB4_vCYmiIzQL56Zc")

# Model Parameters
model_name = "gemini-1.5-flash"
DEFAULT_TEMPERATURE = 0.2 # A good starting point for factual responses

def generate_answer(question: str, chunks_content: list[str], temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Generates an answer to the question based on provided code chunks.

    Args:
        question (str): The user's question.
        chunks_content (list[str]): A list of code/comment strings (chunk content) to use as context.
        temperature (float): Controls the randomness of the output. Lower is more deterministic.

    Returns:
        str: The generated answer.
    """
    context = "\n\n".join(chunks_content)
    prompt = f"""
    You are a senior software engineer AI assistant.
    Your goal is to provide accurate, concise, and developer-friendly answers to code questions.

    Instructions:
    1. Explain the function line by line using short, structured, clean sentences.
    2. Avoid repeating phrases like 'This line declares...' or 'This line does...'.
    3. Group related lines into blocks if appropriate, rather than one explanation per line.
    4. Highlight function and variable names clearly (e.g., selectionSort, a, min, temp) without using Markdown or special characters like asterisks or backticks.
    5. Avoid unnecessary formality or fluff — be concise and helpful.
    6. Do NOT use Markdown headers or list formatting.
    7. If the function is clear, explain only the important parts that help understanding.

    Code Context:
    {context}

    Question: {question}

    Answer (clean and developer-friendly, no Markdown formatting):
    """
    # Configure generation with the temperature parameter
    generation_config = {
        "temperature": temperature
    }

    model = genai.GenerativeModel(model_name)
    res = model.generate_content(prompt, stream=False, generation_config=generation_config)

    if res.text:
        return res.text
    else:
        return "No answer could be generated."
