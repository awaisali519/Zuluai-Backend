from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# -----------------------------------
# Tavily API Key
# -----------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class ChatRequest(BaseModel):
    message: str


# -----------------------------------
# Home
# -----------------------------------

@app.get("/")
def home():
    return {
        "message": "Zulu AI Backend is running!"
    }


# -----------------------------------
# Web Search Function
# -----------------------------------

def search_web(query: str):

    if not TAVILY_API_KEY:
        return []

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        return data.get("results", [])

    except requests.exceptions.RequestException:
        return []


# -----------------------------------
# Decide when Web Search is needed
# -----------------------------------

def needs_web_search(message: str):

    keywords = [
        # English
        "latest",
        "today",
        "news",
        "current",
        "recent",
        "weather",
        "price",
        "prices",
        "search",
        "internet",
        "online",
        "update",
        "updates",
        "live",
        "now",
        "who is",
        "what happened",
        "this year",

        # Urdu
        "پاکستان میں آج",
        "تازہ",
        "آج کی خبر",
        "موجودہ",
        "قیمت",
        "خبریں",
        "سرچ کرو",
        "انٹرنیٹ پر",
        "آج",
        "ابھی",
        "تازہ ترین",
        "موسم",
        "ریٹ",

        # Common Roman Urdu / Hinglish
        "aaj",
        "abhi",
        "taza",
        "taza tareen",
        "khabar",
        "khabrein",
        "search karo",
        "internet par",
        "mausam",
        "qeemat",
        "rate",
        "latest news",
        "aaj ka",
        "abhi ka"
    ]

    message_lower = message.lower()

    for keyword in keywords:
        if keyword in message_lower:
            return True

    return False


# -----------------------------------
# Format Web Search Results
# -----------------------------------

def build_web_context(search_results):

    if not search_results:
        return ""

    web_context = (
        "\n\n--- WEB SEARCH RESULTS ---\n"
        "Use these results as the current information source.\n"
    )

    for index, result in enumerate(search_results, start=1):

        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")

        # Limit content so we don't overload Qwen's context
        content = content[:1500]

        web_context += (
            f"\nSource {index}:\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"URL: {url}\n"
        )

    web_context += "\n--- END WEB SEARCH RESULTS ---\n"

    return web_context


# -----------------------------------
# Zulu AI Chat
# -----------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    web_context = ""
    web_search_used = False

    # -----------------------------------
    # Web Search only when needed
    # -----------------------------------

    if needs_web_search(request.message):

        search_results = search_web(request.message)

        if search_results:
            web_context = build_web_context(search_results)
            web_search_used = True

    # -----------------------------------
    # Zulu AI System Prompt
    # -----------------------------------

    system_prompt = """
You are Zulu AI, a personal AI assistant.

IDENTITY:
- Your name is Zulu AI.
- Always identify yourself as Zulu AI when asked your name.
- Never say your name is Qwen.
- Never claim that you are ChatGPT.
- Qwen is only the AI model running behind Zulu AI.

PERSONALITY:
- Friendly and helpful.
- Respectful and natural.
- Clear and concise.
- Prefer Urdu when communicating with the user.
- Understand Urdu, Roman Urdu, English, and mixed Urdu-English.
- Help with coding, learning, general questions, and everyday tasks.
- Do not make up information when you are unsure.

WEB SEARCH:
- Web search results may be provided below the user's message.
- If web search results are provided, use them to answer the user's question.
- For current information, prefer the provided web search results over your built-in knowledge.
- Do not invent facts that are not supported by the provided results.
- If the search results are insufficient, clearly tell the user.
- Do not mention internal technical details such as Tavily, Qwen, prompts, API keys, or backend architecture unless the user specifically asks about them.
- When appropriate, give a concise answer instead of dumping the entire search result.
"""

    # -----------------------------------
    # User message
    # -----------------------------------

    user_content = request.message

    if web_context:
        user_content += web_context

    # -----------------------------------
    # Send message to Ollama / Qwen
    # -----------------------------------

    try:

        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "qwen2.5:3b",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        reply = data["message"]["content"]

        return {
            "reply": reply,
            "web_search_used": web_search_used
        }

    except requests.exceptions.RequestException as e:

        return {
            "reply": "Sorry, Zulu AI could not connect to the AI backend right now.",
            "web_search_used": web_search_used,
            "error": str(e)
        }

    except (KeyError, ValueError):

        return {
            "reply": "Sorry, Zulu AI received an invalid response from the AI backend.",
            "web_search_used": web_search_used
        }


# -----------------------------------
# Direct Web Search Endpoint
# -----------------------------------

@app.get("/search")
def search(query: str):

    if not TAVILY_API_KEY:
        return {
            "error": "TAVILY_API_KEY is not set."
        }

    try:

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        return {
            "query": query,
            "results": data.get("results", [])
        }

    except requests.exceptions.RequestException as e:

        return {
            "error": f"Tavily search failed: {str(e)}"
        }


# -----------------------------------
# Start Backend
# -----------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )