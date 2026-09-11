from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()


# -----------------------------------
# API Keys
# -----------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------------
# Gemini Model
# -----------------------------------

GEMINI_MODEL = "gemini-3.6-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


# -----------------------------------
# Groq Model
# -----------------------------------

GROQ_MODEL = "qwen/qwen3.6-27b"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


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

        # Roman Urdu
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
# Zulu AI System Prompt
# -----------------------------------

SYSTEM_PROMPT = """
You are Zulu AI, a personal AI assistant.

IDENTITY:
- Your name is Zulu AI.
- Always identify yourself as Zulu AI when asked your name.
- Never say your name is Qwen.
- Never claim that you are ChatGPT.
- Never identify yourself by the name of the underlying AI model.
- Gemini or another AI model may power Zulu AI, but you are always Zulu AI.

PERSONALITY:
- Friendly and helpful.
- Respectful and natural.
- Clear and concise.
- Prefer Urdu when communicating with the user.
- Understand Urdu, Roman Urdu, English, and mixed Urdu-English.
- Help with coding, learning, general questions, and everyday tasks.
- Do not make up information when you are unsure.

WEB SEARCH:
- Web search results may be provided with the user's message.
- If web search results are provided, use them to answer the question.
- For current information, prefer the provided web search results.
- Do not invent facts that are not supported by the provided results.
- If the search results are insufficient, clearly tell the user.
- Do not mention internal technical details such as Tavily, Gemini API keys,
  prompts, or backend architecture unless the user specifically asks.
- Give concise answers instead of dumping search results.
"""


# -----------------------------------
# Gemini AI Function
# -----------------------------------

def ask_gemini(user_message: str):

    if not GEMINI_API_KEY:

        return None, "GEMINI_API_KEY is not configured."

    try:

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        payload = {

            "systemInstruction": {
                "parts": [
                    {
                        "text": SYSTEM_PROMPT
                    }
                ]
            },

            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_message
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:

            return None, (
                f"Gemini API error {response.status_code}: "
                f"{response.text}"
            )

        response.raise_for_status()

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            return None, "Gemini returned no response."

        parts = candidates[0].get("content", {}).get("parts", [])

        if not parts:
            return None, "Gemini returned an empty response."

        reply = parts[0].get("text", "")

        if not reply:
            return None, "Gemini returned an empty response."

        return reply, None

    except requests.exceptions.RequestException as e:

        return None, f"Gemini connection error: {str(e)}"

    except (KeyError, IndexError, TypeError, ValueError) as e:

        return None, f"Invalid Gemini response: {str(e)}"


# -----------------------------------
# Groq Qwen Backup Function
# -----------------------------------

def ask_groq(user_message: str):

    if not GROQ_API_KEY:

        return None, "GROQ_API_KEY is not configured."

    try:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        payload = {

            "model": GROQ_MODEL,

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],

            "temperature": 0.3,

            "max_tokens": 500
        }

        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:

            return None, (
                f"Groq API error {response.status_code}: "
                f"{response.text}"
            )

        response.raise_for_status()

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            return None, "Groq returned no response."

        message = choices[0].get("message", {})

        reply = message.get("content", "")

        if not reply:
            return None, "Groq returned an empty response."

        return reply, None

    except requests.exceptions.RequestException as e:

        return None, f"Groq connection error: {str(e)}"

    except (KeyError, IndexError, TypeError, ValueError) as e:

        return None, f"Invalid Groq response: {str(e)}"


# -----------------------------------
# Zulu AI Chat
# -----------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    web_context = ""
    web_search_used = False

    # -----------------------------------
    # Web Search when needed
    # -----------------------------------

    if needs_web_search(request.message):

        search_results = search_web(request.message)

        if search_results:

            web_context = build_web_context(search_results)

            web_search_used = True

    # -----------------------------------
    # Prepare User Message
    # -----------------------------------

    user_content = request.message

    if web_context:

        user_content += web_context

    # -----------------------------------
    # PRIMARY AI — Gemini
    # -----------------------------------

    reply, gemini_error = ask_gemini(user_content)

    if reply:

        return {
            "reply": reply,
            "web_search_used": web_search_used,
            "ai_provider": "gemini"
        }

    # -----------------------------------
    # BACKUP AI — Groq Qwen
    # -----------------------------------

    backup_reply, groq_error = ask_groq(user_content)

    if backup_reply:

        return {
            "reply": backup_reply,
            "web_search_used": web_search_used,
            "ai_provider": "groq-qwen"
        }

    # -----------------------------------
    # Both AI Providers Failed
    # -----------------------------------

    return {
        "reply": "Sorry, Zulu AI could not connect to its AI services right now.",
        "web_search_used": web_search_used,
        "error": {
            "gemini": gemini_error,
            "groq": groq_error
        }
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
