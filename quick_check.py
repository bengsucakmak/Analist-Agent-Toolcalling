# quick_check.py
import os, json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY boş. Terminalde 'export OPENROUTER_API_KEY=...' yap.")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,                                 # <-- anahtarı doğrudan veriyoruz
    model="openrouter/sonoma-sky-alpha",             # tool-calling destekli ve ücretsiz
    temperature=0.0,
    default_headers={                                # (OpenRouter önerisi – opsiyonel)
        "HTTP-Referer": "http://localhost",
        "X-Title": "Analist AI Agent Dev"
    }
)

@tool
def ping_tool(q: str) -> str:
    "Echo a message"
    return f"PONG: {q}"

bound = llm.bind_tools([ping_tool])
resp = bound.invoke([HumanMessage(content="Call ping_tool with q='hello'")])

print("tool_calls:", getattr(resp, "tool_calls", None))
print("content:", resp.content)
