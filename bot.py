import os
import json
import base64
import asyncio
import mimetypes
import math
import re
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional

import aiohttp
import discord
from discord.ext import commands
from discord import app_commands  

# -------------- Configuration --------------
DEFAULT_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text:v1.5" 
KNOWLEDGE_DIR = "knowledge" 

# Allowed models
ALLOWED_MODELS: Dict[str, str] = {
    "gemma3:12b":     "Wysoka jakość, zrównoważony.",
    "llama3:8b":      "Szybki, dobry do rozmowy.",
    "deepseek-r1:8b": "Silna logika i kodowanie.",
    "ministral-3:8b": "Model widzący (obsługuje obrazy).",
}

# Aliases
MODEL_ALIASES: Dict[str, str] = {
    "gemma":    "gemma3:12b",
    "llama":    "llama3:8b",
    "deepseek": "deepseek-r1:8b",
    "ministral":"ministral-3:8b",
}

VISION_MODELS = {"ministral-3:8b"}
VISION_SUGGESTED_MODEL = "ministral-3:8b"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


SYSTEM_PROMPT = (
    "Jesteś pomocnym asystentem AI. Masz dostęp do fragmentów BAZY WIEDZY (Kontekst), która jest w folderze knowledge. "
    "Twoim zadaniem jest odpowiadanie na pytania zgodnie z poniższym algorytmem priorytetów: "
    "1. PRIORYTET: Sprawdź dostarczony KONTEKST. Jeśli zawiera on informacje na temat pytania, "
    "MUSISZ oprzeć odpowiedź na nim, podając wszystkie szczegóły (składniki, liczby, nazwy) dokładnie tak, jak w tekście. "
    "2. Jeśli KONTEKST nie zawiera odpowiedzi lub jest nie na temat, użyj swojej OGÓLNEJ WIEDZY, aby pomóc użytkownikowi. "
    "3. Odpowiadaj ZAWSZE w języku polskim."
    "I pamiętaj, musisz być miły, nie przeklinać ani wyzywać"

)

MAX_HISTORY = 8
DISCORD_MESSAGE_LIMIT = 1900
GUILD_ID_ENV = os.environ.get("GUILD_ID")
GUILD_OBJECT = discord.Object(id=int(GUILD_ID_ENV)) if GUILD_ID_ENV else None

# -------------- Content Filter --------------
class ContentFilter:
    BAD_STEMS = ["kurw", "chuj", "jeb", "pierdol", "fiut", "cip", "nigg"]

    @staticmethod
    def is_safe(text: str) -> bool:
        text_lower = text.lower()
        for stem in ContentFilter.BAD_STEMS:
            if stem in text_lower:
                return False
        return True

# -------------- Knowledge Base / RAG Engine --------------
class KnowledgeBase:
    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []
        self.loaded = False

    async def load_documents(self, directory: str, session: aiohttp.ClientSession):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created knowledge directory: {directory}")
            return

        print("Indexing knowledge base...")
        self.chunks = []
        self.embeddings = []
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                path = os.path.join(directory, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
                    self.chunks.extend(raw_chunks)
        
        for chunk in self.chunks:
            emb = await self._fetch_embedding(session, chunk)
            if emb:
                self.embeddings.append(emb)
        
        self.loaded = True
        print(f"Knowledge base loaded: {len(self.chunks)} chunks.")

    async def _fetch_embedding(self, session: aiohttp.ClientSession, text: str) -> Optional[List[float]]:
        url = f"{OLLAMA_HOST}/api/embeddings"
        payload = {"model": EMBEDDING_MODEL, "prompt": text}
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200: return None
                data = await resp.json()
                return data.get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def find_relevant(self, query_embedding: List[float], top_k=4) -> str:
        if not self.loaded or not self.embeddings:
            return ""
        
        scores = []
        for i, emb in enumerate(self.embeddings):
            dot_product = sum(a * b for a, b in zip(query_embedding, emb))
            scores.append((dot_product, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        best_indices = [idx for _, idx in scores[:top_k]]
        return "\n---\n".join([self.chunks[i] for i in best_indices])

knowledge_base = KnowledgeBase()

# -------------- State --------------
channel_histories: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
channel_models: Dict[int, str] = defaultdict(lambda: DEFAULT_MODEL)
_model_switch_lock = asyncio.Lock()

# -------------- Ollama HTTP (generate) --------------
async def get_query_embedding(session, text):
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    async with session.post(url, json=payload) as resp:
        if resp.status == 200:
            data = await resp.json()
            return data.get("embedding")
    return None

async def ollama_stream_generate(session, model, prompt, history, context=""):
    system_msg = SYSTEM_PROMPT
    if context:
        system_msg += f"\n\nCONTEXT INFORMATION:\n{context}\n"

    lines = [f"[System]: {system_msg}"]
    for role, content in history:
        lines.append(f"[{role.capitalize()}]: {content}")
    lines.append(f"[User]: {prompt}")
    lines.append("[Assistant]:")
    
    full_prompt = "\n".join(lines)
    payload = {"model": model, "prompt": full_prompt, "stream": True}
    url = f"{OLLAMA_HOST}/api/generate"

    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            if not raw: continue
            try:
                obj = json.loads(raw.decode("utf-8"))
                yield obj.get("response", "")
                if obj.get("done"): break
            except: pass

async def ollama_vision_generate(session, model, prompt, history, images):
    lines = [f"[System]: {SYSTEM_PROMPT}"]
    for role, content in history:
        lines.append(f"[{role}]: {content}")
    lines.append(f"[User]: {prompt}")
    
    payload = {
        "model": model,
        "prompt": "\n".join(lines),
        "images": [base64.b64encode(img).decode('ascii') for img in images],
        "stream": False
    }
    async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload) as resp:
        data = await resp.json()
        return data.get("response", "")

# -------------- Discord bot setup --------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

async def send_chunked(channel, text):
    if not text: return
    for i in range(0, len(text), DISCORD_MESSAGE_LIMIT):
        await channel.send(text[i:i+DISCORD_MESSAGE_LIMIT])

async def run_chat_logic(channel, prompt, attachment=None, use_rag=False):
    channel_id = channel.id
    
    if not ContentFilter.is_safe(prompt):
        await channel.send("Wiadomość zablokowana przez filtr bezpieczeństwa.")
        return

    model = channel_models[channel_id]
    
    async with channel.typing():
        async with aiohttp.ClientSession() as session:
            images = []
            if attachment:
                images.append(await attachment.read())
            
            if images:
                if model not in VISION_MODELS:
                    await channel.send(f"Model `{model}` nie obsługuje obrazów. Użyj `/model` i wybierz Ministral.")
                else:
                    try:
                        resp = await ollama_vision_generate(session, model, prompt, channel_histories[channel_id], images)
                        await send_chunked(channel, resp)
                        channel_histories[channel_id].append(("user", f"{prompt} [image]"))
                        channel_histories[channel_id].append(("assistant", resp))
                        return
                    except Exception as e:
                        await channel.send(f"Błąd wizji: {e}")
                        return

            context_data = ""
            if use_rag and knowledge_base.loaded:
                q_emb = await get_query_embedding(session, prompt)
                if q_emb:
                    context_data = knowledge_base.find_relevant(q_emb)
            
            full_response = ""
            buffer = ""
            try:
                async for chunk in ollama_stream_generate(session, model, prompt, channel_histories[channel_id], context_data):
                    buffer += chunk
                    full_response += chunk
                    if len(buffer) > 400:
                        await send_chunked(channel, buffer)
                        buffer = ""
                if buffer:
                    await send_chunked(channel, buffer)
                
                if not ContentFilter.is_safe(full_response):
                    await channel.send("Odpowiedź modelu została usunięta przez filtr (zawierała niedozwolone treści).")
                else:
                    channel_histories[channel_id].append(("user", prompt))
                    channel_histories[channel_id].append(("assistant", full_response))

            except Exception as e:
                
                if "404" in str(e):
                    await channel.send(f"Błąd: Nie masz pobranego modelu `{model}`. Wpisz w terminalu: `ollama pull {model}`")
                else:
                    await channel.send(f"Błąd generowania: {e}")

# -------------- Slash Commands --------------

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    async with aiohttp.ClientSession() as session:
        try:
            await knowledge_base.load_documents(KNOWLEDGE_DIR, session)
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")

    await bot.tree.sync(guild=GUILD_OBJECT)
    print("Commands synced (Ready).")

@bot.tree.command(name="chat", description="Rozmowa z modelem (bez przeszukiwania bazy wiedzy)")
async def slash_chat(interaction: discord.Interaction, message: str, image: Optional[discord.Attachment] = None):
    await interaction.response.defer(thinking=True)
    await run_chat_logic(interaction.channel, message, image, use_rag=False)
    await interaction.delete_original_response()

@bot.tree.command(name="ask", description="Zadaj pytanie (korzysta z bazy wiedzy RAG)")
async def slash_ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    await run_chat_logic(interaction.channel, question, use_rag=True)
    await interaction.edit_original_response(content="Przeszukano bazę wiedzy.")

@bot.tree.command(name="status", description="Pokazuje aktualnie używany model")
async def slash_status(interaction: discord.Interaction):
    current_model = channel_models[interaction.channel_id]
    await interaction.response.send_message(f"Obecnie używany model to: **`{current_model}`**", ephemeral=True)

@bot.tree.command(name="listmodels", description="Lista dostępnych modeli i ich skróty")
async def slash_listmodels(interaction: discord.Interaction):
    lines = ["**Dostępne modele:**"]
    aliases_reversed = {v: k for k, v in MODEL_ALIASES.items()}
    for full_name, description in ALLOWED_MODELS.items():
        alias = aliases_reversed.get(full_name)
        if alias:
            lines.append(f"• **`{full_name}`** (skrót: **`{alias}`**) – {description}")
        else:
            lines.append(f"• **`{full_name}`** – {description}")
    lines.append("\n*Zmień model używając: `/model`*")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@bot.tree.command(name="model", description="Zmień model AI")
@app_commands.describe(wybor="Wybierz model z listy")
@app_commands.choices(wybor=[
    app_commands.Choice(name="Llama 3 (Szybki, Ogólny)", value="llama"),
    app_commands.Choice(name="Gemma 3 (Wysoka jakość)", value="gemma"),
    app_commands.Choice(name="DeepSeek (Logika/Kod)", value="deepseek"),
    app_commands.Choice(name="Ministral (Wizja/Obrazy)", value="ministral"),
])
async def slash_model(interaction: discord.Interaction, wybor: app_commands.Choice[str]):
    raw = wybor.value
    chosen = MODEL_ALIASES.get(raw, raw)
    
    if chosen not in ALLOWED_MODELS:
         await interaction.response.send_message(f"Błąd: {chosen} nie jest na liście ALLOWED_MODELS.", ephemeral=True)
         return

    async with _model_switch_lock:
        channel_models[interaction.channel_id] = chosen
        await interaction.response.send_message(f"Zmieniono model na: **`{chosen}`**", ephemeral=False)

@bot.tree.command(name="help", description="Lista komend")
async def slash_help(interaction: discord.Interaction):
    msg = """**Dostępne komendy:**
    🔹 `/chat [wiadomość] [obraz]` – Luźna rozmowa (obsługuje zdjęcia przy modelu Ministral).
    🔹 `/ask [pytanie]` – Pytanie do bazy wiedzy (przeszuka Twoje pliki .txt).
    🔹 `/model` – Zmiana modelu (Wybierz z listy: Llama, Gemma, DeepSeek, Ministral).
    🔹 `/listmodels` – Pokaż szczegóły modeli.
    🔹 `/status` – Sprawdź, jaki model jest teraz włączony.
    🔹 `/reset` – Wyczyszczenie pamięci rozmowy.
    """
    await interaction.response.send_message(msg, ephemeral=True)

@bot.tree.command(name="reset", description="Resetuje kontekst rozmowy")
async def slash_reset(interaction: discord.Interaction):
    channel_histories[interaction.channel_id].clear()
    await interaction.response.send_message("Pamięć wyczyszczona.", ephemeral=False)

# -------------- Start --------------
if __name__ == "__main__":

#    bot.run('')
