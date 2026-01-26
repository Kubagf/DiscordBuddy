import os
import json
import base64
import asyncio
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
    "gemma":     "gemma3:12b",
    "llama":     "llama3:8b",
    "deepseek":  "deepseek-r1:8b",
    "ministral": "ministral-3:8b",
}

VISION_MODELS = {"ministral-3:8b"}
VISION_SUGGESTED_MODEL = "ministral-3:8b"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

SYSTEM_PROMPT = (
    "Jesteś inteligentnym i naturalnym rozmówcą na serwerze Discord. "
    "Twoim zadaniem jest udzielanie konkretnych i pomocnych odpowiedzi w języku polskim. "
    
    "ZASADY KORZYSTANIA Z WIEDZY (ŚCIŚLE PRZESTRZEGAJ): "
    "1. Otrzymasz fragmenty tekstu (KONTEKST). Masz je traktować jak swoją własną, wrodzoną wiedzę. "
    "2. ABSOLUTNY ZAKAZ pisania zdań typu: 'na podstawie kontekstu', 'przeszukałem bazę', 'znalazłem w pliku', 'algorytm wskazuje'. "
    "3. Użytkownik nie może wiedzieć, że czytasz jakieś pliki. Po prostu odpowiadaj tak, jakbyś to wiedział od zawsze. "
    "4. Jeśli w KONTEKŚCIE jest odpowiedź -> podaj ją precyzyjnie. "
    "5. Jeśli w KONTEKŚCIE nie ma odpowiedzi -> użyj swojej wiedzy ogólnej lub dopytaj użytkownika, ale nie tłumacz się brakiem danych w plikach. "
    
    "TON WYPOWIEDZI: "
    "Bądź uprzejmy, ale bezpośredni. Unikaj zbędnego lania wody i dziwnych powitań. Pisz jak człowiek, a nie jak robot."
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
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def find_relevant(self, query_embedding: List[float], top_k: int = 4) -> str:
        if (not self.loaded) or (not self.embeddings):
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

# -------------- Helpers (Discord chunking) --------------
def split_for_discord(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> List[str]:
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    parts: List[str] = []
    i = 0

    while i < len(text):
        end = min(i + limit, len(text))

        if end == len(text):
            parts.append(text[i:end])
            break

        cut = text.rfind("\n", i, end)
        if cut == -1:
            cut = text.rfind(" ", i, end)

        if (cut == -1) or (cut <= i + 20):
            cut = end

        parts.append(text[i:cut].rstrip())
        i = cut

        # pomiń białe znaki po cięciu
        while (i < len(text)) and (text[i] == "\n" or text[i] == " "):
            i += 1

    return parts

async def send_chunked_channel(channel: discord.abc.Messageable, text: str) -> None:
    parts = split_for_discord(text)
    for part in parts:
        await channel.send(part)

async def send_chunked_interaction(interaction: discord.Interaction, text: str, ephemeral: bool = False) -> None:
    parts = split_for_discord(text)
    if not parts:
        await interaction.edit_original_response(content="(brak odpowiedzi)")
        return

    await interaction.edit_original_response(content=parts[0])
    for part in parts[1:]:
        await interaction.followup.send(part, ephemeral=ephemeral)


def get_context_key(interaction: discord.Interaction) -> int:
    if interaction.channel_id is not None:
        return int(interaction.channel_id)
    if interaction.channel is not None and hasattr(interaction.channel, "id"):
        return int(interaction.channel.id)
    if interaction.user is not None:
        return int(interaction.user.id)
    return 0

async def get_query_embedding(session: aiohttp.ClientSession, text: str) -> Optional[List[float]]:
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    async with session.post(url, json=payload) as resp:
        if resp.status == 200:
            data = await resp.json()
            return data.get("embedding")
    return None

async def ollama_stream_generate(
    session: aiohttp.ClientSession,
    model: str,
    prompt: str,
    history: Deque[Tuple[str, str]],
    context: str = ""
):
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
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
                yield obj.get("response", "")
                if obj.get("done"):
                    break
            except Exception:
                pass

async def ollama_vision_generate(
    session: aiohttp.ClientSession,
    model: str,
    prompt: str,
    history: Deque[Tuple[str, str]],
    images: List[bytes]
) -> str:
    lines = [f"[System]: {SYSTEM_PROMPT}"]
    for role, content in history:
        lines.append(f"[{role}]: {content}")
    lines.append(f"[User]: {prompt}")

    payload = {
        "model": model,
        "prompt": "\n".join(lines),
        "images": [base64.b64encode(img).decode("ascii") for img in images],
        "stream": False
    }
    async with session.post(f"{OLLAMA_HOST}/api/generate", json=payload) as resp:
        data = await resp.json()
        return data.get("response", "")

# -------------- Core chat logic --------------
async def generate_response(
    channel_id: int,
    prompt: str,
    attachment: Optional[discord.Attachment] = None,
    use_rag: bool = False
) -> str:

    if not ContentFilter.is_safe(prompt):
        return "Wiadomość zablokowana przez filtr bezpieczeństwa."

    model = channel_models[channel_id]

    async with aiohttp.ClientSession() as session:
        images: List[bytes] = []
        if attachment is not None:
            try:
                images.append(await attachment.read())
            except Exception as e:
                return f"Nie udało się odczytać załącznika: {e}"

        if images:
            if model not in VISION_MODELS:
                return f"Model `{model}` nie obsługuje obrazów. Użyj `/model` i wybierz `{VISION_SUGGESTED_MODEL}`."
            try:
                resp = await ollama_vision_generate(session, model, prompt, channel_histories[channel_id], images)
                if not ContentFilter.is_safe(resp):
                    return "Odpowiedź modelu została usunięta przez filtr (zawierała niedozwolone treści)."

                channel_histories[channel_id].append(("user", f"{prompt} [image]"))
                channel_histories[channel_id].append(("assistant", resp))
                return resp
            except Exception as e:
                return f"Błąd wizji: {e}"

        context_data = ""
        if use_rag and knowledge_base.loaded:
            try:
                q_emb = await get_query_embedding(session, prompt)
                if q_emb:
                    context_data = knowledge_base.find_relevant(q_emb)
            except Exception as e:
                context_data = ""
                print(f"RAG error: {e}")

        full_response = ""
        try:
            async for chunk in ollama_stream_generate(session, model, prompt, channel_histories[channel_id], context_data):
                full_response += chunk
        except Exception as e:
            if "404" in str(e):
                return f"Błąd: Nie masz pobranego modelu `{model}`. Wpisz w terminalu: `ollama pull {model}`"
            return f"Błąd generowania: {e}"

        if not full_response:
            return "(brak odpowiedzi)"

        if not ContentFilter.is_safe(full_response):
            return "Odpowiedź modelu została usunięta przez filtr (zawierała niedozwolone treści)."

        channel_histories[channel_id].append(("user", prompt))
        channel_histories[channel_id].append(("assistant", full_response))
        return full_response

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    async with aiohttp.ClientSession() as session:
        try:
            await knowledge_base.load_documents(KNOWLEDGE_DIR, session)
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")

    if GUILD_OBJECT is not None:
        await bot.tree.sync(guild=GUILD_OBJECT)
        print("Commands synced (guild).")
    else:
        await bot.tree.sync()
        print("Commands synced (global).")



@bot.tree.command(name="chat", description="Luźna rozmowa (bez RAG). Opcjonalnie obraz (tylko Ministral).")
@app_commands.describe(prompt="Twoja wiadomość", obraz="Opcjonalny obraz (tylko przy modelu Ministral)")
async def slash_chat(interaction: discord.Interaction, prompt: str, obraz: Optional[discord.Attachment] = None):
    await interaction.response.defer(thinking=True)   
    channel_id = get_context_key(interaction)
    resp = await generate_response(channel_id, prompt, attachment=obraz, use_rag=False)
    
    final_msg = f"{interaction.user.mention}: {prompt}\n\n{resp}"
    
    await send_chunked_interaction(interaction, final_msg, ephemeral=False)


@bot.tree.command(name="ask", description="Zadaj pytanie (korzysta z bazy wiedzy RAG)")
@app_commands.describe(question="Twoje pytanie")
async def slash_ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    
    channel_id = get_context_key(interaction)
    resp = await generate_response(channel_id, question, attachment=None, use_rag=True)
    final_msg = f"{interaction.user.mention}: {question}\n\n{resp}"
    await send_chunked_interaction(interaction, final_msg, ephemeral=False)

@bot.tree.command(name="status", description="Pokazuje aktualnie używany model")
async def slash_status(interaction: discord.Interaction):
    current_model = channel_models[get_context_key(interaction)]
    await interaction.response.send_message(
        f"Obecnie używany model to: **`{current_model}`**",
        ephemeral=True
    )

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
        await interaction.response.send_message(
            f"Błąd: `{chosen}` nie jest na liście ALLOWED_MODELS.",
            ephemeral=True
        )
        return

    async with _model_switch_lock:
        channel_models[get_context_key(interaction)] = chosen
        await interaction.response.send_message(f"Zmieniono model na: **`{chosen}`**", ephemeral=False)

@bot.tree.command(name="help", description="Lista komend")
async def slash_help(interaction: discord.Interaction):
    msg = (
        "**Dostępne komendy:**\n"
        "• `/chat prompt [obraz]` – Luźna rozmowa (obraz tylko przy Ministral).\n"
        "• `/ask question` – Pytanie do bazy wiedzy (RAG, Twoje pliki .txt).\n"
        "• `/model` – Zmiana modelu (Llama/Gemma/DeepSeek/Ministral).\n"
        "• `/listmodels` – Pokaż szczegóły modeli.\n"
        "• `/status` – Sprawdź, jaki model jest teraz włączony.\n"
        "• `/reset` – Wyczyść pamięć rozmowy w danym kanale.\n"
    )
    await interaction.response.send_message(msg, ephemeral=True)

@bot.tree.command(name="reset", description="Resetuje kontekst rozmowy")
async def slash_reset(interaction: discord.Interaction):
    channel_histories[get_context_key(interaction)].clear()
    await interaction.response.send_message("Pamięć wyczyszczona.", ephemeral=False)

# -------------- Start --------------
if __name__ == "__main__":
    DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
    if not DISCORD_TOKEN:
        print("Błąd: Nie ustawiono zmiennej środowiskowej DISCORD_TOKEN.")
    else:
        bot.run(DISCORD_TOKEN)