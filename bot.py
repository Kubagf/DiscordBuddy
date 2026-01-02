import os
import json
import base64
import asyncio
import mimetypes
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional

import aiohttp
import discord
from discord.ext import commands
from discord import app_commands

# -------------- Configuration --------------
DEFAULT_MODEL = "gemma3:12b"

# Allowed models
ALLOWED_MODELS: Dict[str, str] = {
    "gemma3:12b":     "Higher quality, heavier.",
    "llama3:8b":      "Good general chat (alias: llama:8b).",
    "deepseek-r1:8b": "DeepSeek 8B (alias: deepseek:8b).",
    "ministral-3:8b":       "Vision model (images).",
}

# Aliases
MODEL_ALIASES: Dict[str, str] = {
    "gemma":    "gemma3:12b",
    "llama":    "llama3:8b",
    "deepseek": "deepseek-r1:8b",
    "ministral":"ministral-3:8b",
}

# Vision setup
VISION_SUGGESTED_MODEL = "ministral-3:8b"
VISION_MODELS = {"ministral-3:8b"}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
SYSTEM_PROMPT = (
    "You are a friendly, concise assistant for a Discord server. "
    "Be helpful, truthful, and avoid overly long answers unless asked."
    "Answer in the same language as the user."
)
MAX_HISTORY = 8
DISCORD_MESSAGE_LIMIT = 1900

GUILD_ID_ENV = os.environ.get("GUILD_ID")
GUILD_OBJECT = discord.Object(id=int(GUILD_ID_ENV)) if GUILD_ID_ENV else None

# -------------- State --------------
channel_histories: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
channel_models: Dict[int, str] = defaultdict(lambda: DEFAULT_MODEL)
_model_switch_lock = asyncio.Lock()  # serialize model switches

# -------------- Ollama HTTP (generate) --------------
async def ollama_generate_text(session: aiohttp.ClientSession, model: str, prompt: str, options: Optional[dict] = None) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data.get("response", "") or ""

async def ollama_stream_generate(
    session: aiohttp.ClientSession,
    model: str,
    prompt: str,
    history: List[Tuple[str, str]],
):
    lines: List[str] = [f"[System]: {SYSTEM_PROMPT}"]
    for role, content in history:
        lines.append(f"[User]: {content}" if role == "user" else f"[Assistant]: {content}")
    lines.append(f"[User]: {prompt}")
    lines.append("[Assistant]:")
    stitched = "\n".join(lines)

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model, "prompt": stitched, "stream": True}

    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
                if "response" in obj and obj["response"]:
                    yield obj["response"]
                if obj.get("done"):
                    break
            except Exception:
                continue  # skip malformed chunk safely

async def ollama_vision_generate_text(
    session: aiohttp.ClientSession,
    model: str,
    prompt: str,
    history: List[Tuple[str, str]],
    images: List[bytes],
) -> str:
    lines: List[str] = [f"[System]: {SYSTEM_PROMPT}"]
    for role, content in history:
        lines.append(f"[User]: {content}" if role == "user" else f"[Assistant]: {content}")
    lines.append(f"[User]: {prompt}")
    lines.append("[Assistant]:")
    stitched = "\n".join(lines)

    images_b64 = [base64.b64encode(b).decode("ascii") for b in images]
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": stitched,
        "images": images_b64,
        "stream": False,
    }
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data.get("response", "") or ""

# -------------- Ollama CLI / REST helpers --------------
async def run_ollama_cli(args: List[str]) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out_b, err_b = await proc.communicate()
    return proc.returncode, out_b.decode(errors="ignore"), err_b.decode(errors="ignore")

async def pull_model_via_rest(model: str) -> None:
    url = f"{OLLAMA_HOST}/api/pull"
    payload = {"name": model}
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for _ in resp.content:
                pass

async def stop_model_best_effort(old_model: str) -> str:
    try:
        rc, out, err = await run_ollama_cli(["ollama", "stop", old_model])
        if rc == 0:
            return f"Stopped previous model: {old_model}."
        return f"Stop previous model skipped/failed (non-fatal): {(err.strip() or out.strip() or 'no details')}"
    except FileNotFoundError:
        return "Stop previous model skipped (ollama CLI not found; non-fatal)."

async def ensure_model_available(new_model: str, old_model: Optional[str]) -> str:
    messages: List[str] = []

    if old_model and old_model != new_model:
        messages.append(await stop_model_best_effort(old_model))

    try:
        messages.append(f"Pulling model: {new_model} …")
        try:
            rc, out, err = await run_ollama_cli(["ollama", "pull", new_model])
            if rc != 0:
                raise RuntimeError(err.strip() or out.strip() or "pull failed")
        except FileNotFoundError:
            await pull_model_via_rest(new_model)
    except Exception as e:
        raise RuntimeError(f"Failed to pull {new_model}: {e}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        try:
            _ = await ollama_generate_text(session, new_model, "warmup")
            messages.append("Warm-up done.")
        except Exception as e:
            err_txt = str(e)
            if "CUDA" in err_txt or "cuda" in err_txt:
                messages.append("Warm-up failed (CUDA). Possibly not enough VRAM.")
            else:
                messages.append(f"Warm-up failed (non-fatal): {e}")

    return "\n".join(messages)

# -------------- Discord bot setup --------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# -------------- Helpers --------------
async def send_long_message(channel: discord.abc.Messageable, text: str):
    if not text:
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(start + DISCORD_MESSAGE_LIMIT, n)
        await channel.send(text[start:end])
        start = end

async def run_chat(
    channel: discord.abc.Messageable,
    channel_id: int,
    prompt: str,
    images: Optional[List[bytes]] = None
):
    model = channel_models[channel_id]
    async with channel.typing():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            history = list(channel_histories[channel_id])

            # Vision branch
            if images and model in VISION_MODELS:
                try:
                    answer = await ollama_vision_generate_text(session, model, prompt, history, images)
                    await send_long_message(channel, answer)
                    channel_histories[channel_id].append(("user", f"{prompt} [image x{len(images)}]"))
                    channel_histories[channel_id].append(("assistant", answer))
                    return
                except Exception as e:
                    await channel.send(f"Vision error: {e}")
                    return

            # Inform if image present but model is not vision
            if images and model not in VISION_MODELS:
                await channel.send(
                    f"Detected an image, but current model `{model}` is not vision-capable. "
                    f"Use `/setmodel name:{VISION_SUGGESTED_MODEL}` and try again."
                )
                # continue with text-only answer below

            # Text-only streaming
            buffer: List[str] = []
            current_chunk = ""
            try:
                async for chunk in ollama_stream_generate(session, model, prompt, history):
                    buffer.append(chunk)
                    current_chunk += chunk
                    if len(current_chunk) > 500:
                        await send_long_message(channel, current_chunk)
                        current_chunk = ""
                if current_chunk:
                    await send_long_message(channel, current_chunk)

                assistant_text = "".join(buffer).strip()
                channel_histories[channel_id].append(("user", prompt))
                channel_histories[channel_id].append(("assistant", assistant_text))
            except aiohttp.ClientResponseError as e:
                await channel.send(f"Ollama HTTP error: {e.status} {e.message}")
            except asyncio.TimeoutError:
                await channel.send("Timeout while talking to the model. Try again or shorten your prompt.")
            except Exception as e:
                await channel.send(f"Error: {e}")

def _is_image_attachment(att: discord.Attachment) -> bool:
    ctype = att.content_type or mimetypes.guess_type(att.filename or "")[0] or ""
    if ctype.startswith("image/"):
        return True
    name = (att.filename or "").lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))

# -------------- Events --------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    try:
        if GUILD_OBJECT:
            synced = await bot.tree.sync(guild=GUILD_OBJECT)
            print(f"App commands synced for guild {GUILD_OBJECT.id}: {len(synced)}")
        else:
            synced = await bot.tree.sync()
            print(f"Global app commands synced: {len(synced)}")
    except Exception as e:
        print(f"Slash command sync failed: {e}")
    print("Ready.")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.attachments:
        if any(_is_image_attachment(att) for att in message.attachments):
            current_model = channel_models.get(message.channel.id, DEFAULT_MODEL)
            if current_model not in VISION_MODELS:
                await message.channel.send(
                    f"{message.author.mention} Detected an image. Current model (`{current_model}`) "
                    f"does not support vision. Switch with `/setmodel name:{VISION_SUGGESTED_MODEL}`."
                )
    await bot.process_commands(message)

# -------------- Slash commands --------------
@bot.tree.command(name="help", description="Show available commands")
async def slash_help(interaction: discord.Interaction):
    msg = (
        "**Commands**\n"
        "/chat message:<text> image:<attachment?> — chat (supports image if vision model is active)\n"
        "/listmodels — show allowed models\n"
        "/setmodel name:<model> — change model (whitelist, auto pull)\n"
        "/model — show current model\n"
        "/reset — clear context\n"
    )
    await interaction.response.send_message(msg, ephemeral=True)

@bot.tree.command(name="listmodels", description="Show allowed models")
async def slash_listmodels(interaction: discord.Interaction):
    lines = [f"- `{name}` — {desc}" for name, desc in ALLOWED_MODELS.items()]
    await interaction.response.send_message("Allowed models:\n" + "\n".join(lines), ephemeral=True)

@bot.tree.command(name="setmodel", description="Stop old (best-effort), pull & warm the new model for this channel")
async def slash_setmodel(interaction: discord.Interaction, name: str):
    await interaction.response.defer(thinking=True, ephemeral=True)

    ch = interaction.channel
    if ch is None:
        await interaction.followup.send("This command must be used in a channel.", ephemeral=True)
        return

    raw = name.strip()
    cand = MODEL_ALIASES.get(raw.lower(), raw)
    if cand not in ALLOWED_MODELS:
        lines = [f"- `{k}` — {v}" for k, v in ALLOWED_MODELS.items()]
        await interaction.followup.send(
            "This model is not allowed. Available options:\n" + "\n".join(lines),
            ephemeral=True
        )
        return

    new_model = cand
    old_model = channel_models.get(ch.id, DEFAULT_MODEL)
    if old_model == new_model:
        await interaction.followup.send(f"Model `{new_model}` is already set for this channel.", ephemeral=True)
        return

    async with _model_switch_lock:
        try:
            summary = await ensure_model_available(new_model, old_model)
            channel_models[ch.id] = new_model
            await interaction.followup.send(
                f"Model switched to `{new_model}` for this channel.\n{summary}",
                ephemeral=True
            )
        except Exception as e:
            await interaction.followup.send(f"Model switch failed: {e}", ephemeral=True)

@bot.tree.command(name="model", description="Show current model for this channel")
async def slash_model(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Current model: `{channel_models.get(interaction.channel.id, DEFAULT_MODEL)}`",
        ephemeral=True
    )

@bot.tree.command(name="reset", description="Clear channel memory (short context)")
async def slash_reset(interaction: discord.Interaction):
    channel_histories[interaction.channel.id].clear()
    await interaction.response.send_message("Channel memory cleared.", ephemeral=True)

@bot.tree.command(name="chat", description="Chat with the current model (optionally with an image)")
async def slash_chat(interaction: discord.Interaction, message: str, image: Optional[discord.Attachment] = None):
    await interaction.response.defer(thinking=True, ephemeral=True)
    imgs: List[bytes] = []
    if image is not None:
        try:
            imgs.append(await image.read())
        except Exception:
            pass
    await run_chat(interaction.channel, interaction.channel.id, message, images=imgs)
    await interaction.followup.send("Response sent to this channel.", ephemeral=True)

# -------------- Optional prefix commands --------------
@bot.command(name="chat")
async def prefix_chat(ctx: commands.Context, *, prompt: str):
    imgs: List[bytes] = []
    try:
        if ctx.message.attachments:
            for att in ctx.message.attachments:
                if _is_image_attachment(att):
                    imgs.append(await att.read())
    except Exception:
        pass
    await run_chat(ctx.channel, ctx.channel.id, prompt, images=imgs)

@bot.command(name="reset")
async def prefix_reset(ctx: commands.Context):
    channel_histories[ctx.channel.id].clear()
    await ctx.send("Channel memory cleared.")

# -------------- Entry point --------------
if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Please set DISCORD_TOKEN env var.")
    bot.run(token)
