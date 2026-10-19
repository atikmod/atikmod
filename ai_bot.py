#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram AI Bot v3 - Speed Edition (Termux friendly)
Developer: MK Gang

What makes it fast:
- STREAMING: text shows up in Telegram as soon as the AI starts writing
  (no waiting for the full answer)
- "Thinking" is turned off / reduced - it is the biggest source of delay on newer models
- If a model is busy (503/429) it switches to another model within a second
- Connections are reused (no fresh TLS handshake on every request)
- 8 messages are processed at the same time (it was 2 before)
- Voice replies run in a separate thread - the text arrives first

Hosting: works on Termux, a VPS, or Render free web service (keep-alive built in).

Setup (Termux):
    pkg update && pkg install python -y
    pip install pyTelegramBotAPI requests gTTS
    export BOT_TOKEN="..."
    export GEMINI_API_KEY="..."
    export OWNER_ID="your Telegram ID"
    python ai_bot.py
"""

import os
import io
import re
import json
import html
import time
import base64
import functools
import threading
import requests
import telebot
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ================== Config ==================
# Set these as environment variables. Never hardcode real keys in the file.
BOT_TOKEN = os.getenv("BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PUT_YOUR_GEMINI_API_KEY_HERE")

TEXT_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash,gemini-flash-latest,gemini-3.1-flash,gemini-2.5-flash-lite"
).split(",") if m.strip()]

IMAGE_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_IMAGE_MODEL",
    "gemini-3.1-flash-image,gemini-3-pro-image,gemini-2.5-flash-image"
).split(",") if m.strip()]

OWNER_ID = int(os.getenv("OWNER_ID", "0"))

# "off" = fast (default), "on" = the model thinks deeply (slower, better on hard questions)
THINKING = os.getenv("GEMINI_THINKING", "off").lower()

SYSTEM_PROMPT = (
    "You are a friendly, intelligent AI assistant. Always reply in the same language the user "
    "writes or speaks in (Bengali -> Bengali, Hindi -> Hindi, English -> English, Arabic -> Arabic, "
    "and so on). If the user switches language, switch with them. "
    "Keep answers direct and short - no introductions or filler. "
    "When you receive a voice message, listen to it, understand it and answer directly.\n"
    "When asked for code: always give complete, runnable code - never cut it off halfway or leave "
    "it incomplete with '...'. Always put code inside fenced code blocks with the language "
    "(for example ```python). Keep explanations outside the code short. "
    "If there are multiple files, write the file name before each one."
)

MAX_HISTORY = 16
MAX_FILE_MB = 15
TTS_LANG_DEFAULT = "bn"
RETRY_CODES = {429, 500, 502, 503, 504}
STREAM_EDIT_INTERVAL = 1.2   # seconds between text updates in Telegram while streaming
# ============================================

bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None, threaded=True, num_threads=8)

history = {}
voice_mode = {}
tts_lang = {}
last_photo = {}
good_model = {"text": None, "image": None}
discovered = {"text": None, "image": None}
lock = threading.Lock()

BASE = "https://generativelanguage.googleapis.com/v1beta"

_local = threading.local()


def session():
    """One persistent connection per thread - saves repeated TLS handshakes."""
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        _local.s = s
    return s


class GeminiError(Exception):
    pass


class HttpErr(Exception):
    def __init__(self, code, msg):
        super().__init__(f"[{code}] {msg}")
        self.code, self.msg = code, msg


# The way to turn Thinking off differs between models - remember whichever one works
THINK_VARIANTS = [None] if THINKING == "on" else [
    {"thinkingBudget": 0},
    {"thinkingLevel": "minimal"},
    {"thinkingLevel": "low"},
    None,
]
think_idx = {}


def apply_thinking(payload, model, kind):
    if kind != "text":
        return payload
    v = THINK_VARIANTS[think_idx.get(model, 0)]
    if not v:
        return payload
    p = dict(payload)
    gc = dict(p.get("generationConfig", {}))
    gc["thinkingConfig"] = v
    p["generationConfig"] = gc
    return p


def bump_thinking(model):
    i = think_idx.get(model, 0)
    if i < len(THINK_VARIANTS) - 1:
        think_idx[model] = i + 1
        return True
    return False


# ==================== Gemini calls ====================
def _err_msg(r):
    try:
        return r.json().get("error", {}).get("message", r.text)[:300]
    except Exception:
        return r.text[:300]


def discover(kind):
    if discovered[kind] is not None:
        return discovered[kind]
    found = []
    try:
        r = session().get(f"{BASE}/models", params={"key": GEMINI_API_KEY, "pageSize": 200}, timeout=15)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                name = m.get("name", "").replace("models/", "")
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                if not name.startswith("gemini"):
                    continue
                low = name.lower()
                if kind == "image":
                    if "image" in low:
                        found.append(name)
                elif "flash" in low and not any(x in low for x in
                                                 ("image", "tts", "live", "audio", "embed", "native", "robotics")):
                    found.append(name)
            found.sort(reverse=True)
    except Exception:
        pass
    discovered[kind] = found
    return found


def _attempt(model, payload, stream, on_text):
    url = f"{BASE}/models/{model}:{'streamGenerateContent' if stream else 'generateContent'}"
    params = {"key": GEMINI_API_KEY}
    if stream:
        params["alt"] = "sse"
    r = session().post(url, params=params, json=payload, timeout=(10, 120), stream=stream)
    if r.status_code != 200:
        raise HttpErr(r.status_code, _err_msg(r))
    if not stream:
        return r.json()

    acc, finish, block = "", None, None
    try:
        for raw in r.iter_lines(chunk_size=1):
            if not raw or not raw.startswith(b"data:"):
                continue
            try:
                j = json.loads(raw[5:].decode("utf-8").strip())
            except Exception:
                continue
            if j.get("promptFeedback", {}).get("blockReason"):
                block = j["promptFeedback"]["blockReason"]
            cands = j.get("candidates") or []
            if not cands:
                continue
            finish = cands[0].get("finishReason") or finish
            for p in cands[0].get("content", {}).get("parts", []):
                if p.get("text") and not p.get("thought"):
                    acc += p["text"]
            if on_text and acc:
                on_text(acc)
    except requests.RequestException:
        if not acc:
            raise
        return {"text": acc + "\n\n⚠️ (The answer was cut off midway)", "finish": "CUT", "block": None}
    return {"text": acc, "finish": finish, "block": block}


def call_gemini(payload, kind="text", stream=False, on_text=None):
    base = TEXT_MODELS if kind == "text" else IMAGE_MODELS
    if good_model[kind]:
        base = [good_model[kind]] + [m for m in base if m != good_model[kind]]

    dead, last_err, busy = set(), "Unknown problem", False
    for rnd in range(2):                      # at most 2 rounds
        busy_this_round = False
        tried = set()
        for stage in (0, 1):
            for model in (base if stage == 0 else discover(kind)):
                if model in dead or model in tried:
                    continue
                tried.add(model)
                while True:                   # loop only used to change the thinking setting
                    try:
                        res = _attempt(model, apply_thinking(payload, model, kind), stream, on_text)
                        good_model[kind] = model
                        return res, model
                    except requests.RequestException as e:
                        last_err = f"Network problem ({type(e).__name__})"
                        busy_this_round = True
                        break
                    except HttpErr as e:
                        low = e.msg.lower()
                        last_err = str(e)
                        if e.code == 400 and "think" in low and kind == "text" and bump_thinking(model):
                            continue
                        if e.code in (400, 401, 403) and ("api key" in low or "api_key" in low
                                                          or "permission" in low):
                            raise GeminiError("🔑 The Gemini API key is wrong or has no permission. Check your key.\n" + e.msg)
                        if e.code in RETRY_CODES:
                            busy_this_round = True
                            if "quota" in low and ("per day" in low or "daily" in low):
                                dead.add(model)
                        else:
                            dead.add(model)   # 404/400 -> skip this model from now on
                        break
        busy = busy or busy_this_round
        if not busy_this_round:
            break
        if rnd == 0:
            time.sleep(1.5)                   # every model was busy - pause briefly and try again

    if busy:
        raise GeminiError("⏳ Google's AI is very busy right now or the limit is used up. "
                          "Please send again in a few seconds.\n"
                          f"(Details: {last_err})")
    raise GeminiError(f"⚠️ No model could be run.\n({last_err})")


def ask_gemini(chat_id, user_text, media=None, on_text=None):
    with lock:
        past = list(history.get(chat_id, []))
    contents = [{"role": h["role"], "parts": [{"text": h["text"]}]} for h in past]

    parts = []
    if media:
        mime, data = media
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
    parts.append({"text": user_text or "Look at this file and answer / explain it."})
    contents.append({"role": "user", "parts": parts})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 8192},
    }
    try:
        res, _ = call_gemini(payload, "text", stream=True, on_text=on_text)
    except GeminiError as e:
        return False, str(e)

    answer = (res.get("text") or "").strip()
    if not answer:
        if res.get("block"):
            return False, f"🚫 Request blocked: {res['block']}"
        return False, f"Sorry, no answer was generated (reason: {res.get('finish') or 'unknown'})."

    saved_user = user_text if user_text else "[A file/voice message was sent]"
    if media and user_text:
        saved_user = "[With file] " + user_text
    with lock:
        h = history.setdefault(chat_id, [])
        h.append({"role": "user", "text": saved_user})
        h.append({"role": "model", "text": answer})
        history[chat_id] = h[-MAX_HISTORY:]
    return True, answer


def gen_image(prompt, image=None):
    parts = []
    if image:
        parts.append({"inline_data": {"mime_type": image[0],
                                      "data": base64.b64encode(image[1]).decode()}})
    parts.append({"text": prompt})
    payload = {"contents": [{"role": "user", "parts": parts}],
               "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}}
    j, _ = call_gemini(payload, "image")

    imgs, txt = [], ""
    cands = j.get("candidates") or []
    if not cands:
        raise GeminiError("🚫 Request blocked: " + j.get("promptFeedback", {}).get("blockReason", "unknown"))
    for p in cands[0].get("content", {}).get("parts", []):
        d = p.get("inlineData") or p.get("inline_data")
        if d and d.get("data"):
            imgs.append((d.get("mimeType") or d.get("mime_type") or "image/png", base64.b64decode(d["data"])))
        elif p.get("text"):
            txt += p["text"]
    if not imgs:
        raise GeminiError("No image was created. The model said: " + (txt[:300] or "nothing"))
    return imgs, txt.strip()


# ==================== Formatting ====================
EXT = {"python": "py", "py": "py", "javascript": "js", "js": "js", "typescript": "ts", "ts": "ts",
       "bash": "sh", "sh": "sh", "shell": "sh", "java": "java", "kotlin": "kt", "html": "html",
       "css": "css", "json": "json", "cpp": "cpp", "c++": "cpp", "c": "c", "php": "php",
       "sql": "sql", "go": "go", "rust": "rs", "xml": "xml", "yaml": "yml", "yml": "yml",
       "dart": "dart", "swift": "swift", "ruby": "rb", "markdown": "md", "md": "md"}


def split_segments(text):
    segs, pos = [], 0
    for m in re.finditer(r"```([\w+#.\-]*)[ \t]*\n?(.*?)```", text, re.S):
        if m.start() > pos:
            segs.append(("text", text[pos:m.start()]))
        segs.append(("code", m.group(1) or "", m.group(2).rstrip("\n")))
        pos = m.end()
    if pos < len(text):
        segs.append(("text", text[pos:]))
    return segs


def fmt_inline(t):
    out = []
    for p in re.split(r"(`[^`\n]+`)", t):
        if len(p) >= 2 and p.startswith("`") and p.endswith("`"):
            out.append("<code>" + html.escape(p[1:-1]) + "</code>")
        else:
            p = html.escape(p)
            p = re.sub(r"(?m)^(\s*)[\*\-]\s+", r"\1• ", p)
            p = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", p)
            p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p, flags=re.S)
            out.append(p)
    return "".join(out)


def split_text(t, n=3000):
    out = []
    while len(t) > n:
        cut = t.rfind("\n", 0, n)
        if cut < n // 2:
            cut = n
        out.append(t[:cut])
        t = t[cut:]
    out.append(t)
    return out


def render_answer(answer):
    """Split the answer into Telegram HTML chunks and large code files."""
    chunks, files, cur, n = [], [], "", 0

    def add(h):
        nonlocal cur
        if cur and len(cur) + len(h) + 1 > 3800:
            chunks.append(cur)
            cur = ""
        cur = (cur + "\n" + h) if cur else h

    for seg in split_segments(answer or ""):
        if seg[0] == "text":
            if not seg[1].strip():
                continue
            for piece in split_text(seg[1].strip("\n")):
                add(fmt_inline(piece))
        else:
            _, lang, code = seg
            if len(code) > 3000:
                n += 1
                name = f"code_{n}.{EXT.get(lang.lower(), 'txt')}"
                files.append((name, code))
                add(f"📎 <i>The code is large, so it was sent as a file: {name}</i>")
            else:
                sl = re.sub(r"[^\w+#.\-]", "", lang)
                cls = f' class="language-{sl}"' if sl else ""
                add(f"<pre><code{cls}>{html.escape(code)}</code></pre>")
    if cur:
        chunks.append(cur)
    if not chunks and not files:
        chunks = ["(empty answer)"]
    return chunks, files


def to_plain(c):
    return html.unescape(re.sub(r"<[^>]+>", "", c))


def send_rendered(chat_id, chunks, files, reply_to=None):
    first = True
    for c in chunks:
        rt = reply_to if first else None
        try:
            bot.send_message(chat_id, c, parse_mode="HTML", reply_to_message_id=rt)
        except Exception:
            bot.send_message(chat_id, to_plain(c)[:4000], reply_to_message_id=rt)
        first = False
    for name, code in files:
        b = io.BytesIO(code.encode("utf-8"))
        b.name = name
        bot.send_document(chat_id, b)


class Live:
    """Shows the answer in Telegram while it is still being written (streaming)."""

    def __init__(self, m):
        self.chat, self.reply_to = m.chat.id, m.message_id
        self.msg_id, self.last, self.shown = None, 0.0, ""
        self.interval = STREAM_EDIT_INTERVAL

    def update(self, text):
        if not text.strip():
            return
        view = text if len(text) <= 3900 else "…" + text[-3800:]
        now = time.time()
        try:
            if self.msg_id is None:
                sent = bot.send_message(self.chat, view, reply_to_message_id=self.reply_to)
                self.msg_id, self.shown, self.last = sent.message_id, view, now
                return
            if now - self.last < self.interval or view == self.shown:
                return
            bot.edit_message_text(view, self.chat, self.msg_id)
            self.shown = view
        except Exception as e:
            if "429" in str(e) or "too many" in str(e).lower():
                self.interval = min(self.interval * 2, 6)
        self.last = time.time()

    def finish(self, ok, answer):
        if not ok:
            if self.msg_id:
                try:
                    bot.edit_message_text(answer[:4000], self.chat, self.msg_id)
                    return
                except Exception:
                    pass
            bot.send_message(self.chat, answer[:4000], reply_to_message_id=self.reply_to)
            return
        chunks, files = render_answer(answer)
        if self.msg_id and len(chunks) == 1 and not files:
            try:
                bot.edit_message_text(chunks[0], self.chat, self.msg_id, parse_mode="HTML")
            except Exception as e:
                if "not modified" in str(e).lower():
                    return
                try:
                    bot.edit_message_text(to_plain(chunks[0])[:4000], self.chat, self.msg_id)
                except Exception:
                    pass
            return
        if self.msg_id:
            try:
                bot.delete_message(self.chat, self.msg_id)
            except Exception:
                pass
        send_rendered(self.chat, chunks, files, self.reply_to)


# ==================== Voice ====================
def clean_for_tts(text):
    text = re.sub(r"```.*?```", " (code provided) ", text, flags=re.S)
    text = re.sub(r"[`*#_>|]", "", text)
    return text.strip()


def reply_voice(chat_id, text):
    text = clean_for_tts(text)
    if not text:
        return
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text[:1500], lang=tts_lang.get(chat_id, TTS_LANG_DEFAULT)).write_to_fp(buf)
        buf.seek(0)
        buf.name = "reply.mp3"
        bot.send_audio(chat_id, buf, title="AI Voice Reply")
    except Exception as e:
        bot.send_message(chat_id, f"⚠️ Problem creating the voice reply: {e}")


# ==================== Helpers ====================
def allowed(m):
    return OWNER_ID == 0 or m.from_user.id == OWNER_ID


def safe(fn):
    @functools.wraps(fn)
    def wrapper(m, *a, **k):
        try:
            if not allowed(m):
                return
            return fn(m, *a, **k)
        except Exception as e:
            try:
                bot.send_message(m.chat.id, f"⚠️ Unexpected problem: {type(e).__name__}: {e}")
            except Exception:
                pass
    return wrapper


def respond(m, text, media=None):
    """Ask the AI and show the answer with streaming."""
    show_typing(m.chat.id)
    live = Live(m)
    ok, answer = ask_gemini(m.chat.id, text, media, on_text=live.update)
    live.finish(ok, answer)
    if ok and voice_mode.get(m.chat.id):
        threading.Thread(target=reply_voice, args=(m.chat.id, answer), daemon=True).start()


def download(file_id, size=None):
    if size and size > MAX_FILE_MB * 1024 * 1024:
        return None
    info = bot.get_file(file_id)
    return bot.download_file(info.file_path)


def show_typing(chat_id, action="typing"):
    try:
        bot.send_chat_action(chat_id, action)
    except Exception:
        pass


# Edit trigger words: /edit, edit, and the Bengali words (এডিট / ছবি এডিট) so Bengali users can still use them
EDIT_PREFIX = re.compile(r"^\s*(/edit|edit\s*:?|এডিট\s*:?|ছবি\s*এডিট\s*:?)\s*", re.I)


def is_edit(text):
    return bool(text and EDIT_PREFIX.match(text))


def do_edit(m, instruction, img):
    show_typing(m.chat.id, "upload_photo")
    wait = bot.reply_to(m, "🎨 Editing the image, please wait...")
    try:
        imgs, txt = gen_image(instruction or "Make the image look better", img)
    except GeminiError as e:
        bot.reply_to(m, str(e) + "\n\nℹ️ Image models often work only with a paid API key.")
        return
    finally:
        try:
            bot.delete_message(m.chat.id, wait.message_id)
        except Exception:
            pass
    for mime, data in imgs:
        ext = "jpg" if "jpeg" in mime else mime.split("/")[-1]
        b = io.BytesIO(data)
        b.name = f"edited.{ext}"
        bot.send_photo(m.chat.id, b, caption=(txt[:900] if txt else None), reply_to_message_id=m.message_id)
        b2 = io.BytesIO(data)
        b2.name = f"edited.{ext}"
        bot.send_document(m.chat.id, b2)
        last_photo[m.chat.id] = (mime, data)


# ==================== Commands ====================
@bot.message_handler(commands=["start", "help"])
@safe
def cmd_start(m):
    bot.reply_to(m, (
        "👋 I'm your AI bot!\n\n"
        "✍️ Ask anything / ask for code - tap the code to copy it\n"
        "🎤 Send a voice message - I'll listen and reply\n"
        "📎 Send a photo / PDF / text / code file - I'll read and explain it\n"
        "🎨 Edit a photo: send it with the caption  /edit make the background blue\n"
        "   or after sending a photo write: /edit make the eyes bigger\n"
        "🖼 New image: /image a beautiful mountain scene\n\n"
        "Commands:\n"
        "/voice on|off - get replies as voice or not\n"
        "/lang bn|en|hi - voice language\n"
        "/sendfile path - send a file from your phone\n"
        "/clear - clear the conversation\n"
        "/status - see which model is running\n"
        f"\nYour ID: {m.from_user.id}"
    ))


@bot.message_handler(commands=["clear"])
@safe
def cmd_clear(m):
    with lock:
        history.pop(m.chat.id, None)
    last_photo.pop(m.chat.id, None)
    bot.reply_to(m, "🧹 Conversation cleared.")


@bot.message_handler(commands=["status"])
@safe
def cmd_status(m):
    bot.reply_to(m, f"Text model: {good_model['text'] or 'not used yet'}\n"
                    f"Image model: {good_model['image'] or 'not used yet'}\n"
                    f"Thinking: {THINKING}")


@bot.message_handler(commands=["voice"])
@safe
def cmd_voice(m):
    arg = (m.text.split(maxsplit=1) + [""])[1].strip().lower()
    if arg in ("on", "অন"):
        voice_mode[m.chat.id] = True
        bot.reply_to(m, "🔊 Voice replies turned on.")
    elif arg in ("off", "অফ"):
        voice_mode[m.chat.id] = False
        bot.reply_to(m, "🔇 Voice replies turned off.")
    else:
        bot.reply_to(m, "Usage: /voice on  or  /voice off")


@bot.message_handler(commands=["lang"])
@safe
def cmd_lang(m):
    arg = (m.text.split(maxsplit=1) + [""])[1].strip().lower()
    if arg:
        tts_lang[m.chat.id] = arg
        bot.reply_to(m, f"🌐 Voice language: {arg}")
    else:
        bot.reply_to(m, "Usage: /lang bn  (or en, hi, ar ...)")


@bot.message_handler(commands=["sendfile"])
@safe
def cmd_sendfile(m):
    if OWNER_ID == 0 or m.from_user.id != OWNER_ID:
        bot.reply_to(m, "🔒 You must set OWNER_ID for this command (otherwise anyone could take files from your phone).")
        return
    arg = (m.text.split(maxsplit=1) + [""])[1].strip()
    if not arg:
        bot.reply_to(m, "Usage: /sendfile /path/to/file")
        return
    path = os.path.expanduser(arg)
    if not os.path.isfile(path):
        bot.reply_to(m, "❌ File not found.")
        return
    show_typing(m.chat.id, "upload_document")
    with open(path, "rb") as f:
        bot.send_document(m.chat.id, f, reply_to_message_id=m.message_id)


@bot.message_handler(commands=["image"])
@safe
def cmd_image(m):
    prompt = (m.text.split(maxsplit=1) + [""])[1].strip()
    if not prompt:
        bot.reply_to(m, "Usage: /image a beautiful sunset")
        return
    show_typing(m.chat.id, "upload_photo")
    try:
        imgs, txt = gen_image(prompt)
    except GeminiError as e:
        bot.reply_to(m, str(e) + "\n\nℹ️ Image models often work only with a paid API key.")
        return
    for mime, data in imgs:
        ext = "jpg" if "jpeg" in mime else mime.split("/")[-1]
        b = io.BytesIO(data)
        b.name = f"image.{ext}"
        bot.send_photo(m.chat.id, b, caption=(txt[:900] if txt else None), reply_to_message_id=m.message_id)
        last_photo[m.chat.id] = (mime, data)


@bot.message_handler(commands=["edit"])
@safe
def cmd_edit(m):
    instruction = EDIT_PREFIX.sub("", m.text, count=1).strip()
    img = None
    r = m.reply_to_message
    if r and r.photo:
        data = download(r.photo[-1].file_id, r.photo[-1].file_size)
        if data:
            img = ("image/jpeg", data)
    if img is None:
        img = last_photo.get(m.chat.id)
    if img is None:
        bot.reply_to(m, "Send a photo first (or write /edit as a reply to a photo).")
        return
    if not instruction:
        bot.reply_to(m, "Tell me what to edit. Example: /edit make the background white")
        return
    do_edit(m, instruction, img)


# ==================== Text / voice / photo / file ====================
@bot.message_handler(content_types=["text"])
@safe
def on_text(m):
    if is_edit(m.text) and m.chat.id in last_photo:
        do_edit(m, EDIT_PREFIX.sub("", m.text, count=1).strip(), last_photo[m.chat.id])
        return
    respond(m, m.text)


@bot.message_handler(content_types=["voice", "audio"])
@safe
def on_voice(m):
    show_typing(m.chat.id)
    obj = m.voice if m.content_type == "voice" else m.audio
    data = download(obj.file_id, getattr(obj, "file_size", None))
    if not data:
        bot.reply_to(m, f"❌ File is too large (max {MAX_FILE_MB}MB).")
        return
    mime = getattr(obj, "mime_type", None) or "audio/ogg"
    prompt = m.caption or ("Listen to this audio and directly answer / solve what the user said, "
                           "in the same language the user spoke.")
    respond(m, prompt, media=(mime, data))


@bot.message_handler(content_types=["photo"])
@safe
def on_photo(m):
    photo = m.photo[-1]
    data = download(photo.file_id, photo.file_size)
    if not data:
        bot.reply_to(m, f"❌ File is too large (max {MAX_FILE_MB}MB).")
        return
    last_photo[m.chat.id] = ("image/jpeg", data)
    cap = m.caption or ""
    if is_edit(cap):
        do_edit(m, EDIT_PREFIX.sub("", cap, count=1).strip(), ("image/jpeg", data))
        return
    if not cap:
        bot.reply_to(m, "📷 Got the photo. What should I do?\n"
                        "• Ask a question about the photo\n"
                        "• To edit it, write: /edit your instruction")
        return
    respond(m, cap, media=("image/jpeg", data))


@bot.message_handler(content_types=["document"])
@safe
def on_document(m):
    show_typing(m.chat.id)
    d = m.document
    data = download(d.file_id, d.file_size)
    if not data:
        bot.reply_to(m, f"❌ File is too large (max {MAX_FILE_MB}MB).")
        return
    name = (d.file_name or "").lower()
    mime = d.mime_type or "application/octet-stream"
    text_ext = (".txt", ".md", ".py", ".js", ".json", ".csv", ".html", ".css", ".xml", ".log",
                ".java", ".c", ".cpp", ".sh", ".yml", ".yaml", ".kt", ".php", ".ts", ".sql")

    if mime.startswith("image/"):
        last_photo[m.chat.id] = (mime, data)
        cap = m.caption or ""
        if is_edit(cap):
            do_edit(m, EDIT_PREFIX.sub("", cap, count=1).strip(), (mime, data))
            return
        respond(m, cap or "Describe this image.", media=(mime, data))
    elif name.endswith(text_ext) or mime.startswith("text/"):
        content = data.decode("utf-8", errors="ignore")[:100000]
        q = m.caption or "Read this file and give a summary and explanation."
        respond(m, f"{q}\n\n--- File: {d.file_name} ---\n{content}")
    elif mime == "application/pdf" or mime.startswith(("audio/", "video/")):
        respond(m, m.caption or "Analyze this file and explain it.", media=(mime, data))
    else:
        bot.reply_to(m, "❌ This file type is not supported (send a PDF, image, audio, or a text/code file).")


# ==================== Keep-alive (Render free plan) ====================
# Render puts a free web service to sleep after 15 minutes without inbound HTTP
# traffic. A polling bot gets none, so we (1) open a tiny web server on $PORT and
# (2) ping our own public URL every 10 minutes. Outside Render (no PORT set) this
# does nothing, so Termux / VPS behave exactly as before.
class Health(BaseHTTPRequestHandler):
    def _ok(self, body=True):
        data = b"OK - bot is running"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if body:
            self.wfile.write(data)

    def do_GET(self):
        self._ok()

    def do_HEAD(self):
        self._ok(body=False)

    def log_message(self, *args):   # keep the logs clean
        pass


def _ping_loop(url):
    while True:
        time.sleep(600)   # 10 minutes (Render sleeps at 15)
        try:
            requests.get(url, timeout=20)
        except Exception as e:
            print("Keep-alive ping failed:", e)


def start_keepalive():
    port = os.getenv("PORT")
    if not port:
        return
    server = ThreadingHTTPServer(("0.0.0.0", int(port)), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"🌐 Health server on port {port}")
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEPALIVE_URL")
    if url:
        threading.Thread(target=_ping_loop, args=(url,), daemon=True).start()
        print(f"🔁 Self-ping every 10 min -> {url}")


# ==================== Start ====================
if __name__ == "__main__":
    if "PUT_YOUR" in BOT_TOKEN or "PUT_YOUR" in GEMINI_API_KEY:
        print("❌ Set BOT_TOKEN and GEMINI_API_KEY first.")
        raise SystemExit(1)
    start_keepalive()
    print("✅ Bot started... (press Ctrl+C to stop)")
    while True:
        try:
            bot.infinity_polling(timeout=30, long_polling_timeout=30, skip_pending=True)
        except Exception as e:
            print("Restarting:", e)
            time.sleep(5)
