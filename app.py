import os
import time
import uuid
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# ------------------ Setup ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(page_title="Climate Education Chatbot", page_icon="🌍", layout="wide")

SYSTEM_PROMPT = (
    "You are a Climate Education tutor. Be scientifically accurate and plain-spoken. "
    "Keep answers concise and age-appropriate. Avoid fear-mongering; focus on clarity and empowerment. "
    "Cite concepts (not URLs) briefly when useful (e.g., “per NASA data”)."
)

AGE_GROUPS = ["Elementary (8–11)", "Middle School (11–14)", "High School (14–18)", "General Audience"]
MODELS = ["OpenAI (GPT-4o-mini)", "Gemini (1.5-flash)"]

# --- Trusted facts bank (demo — swap with NASA/NOAA/IPCC later) ---
FACTS_BANK = [
    {"text": "CO2 from burning fossil fuels is the largest human driver of current climate change.", "source": "IPCC AR6"},
    {"text": "Global average surface temperatures have risen ~1.1°C since the late 19th century.", "source": "NASA"},
    {"text": "Sea level is rising due to ocean warming and melting land ice.", "source": "NOAA"},
    {"text": "Cutting energy waste (LEDs, efficient appliances) directly reduces emissions.", "source": "US DOE"},
    {"text": "Public transit, biking, and walking lower transportation emissions vs single-occupancy driving.", "source": "EPA"},
]


# ------------------ State helpers ------------------
def ensure_state():
    # chats: list of {id,title,model,age_group,messages:[{role,content}],updated_at,created_at}
    if "chats" not in st.session_state:
        st.session_state.chats: List[Dict] = []
    if "active_id" not in st.session_state:
        st.session_state.active_id = None


def new_chat(model: str, age_group: str):
    """Create a fresh chat and make it active."""
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats.insert(
        0,
        {
            "id": chat_id,
            "title": "New chat",  # placeholder; updated after first user message
            "model": model,
            "age_group": age_group,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "created_at": time.time(),
            "updated_at": time.time(),
        },
    )
    st.session_state.active_id = chat_id


def get_active():
    for c in st.session_state.chats:
        if c["id"] == st.session_state.active_id:
            return c
    return None


def set_active(chat_id: str):
    st.session_state.active_id = chat_id


def title_from_first_user(chat: Dict) -> str:
    for m in chat["messages"]:
        if m["role"] == "user":
            t = m["content"].splitlines()[0].strip()
            return (t[:40] + "…") if len(t) > 40 else t
    return "New chat"


def prune_empty_chats():
    """Remove chats that were created but never used (only system message)."""
    kept = [c for c in st.session_state.chats if not (c["title"] == "New chat" and len(c["messages"]) == 1)]
    st.session_state.chats = kept
    if st.session_state.active_id and not any(c["id"] == st.session_state.active_id for c in kept):
        st.session_state.active_id = kept[0]["id"] if kept else None


# ------------------ Providers ------------------
def get_openai() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def call_openai(history: List[Dict[str, str]]) -> str:
    client = get_openai()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def gemini_ok() -> bool:
    if not GEMINI_API_KEY:
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        list(genai.list_models())  # lightweight check
        return True
    except Exception:
        return False


GEMINI_AVAILABLE = gemini_ok()


def call_gemini(history: List[Dict[str, str]]) -> str:
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not available")
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=history[0]["content"],
        generation_config={"temperature": 0.3},
    )
    # Render role turns for Gemini
    turns = []
    for m in history[1:]:
        turns.append(("User: " if m["role"] == "user" else "Assistant: ") + m["content"])
    prompt = "\n\n".join(turns) or "Hello"
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


# ------------------ Tip + Reliability ------------------
def refine_tip_from_answer(answer: str) -> str:
    """Return a one-sentence, age-appropriate, action-oriented sustainability tip."""
    # find "Tip: ..." if model already wrote one
    tip = ""
    for line in answer.splitlines():
        if line.strip().lower().startswith("tip:"):
            tip = line.split(":", 1)[1].strip()
            break
    if not tip:
        tip = "Choose public transit or bike for a short trip this week."

    client = get_openai()
    msg = [
        {"role": "system", "content": "Rewrite the tip as one clear, age-appropriate, action-oriented sentence."},
        {"role": "user", "content": f"Tip: {tip}\nReturn only the improved tip text."},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.2)
    return resp.choices[0].message.content.strip()


def _embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_openai()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


def _cos(a: List[float], b: List[float]) -> float:
    import math
    denom = (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))) or 1e-9
    return sum(x*y for x, y in zip(a, b)) / denom


def reliability_score(answer: str) -> Tuple[float, List[Tuple[str, str]]]:
    """Return score in [0,1] + list of matched (fact, source)."""
    answer_emb = _embed_texts([answer])[0]
    fact_embs = _embed_texts([f["text"] for f in FACTS_BANK])
    sims = [_cos(answer_emb, e) for e in fact_embs]
    pairs = sorted(zip(sims, FACTS_BANK), key=lambda x: x[0], reverse=True)[:3]
    avg_top3 = sum(s for s, _ in pairs) / max(1, len(pairs))
    score = max(0.0, min(1.0, (avg_top3 - 0.5) / 0.5))  # normalize into 0..1
    matches = [(p["text"], p["source"]) for _, p in pairs]
    return score, matches


# ------------------ App ------------------
ensure_state()

# If no chat exists yet, create one active chat (hidden until used)
if st.session_state.active_id is None:
    new_chat(MODELS[0], AGE_GROUPS[2])

active = get_active()

# -------- Sidebar: EXACTLY ONE "New Chat" button, then chat history --------
with st.sidebar:
    st.button(
        "🆕 New Chat",
        use_container_width=True,
        on_click=new_chat,
        args=(active["model"], active["age_group"]),
    )
    st.markdown("---")

    # Show only chats that have started (i.e., not empty placeholder)
    started = [
        c for c in st.session_state.chats if not (c["title"] == "New chat" and len(c["messages"]) == 1)
    ]
    if not started:
        st.caption("No chats yet. Start one!")
    else:
        for chat in started:
            label = chat["title"] or "New chat"
            if st.button(f"🗂️ {label}", key=f"open_{chat['id']}", use_container_width=True):
                set_active(chat["id"])

# Clean up any never-used chats so they don't pile up
prune_empty_chats()
active = get_active() or (st.session_state.chats[0] if st.session_state.chats else None)
if active is None:
    new_chat(MODELS[0], AGE_GROUPS[2])
    active = get_active()

# -------- Main: small top selectors + compare toggle --------
s1, s2, s3 = st.columns([0.25, 0.25, 0.5])
with s1:
    model_choice = st.selectbox(
        "Model",
        MODELS if GEMINI_AVAILABLE else [MODELS[0]],
        index=(MODELS.index(active["model"]) if active["model"] in MODELS else 0),
        key=f"model_{active['id']}",
        label_visibility="collapsed",
    )
    active["model"] = model_choice
with s2:
    age_choice = st.selectbox(
        "Age group",
        AGE_GROUPS,
        index=(AGE_GROUPS.index(active["age_group"]) if active["age_group"] in AGE_GROUPS else 2),
        key=f"age_{active['id']}",
        label_visibility="collapsed",
    )
    active["age_group"] = age_choice
with s3:
    compare_both = st.checkbox("Compare both models this turn", value=False)

st.caption("Model • Age group")
st.markdown("## 🌍 Climate Education Chatbot")

# Render messages (skip system)
for m in active["messages"]:
    if m["role"] == "system":
        continue
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Chat input
user_text = st.chat_input("Ask a climate question (or continue the conversation)…")
if user_text:
    user_turn = f"{user_text}\n\n(Age group: {active['age_group']})"
    active["messages"].append({"role": "user", "content": user_turn})
    active["updated_at"] = time.time()

    with st.chat_message("user"):
        st.markdown(user_text)

    # Generate assistant reply (+ optional comparison)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # Optional: compare both models for this single turn
                if compare_both and GEMINI_AVAILABLE:
                    # Primary = selected model
                    primary = call_openai(active["messages"]) if active["model"].startswith("OpenAI") else call_gemini(active["messages"])
                    tip_p = refine_tip_from_answer(primary)
                    score_p, _ = reliability_score(primary + "\n" + tip_p)

                    # Alternative = the other model
                    alt_model = "Gemini (1.5-flash)" if active["model"].startswith("OpenAI") else "OpenAI (GPT-4o-mini)"
                    alt_reply = call_gemini(active["messages"]) if alt_model.startswith("Gemini") else call_openai(active["messages"])
                    tip_a = refine_tip_from_answer(alt_reply)
                    score_a, _ = reliability_score(alt_reply + "\n" + tip_a)

                    st.markdown("**Model comparison**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**{active['model']}** — reliability `{score_p:.2f}`")
                        st.write(primary)
                        st.success(f"Tip: {tip_p}")
                    with c2:
                        st.markdown(f"**{alt_model}** — reliability `{score_a:.2f}`")
                        st.write(alt_reply)
                        st.success(f"Tip: {tip_a}")

                    # Choose best: higher reliability; tie → shorter text
                    if (score_p > score_a) or (abs(score_p - score_a) < 0.05 and len(primary) <= len(alt_reply)):
                        reply, tip_final, score_final = primary, tip_p, score_p
                    else:
                        reply, tip_final, score_final = alt_reply, tip_a, score_a

                    st.divider()
                    st.markdown("**Recommended answer**")
                    st.write(reply)
                    st.success(f"Tip: {tip_final}")
                    st.caption(f"Reliability score: {score_final:.2f} (0–1)")

                else:
                    # Single-model path (default)
                    reply = call_openai(active["messages"]) if active["model"].startswith("OpenAI") else call_gemini(active["messages"])
                    tip_final = refine_tip_from_answer(reply)
                    score_final, _ = reliability_score(reply + "\n" + tip_final)
                    st.write(reply)
                    st.success(f"Tip: {tip_final}")
                    st.caption(f"Reliability score: {score_final:.2f} (0–1)")

            except Exception as e:
                if active["model"].startswith("Gemini"):
                    # Failover to OpenAI
                    st.warning(f"Gemini error → falling back to OpenAI: {e}")
                    reply = call_openai(active["messages"])
                    tip_final = refine_tip_from_answer(reply)
                    score_final, _ = reliability_score(reply + "\n" + tip_final)
                    st.write(reply)
                    st.success(f"Tip: {tip_final}")
                    st.caption(f"Reliability score: {score_final:.2f}")
                else:
                    reply = f"Sorry, I hit an error: {e}"
                    st.write(reply)

    # Save reply
    active["messages"].append({"role": "assistant", "content": reply})
    active["updated_at"] = time.time()

    # Auto-title after first user message
    if active["title"] == "New chat":
        active["title"] = title_from_first_user(active)

    # Keep most-recent chats on top
    st.session_state.chats = sorted(st.session_state.chats, key=lambda c: c["updated_at"], reverse=True)
