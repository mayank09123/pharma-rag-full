import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv; load_dotenv()
import streamlit as st
from auth_system import AuthDB, get_allowed_modes, ROLE_PERMISSIONS
from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

st.set_page_config(page_title="Pharma RAG — Secure", page_icon="🔒", layout="wide")
st.markdown("""<style>
#MainMenu,footer,header{visibility:hidden;}
.bot-bubble{background:#F0FDF9;border:1px solid #D4F1EE;border-left:4px solid #007A6E;border-radius:0 12px 12px 12px;padding:14px 18px;margin:8px 0 8px 8px;max-width:85%;font-size:0.92rem;line-height:1.65;}
.user-bubble{background:linear-gradient(135deg,#0B1F3A,#007A6E);border-radius:12px 0 12px 12px;padding:12px 18px;margin:8px 8px 8px auto;max-width:75%;font-size:0.92rem;color:white;text-align:right;}
.typing-dot{display:inline-block;width:8px;height:8px;background:#007A6E;border-radius:50%;margin:0 2px;animation:bounce 1.2s infinite;}
.typing-dot:nth-child(2){animation-delay:0.2s;}.typing-dot:nth-child(3){animation-delay:0.4s;}
@keyframes bounce{0%,60%,100%{transform:translateY(0);}30%{transform:translateY(-6px);}}
</style>""", unsafe_allow_html=True)

if "user"      not in st.session_state: st.session_state.user      = None
if "token"     not in st.session_state: st.session_state.token     = None
if "messages"  not in st.session_state: st.session_state.messages  = []
if "input_key" not in st.session_state: st.session_state.input_key = 0

db = AuthDB()

@st.cache_resource
def load_pipeline():
    store = DrugLabelVectorStore()
    return store, PharmaRAGPipeline(vector_store=store)

def show_login():
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("## 🔒 Pharma RAG — Sign In")
        st.caption("Binghamton University · Mayank Pratap Singh Chauhan (B01098725)")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign In", type="primary", use_container_width=True):
            user = db.authenticate(username, password)
            if user:
                st.session_state.user  = user
                st.session_state.token = db.create_session(user.id)
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.divider()
        st.markdown("**Demo accounts:**")
        st.markdown("| User | Password | Role |\n|---|---|---|\n| admin | admin123 | Admin |\n| doctor | doctor123 | Doctor |\n| patient | patient123 | Patient |")

def show_chatbot():
    user          = st.session_state.user
    allowed_modes = get_allowed_modes(user.role)

    with st.sidebar:
        st.markdown(f"**👤 {user.username}** ({user.role})")
        st.divider()
        mode_label = st.selectbox("Output Mode", allowed_modes,
            format_func=lambda x: {"clinical":"🏥 Clinical","marketing":"📢 Marketing",
                                    "patient":"👤 Patient","regulatory":"📋 Regulatory"}[x])
        st.divider()
        for q in ["Warfarin contraindications?","Metformin dosage?","Lisinopril side effects?"]:
            if st.button(q, use_container_width=True): st.session_state["pending"] = q
        st.divider()
        try:
            store, _ = load_pipeline()
            st.metric("Chunks", store.collection_stats()["total_chunks"])
        except: pass
        if user.role == "admin":
            with st.expander("👑 Users"):
                for u in db.get_all_users():
                    st.text(f"{'✅' if u['is_active'] else '❌'} {u['username']} ({u['role']})")
        if st.button("🚪 Logout", use_container_width=True):
            db.logout(st.session_state.token)
            st.session_state.user = None; st.session_state.messages = []; st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []; st.rerun()

    st.title(f"💊 Pharma RAG — Welcome {user.username} ({user.role})")
    if not st.session_state.messages:
        st.markdown(f'<div class="bot-bubble">👋 Welcome <b>{user.username}</b>! You have access to: <b>{", ".join(allowed_modes)}</b> mode(s).</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">{msg["content"].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

    pending = st.session_state.pop("pending", None)
    c1, c2 = st.columns([9,1])
    with c1: user_input = st.text_input("msg", value=pending or "", placeholder="Ask about any drug...", label_visibility="collapsed", key=f"chat_{st.session_state.input_key}")
    with c2: send = st.button("➤", type="primary", use_container_width=True)

    question = (pending or user_input).strip()
    if (send or pending) and question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.empty():
            st.markdown('<div class="bot-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>', unsafe_allow_html=True)
            time.sleep(0.5)
        try:
            store, pipeline = load_pipeline()
            response = pipeline.query(question, mode=OutputMode(mode_label))
            answer = response.answer
        except Exception as e:
            answer = f"❌ Error: {e}"
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.session_state.input_key += 1; st.rerun()

if st.session_state.user is None:
    show_login()
else:
    show_chatbot()
