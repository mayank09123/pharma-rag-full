import sys, os, time, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv; load_dotenv()
import streamlit as st
from database    import PharmaDatabase
from auth_system import AuthDB, get_allowed_modes
from src.retrieval.vector_store  import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

st.set_page_config(page_title="Pharma RAG — Full", page_icon="💊", layout="wide")
st.markdown("""<style>
#MainMenu,footer,header{visibility:hidden;}
.bot-bubble{background:#F0FDF9;border:1px solid #D4F1EE;border-left:4px solid #007A6E;border-radius:0 12px 12px 12px;padding:14px 18px;margin:8px 0 8px 8px;max-width:85%;font-size:0.92rem;line-height:1.65;}
.user-bubble{background:linear-gradient(135deg,#0B1F3A,#007A6E);border-radius:12px 0 12px 12px;padding:12px 18px;margin:8px 8px 8px auto;max-width:75%;font-size:0.92rem;color:white;text-align:right;}
.typing-dot{display:inline-block;width:8px;height:8px;background:#007A6E;border-radius:50%;margin:0 2px;animation:bounce 1.2s infinite;}
.typing-dot:nth-child(2){animation-delay:0.2s;}.typing-dot:nth-child(3){animation-delay:0.4s;}
@keyframes bounce{0%,60%,100%{transform:translateY(0);}30%{transform:translateY(-6px);}}
.stat-card{background:#F0FDF9;border:1px solid #D4F1EE;border-radius:8px;padding:12px;text-align:center;}
.stat-val{font-size:1.8rem;font-weight:700;color:#007A6E;}
.stat-lbl{font-size:0.7rem;color:#6B7280;text-transform:uppercase;}
</style>""", unsafe_allow_html=True)

for k,v in {"user":None,"token":None,"conv_id":None,"messages":[],"input_key":0,"page":"chat"}.items():
    if k not in st.session_state: st.session_state[k] = v

auth_db = AuthDB(); pharma_db = PharmaDatabase()

@st.cache_resource
def load_pipeline():
    store = DrugLabelVectorStore()
    return store, PharmaRAGPipeline(vector_store=store)

def detect_drug(text):
    for d in ["warfarin","metformin","lisinopril","aspirin","atorvastatin","amoxicillin","ibuprofen","sertraline","omeprazole"]:
        if d in text.lower(): return d
    return ""

def show_login():
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown("## 🔒 Pharma RAG")
        st.caption("Persistent conversation history · Binghamton University · 2025")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign In", type="primary", use_container_width=True):
            user = auth_db.authenticate(username, password)
            if user:
                st.session_state.user  = user
                st.session_state.token = auth_db.create_session(user.id)
                conv_id = pharma_db.create_conversation(user.id, user.username,
                    title=f"Session {datetime.now().strftime('%b %d %H:%M')}")
                st.session_state.conv_id  = conv_id
                st.session_state.messages = []
                st.rerun()
            else: st.error("Invalid username or password")
        st.divider()
        st.markdown("**Demo:** admin/admin123 · doctor/doctor123 · patient/patient123")

def show_chatbot():
    user = st.session_state.user
    allowed_modes = get_allowed_modes(user.role)
    with st.sidebar:
        st.markdown(f"**👤 {user.username}** ({user.role})")
        c1,c2 = st.columns(2)
        if c1.button("💬 Chat"):      st.session_state.page = "chat";      st.rerun()
        if c2.button("📊 Analytics"): st.session_state.page = "analytics"; st.rerun()
        st.divider()
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            conv_id = pharma_db.create_conversation(user.id, user.username,
                title=f"Session {datetime.now().strftime('%b %d %H:%M')}")
            st.session_state.conv_id = conv_id; st.session_state.messages = []; st.rerun()
        search = st.text_input("🔍 Search history")
        if search:
            for r in pharma_db.search_conversations(user.id, search)[:5]:
                if st.button(f"📝 {r['title'][:25]}", key=f"sr_{r['id']}", use_container_width=True):
                    st.session_state.conv_id = r["id"]
                    st.session_state.messages = [{"role":m["role"],"content":m["content"],"sources":m.get("sources",[]),"flags":m.get("flags",[])} for m in pharma_db.get_messages(r["id"])]
                    st.rerun()
        st.divider()
        st.markdown("**💬 Recent**")
        for conv in pharma_db.get_conversations(user.id, limit=10):
            ca,cb = st.columns([4,1])
            if ca.button(f"{'▶ ' if conv['id']==st.session_state.conv_id else ''}{conv['title'][:22]}", key=f"c_{conv['id']}", use_container_width=True):
                st.session_state.conv_id = conv["id"]
                st.session_state.messages = [{"role":m["role"],"content":m["content"],"sources":m.get("sources",[]),"flags":m.get("flags",[])} for m in pharma_db.get_messages(conv["id"])]
                st.rerun()
            if cb.button("🗑", key=f"d_{conv['id']}"):
                pharma_db.delete_conversation(conv["id"], user.id)
                if st.session_state.conv_id == conv["id"]: st.session_state.conv_id = None; st.session_state.messages = []
                st.rerun()
        st.divider()
        mode_label = st.selectbox("Mode", allowed_modes, format_func=lambda x: {"clinical":"🏥 Clinical","marketing":"📢 Marketing","patient":"👤 Patient","regulatory":"📋 Regulatory"}[x])
        if st.button("🚪 Logout", use_container_width=True):
            auth_db.logout(st.session_state.token)
            st.session_state.user = None; st.session_state.messages = []; st.rerun()

    if st.session_state.page == "analytics":
        st.markdown("## 📊 Analytics")
        a = pharma_db.get_analytics()
        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="stat-card"><div class="stat-val">{a["total_queries"]}</div><div class="stat-lbl">Total Queries</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-val">{a["total_conversations"]}</div><div class="stat-lbl">Conversations</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-val">{a["total_users"]}</div><div class="stat-lbl">Users</div></div>', unsafe_allow_html=True)
        st.divider()
        cl,cr = st.columns(2)
        with cl:
            st.markdown("### 💊 Top Drugs")
            for d in a.get("top_drugs",[]): st.markdown(f"**{d.get('drug_detected','?').title()}** — {d.get('cnt',0)} queries")
        with cr:
            st.markdown("### 🎯 Modes Used")
            for m in a.get("top_modes",[]): st.markdown(f"**{m.get('mode','?').title()}** — {m.get('cnt',0)} queries")
    else:
        st.title(f"💊 Pharma RAG — {user.username}")
        if not st.session_state.conv_id:
            st.info("Click ➕ New Chat to start.")
            return
        if not st.session_state.messages:
            st.markdown(f'<div class="bot-bubble">👋 Welcome <b>{user.username}</b>! Conversations are saved automatically.</div>', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-bubble">{msg["content"].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
        ci,cb = st.columns([9,1])
        with ci: user_input = st.text_input("msg", placeholder="Ask about any drug...", label_visibility="collapsed", key=f"chat_{st.session_state.input_key}")
        with cb: send = st.button("➤", type="primary", use_container_width=True)
        if send and user_input.strip():
            question = user_input.strip()
            pharma_db.save_message(st.session_state.conv_id, user.id, user.username, "user", question, mode_label, detect_drug(question))
            if not st.session_state.messages:
                pharma_db.update_conversation_title(st.session_state.conv_id, question[:60])
            st.session_state.messages.append({"role":"user","content":question})
            with st.empty():
                st.markdown('<div class="bot-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>', unsafe_allow_html=True)
                time.sleep(0.5)
            t0 = time.time()
            try:
                store, pipeline = load_pipeline()
                response = pipeline.query(question, mode=OutputMode(mode_label))
                answer = response.answer; sources = response.sources; flags = response.compliance_flags; tokens = response.tokens_used
            except Exception as e:
                answer = f"❌ {e}"; sources=[]; flags=[]; tokens=0
            pharma_db.save_message(st.session_state.conv_id, user.id, user.username, "assistant", answer, mode_label, detect_drug(question), [{"metadata":s.get("metadata",{}),"score":s.get("score",0)} for s in sources[:4]], flags, tokens)
            pharma_db.log_query(user.id, user.username, question, mode_label, detect_drug(question), tokens, time.time()-t0)
            st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources[:4],"flags":flags})
            st.session_state.input_key += 1; st.rerun()

if st.session_state.user is None: show_login()
else: show_chatbot()
