import os, json, uuid, sqlite3
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

DB_PATH = "./pharmarag.db"

class PharmaDatabase:
    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    title TEXT NOT NULL,
                    drug_focus TEXT,
                    message_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    mode TEXT DEFAULT 'clinical',
                    drug_query TEXT,
                    sources TEXT DEFAULT '[]',
                    flags TEXT DEFAULT '[]',
                    tokens_used INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS query_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    username TEXT,
                    query TEXT NOT NULL,
                    mode TEXT,
                    drug_detected TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    response_time REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
        print("✓ Database initialized (SQLite)")

    def create_conversation(self, user_id, username, title="New Conversation", drug_focus=None):
        conv_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute("INSERT INTO conversations (id,user_id,username,title,drug_focus) VALUES (?,?,?,?,?)",
                (conv_id, user_id, username, title, drug_focus))
        return conv_id

    def get_conversations(self, user_id, limit=20):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_all_conversations(self, limit=50):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()]

    def update_conversation_title(self, conv_id, title):
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET title=? WHERE id=?", (title[:80], conv_id))

    def delete_conversation(self, conv_id, user_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id))

    def save_message(self, conv_id, user_id, username, role, content,
                     mode="clinical", drug_query="", sources=None, flags=None, tokens=0):
        msg_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute("""INSERT INTO messages
                (id,conversation_id,user_id,username,role,content,mode,drug_query,sources,flags,tokens_used)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (msg_id, conv_id, user_id, username, role, content, mode,
                 drug_query, json.dumps(sources or []), json.dumps(flags or []), tokens))
            conn.execute("UPDATE conversations SET message_count=message_count+1, updated_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), conv_id))
        return msg_id

    def get_messages(self, conv_id):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
                (conv_id,)).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["sources"] = json.loads(d.get("sources","[]"))
                d["flags"]   = json.loads(d.get("flags","[]"))
                result.append(d)
            return result

    def search_conversations(self, user_id, query):
        with self._connect() as conn:
            rows = conn.execute("""SELECT DISTINCT c.* FROM conversations c
                JOIN messages m ON c.id=m.conversation_id
                WHERE c.user_id=? AND LOWER(m.content) LIKE LOWER(?)
                ORDER BY c.updated_at DESC LIMIT 10""",
                (user_id, f"%{query}%")).fetchall()
            return [dict(r) for r in rows]

    def log_query(self, user_id, username, query, mode, drug_detected="", tokens=0, response_time=0):
        with self._connect() as conn:
            conn.execute("""INSERT INTO query_analytics
                (user_id,username,query,mode,drug_detected,tokens_used,response_time)
                VALUES (?,?,?,?,?,?,?)""",
                (user_id, username, query, mode, drug_detected, tokens, response_time))

    def get_analytics(self):
        with self._connect() as conn:
            total_q = dict(conn.execute("SELECT COUNT(*) as cnt FROM query_analytics").fetchone())
            total_c = dict(conn.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone())
            total_u = dict(conn.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM conversations").fetchone())
            top_drugs = [dict(r) for r in conn.execute("""SELECT drug_detected, COUNT(*) as cnt
                FROM query_analytics WHERE drug_detected!=''
                GROUP BY drug_detected ORDER BY cnt DESC LIMIT 5""").fetchall()]
            top_modes = [dict(r) for r in conn.execute("""SELECT mode, COUNT(*) as cnt
                FROM query_analytics GROUP BY mode ORDER BY cnt DESC""").fetchall()]
        return {"total_queries": total_q["cnt"], "total_conversations": total_c["cnt"],
                "total_users": total_u["cnt"], "top_drugs": top_drugs, "top_modes": top_modes}

if __name__ == "__main__":
    print("Setting up database...")
    db = PharmaDatabase()
    print(f"✓ Database ready: {DB_PATH}")
    print("Run: python -m streamlit run chatbot_db.py")
