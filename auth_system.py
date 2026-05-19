import os, sqlite3, hashlib, secrets
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

DB_PATH = "./auth.db"

@dataclass
class User:
    id: int; username: str; email: str
    role: str; created_at: str; last_login: str

class AuthDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'patient',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_login TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def _hash_password(self, password):
        salt = secrets.token_hex(32)
        pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"{salt}:{pw_hash.hex()}"

    def _verify_password(self, password, stored_hash):
        try:
            salt, pw_hash = stored_hash.split(":")
            new_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
            return secrets.compare_digest(pw_hash, new_hash)
        except:
            return False

    def create_user(self, username, email, password, role="patient"):
        try:
            with self._connect() as conn:
                conn.execute("INSERT INTO users (username,email,password_hash,role) VALUES (?,?,?,?)",
                    (username.lower(), email.lower(), self._hash_password(password), role))
            return True
        except sqlite3.IntegrityError:
            return False

    def authenticate(self, username, password):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=? AND is_active=1",
                (username.lower(),)).fetchone()
            if row and self._verify_password(password, row["password_hash"]):
                conn.execute("UPDATE users SET last_login=? WHERE id=?",
                    (datetime.utcnow().isoformat(), row["id"]))
                return User(id=row["id"], username=row["username"], email=row["email"],
                    role=row["role"], created_at=row["created_at"], last_login=datetime.utcnow().isoformat())
        return None

    def create_session(self, user_id, hours=24):
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            conn.execute("INSERT INTO sessions (user_id,token,expires_at) VALUES (?,?,?)",
                (user_id, token, expires_at))
        return token

    def validate_session(self, token):
        with self._connect() as conn:
            row = conn.execute("""SELECT u.* FROM users u JOIN sessions s ON u.id=s.user_id
                WHERE s.token=? AND s.expires_at>? AND u.is_active=1""",
                (token, datetime.utcnow().isoformat())).fetchone()
            if row:
                return User(id=row["id"], username=row["username"], email=row["email"],
                    role=row["role"], created_at=row["created_at"], last_login=row["last_login"] or "")
        return None

    def logout(self, token):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))

    def get_all_users(self):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT id,username,email,role,is_active,created_at,last_login FROM users").fetchall()]

ROLE_PERMISSIONS = {
    "admin":      {"modes": ["clinical","marketing","patient","regulatory"], "description": "Full access"},
    "doctor":     {"modes": ["clinical","regulatory"],                       "description": "Clinical + Regulatory"},
    "researcher": {"modes": ["clinical","regulatory","patient"],             "description": "Research access"},
    "patient":    {"modes": ["patient"],                                     "description": "Patient mode only"},
}

def get_allowed_modes(role):
    return ROLE_PERMISSIONS.get(role, {}).get("modes", ["patient"])

if __name__ == "__main__":
    db = AuthDB()
    defaults = [
        ("admin",      "admin@pharmarag.com",      "admin123",   "admin"),
        ("doctor",     "doctor@pharmarag.com",     "doctor123",  "doctor"),
        ("researcher", "researcher@pharmarag.com", "research123","researcher"),
        ("patient",    "patient@pharmarag.com",    "patient123", "patient"),
    ]
    for username, email, password, role in defaults:
        ok = db.create_user(username, email, password, role)
        print(f"  {'✓' if ok else '⚠ exists'} {username} ({role})")
    print("\nDone! Run: python -m streamlit run chatbot_auth.py")
