"""
PulseIQ Auth Module
====================
Real user accounts backed by a local SQLite database (pulseiq.db, created
next to main.py on first run). Passwords are hashed with pbkdf2_sha256
(pure Python, no compiler/bcrypt needed -- installs cleanly on Windows).
Sessions are stateless JWTs signed with a secret that's generated once and
persisted to secret.key next to the database, so tokens survive backend
restarts but are invalidated if you delete that file.

This is a real, working auth system for a personal/demo project -- but it
has not been hardened for production (no rate limiting on login attempts,
no email verification, no password-reset flow, no refresh-token rotation).
"""

import os
import sqlite3
import secrets
import time
from typing import Optional

import jwt
from passlib.hash import pbkdf2_sha256
from fastapi import Header, HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Writable state goes to PULSEIQ_STATE_DIR on a hosted deployment, where the
# app directory itself is read-only and replaced on every push.
STATE_DIR = os.environ.get("PULSEIQ_STATE_DIR", BASE_DIR)
os.makedirs(STATE_DIR, exist_ok=True)
DB_PATH = os.path.join(STATE_DIR, "pulseiq.db")
SECRET_KEY_PATH = os.path.join(STATE_DIR, "secret.key")

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days

VALID_ROLES = ("patient", "doctor")


def _get_or_create_secret_key() -> str:
    # A deployment should set PULSEIQ_JWT_SECRET so sessions survive restarts
    # even when the state directory is wiped. Never commit this value.
    env_secret = os.environ.get("PULSEIQ_JWT_SECRET", "").strip()
    if env_secret:
        return env_secret
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "r") as f:
            key = f.read().strip()
            if key:
                return key
    key = secrets.token_hex(32)
    with open(SECRET_KEY_PATH, "w") as f:
        f.write(key)
    return key


SECRET_KEY = _get_or_create_secret_key()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            recipient_id INTEGER NOT NULL,
            body TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (sender_id) REFERENCES users(id),
            FOREIGN KEY (recipient_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_pair
        ON messages (sender_id, recipient_id, created_at)
    """)
    # Read receipts. Kept as a nullable column so existing rows stay valid —
    # NULL simply means "not yet read".
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if "read_at" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN read_at REAL")

    # Editing and replying. All three are nullable so every existing row stays
    # valid: NULL edited_at means "never edited", NULL reply_to means "not a
    # reply". Deleting for everyone removes the row outright, so there is no
    # deleted flag here.
    if "edited_at" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN edited_at REAL")
    if "reply_to" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN reply_to INTEGER")

    # "Delete for me" is per-viewer, so it cannot live on the message row —
    # the same message stays visible to the other person.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_hidden (
            message_id INTEGER NOT NULL,
            user_id    INTEGER NOT NULL,
            hidden_at  REAL NOT NULL,
            PRIMARY KEY (message_id, user_id)
        )
    """)

    # Which recording in the dataset belongs to this account. NULL means the
    # account has no data yet — the app must say so rather than showing
    # someone else's recording.
    ucols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "subject_id" not in ucols:
        conn.execute("ALTER TABLE users ADD COLUMN subject_id TEXT")
    conn.commit()
    conn.close()


def init_feature_tables():
    """Medical vault documents and medicine reminders, both owned by a user."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            mime TEXT,
            size_bytes INTEGER NOT NULL,
            category TEXT NOT NULL DEFAULT 'reports',
            uploaded_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medicines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            dosage TEXT,
            times TEXT NOT NULL,          -- JSON array of "HH:MM"
            notes TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medicine_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,            -- YYYY-MM-DD
            slot TEXT NOT NULL,           -- HH:MM
            taken_at REAL NOT NULL,
            UNIQUE (medicine_id, day, slot),
            FOREIGN KEY (medicine_id) REFERENCES medicines(id)
        )
    """)
    conn.commit()
    conn.close()


def set_user_subject(user_id: int, subject_id: Optional[str]) -> None:
    conn = get_db()
    try:
        conn.execute("UPDATE users SET subject_id = ? WHERE id = ?", (subject_id, user_id))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Direct messaging between real accounts
# ---------------------------------------------------------------------------
def list_users_by_role(role: str, exclude_id: Optional[int] = None) -> list:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, name, email, role FROM users WHERE role = ? ORDER BY name", (role,)
        ).fetchall()
        return [dict(r) for r in rows if r["id"] != exclude_id]
    finally:
        conn.close()


# A message can be unsent for everybody only inside this window. After it
# expires the sender can still hide it from their own view, which is what
# "delete for me" does.
DELETE_FOR_EVERYONE_SECONDS = 3600


def insert_message(sender_id: int, recipient_id: int, body: str,
                   reply_to: Optional[int] = None) -> dict:
    body = body.strip()
    if not body:
        raise ValueError("message body must not be empty")
    if len(body) > 4000:
        raise ValueError("message is too long (max 4000 characters)")
    conn = get_db()
    try:
        recipient = conn.execute("SELECT id FROM users WHERE id = ?", (recipient_id,)).fetchone()
        if not recipient:
            raise ValueError("recipient does not exist")
        if reply_to is not None:
            # You may only quote a message from this same conversation,
            # otherwise a reply could leak text out of someone else's thread.
            q = conn.execute(
                """SELECT id FROM messages WHERE id = ?
                   AND ((sender_id = ? AND recipient_id = ?) OR (sender_id = ? AND recipient_id = ?))""",
                (reply_to, sender_id, recipient_id, recipient_id, sender_id),
            ).fetchone()
            if not q:
                raise ValueError("you can only reply to a message in this conversation")
        ts = time.time()
        cur = conn.execute(
            "INSERT INTO messages (sender_id, recipient_id, body, created_at, reply_to) VALUES (?, ?, ?, ?, ?)",
            (sender_id, recipient_id, body, ts, reply_to),
        )
        conn.commit()
        return {"id": cur.lastrowid, "senderId": sender_id, "recipientId": recipient_id,
                "body": body, "createdAt": ts, "replyTo": reply_to}
    finally:
        conn.close()


def edit_message(message_id: int, user_id: int, body: str) -> dict:
    """Rewrite one of your own messages. Stamps edited_at so the UI can say so."""
    body = body.strip()
    if not body:
        raise ValueError("message body must not be empty")
    if len(body) > 4000:
        raise ValueError("message is too long (max 4000 characters)")
    conn = get_db()
    try:
        row = conn.execute("SELECT sender_id FROM messages WHERE id = ?", (message_id,)).fetchone()
        if not row:
            raise LookupError("message not found")
        if row["sender_id"] != user_id:
            raise PermissionError("you can only edit your own messages")
        ts = time.time()
        conn.execute("UPDATE messages SET body = ?, edited_at = ? WHERE id = ?", (body, ts, message_id))
        conn.commit()
        return {"id": message_id, "body": body, "editedAt": ts}
    finally:
        conn.close()


def delete_message_for_me(message_id: int, user_id: int) -> None:
    """Hide a message from one person's view only. Either side may do this."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT sender_id, recipient_id FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if not row:
            raise LookupError("message not found")
        if user_id not in (row["sender_id"], row["recipient_id"]):
            raise PermissionError("that message is not in your conversation")
        conn.execute(
            "INSERT OR IGNORE INTO message_hidden (message_id, user_id, hidden_at) VALUES (?, ?, ?)",
            (message_id, user_id, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def delete_message_for_everyone(message_id: int, user_id: int) -> None:
    """Unsend: the row goes, so it disappears from both sides."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT sender_id, created_at FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if not row:
            raise LookupError("message not found")
        if row["sender_id"] != user_id:
            raise PermissionError("you can only unsend your own messages")
        if time.time() - row["created_at"] > DELETE_FOR_EVERYONE_SECONDS:
            raise TimeoutError("too late to unsend this message")
        # Replies pointing at it lose their quote rather than dangling.
        conn.execute("UPDATE messages SET reply_to = NULL WHERE reply_to = ?", (message_id,))
        conn.execute("DELETE FROM message_hidden WHERE message_id = ?", (message_id,))
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
    finally:
        conn.close()


def get_conversation(user_a: int, user_b: int, limit: int = 200) -> list:
    """Messages in both directions between two users, oldest first.

    `user_a` is the viewer: anything they deleted for themselves is left out,
    while the other side still sees it. Each reply carries a small snapshot of
    the message it answers so the UI can draw the quote without a second pass.
    """
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT m.id, m.sender_id, m.recipient_id, m.body, m.created_at,
                      m.edited_at, m.reply_to,
                      q.body AS q_body, q.sender_id AS q_sender
               FROM messages m
               LEFT JOIN messages q ON q.id = m.reply_to
               WHERE ((m.sender_id = ? AND m.recipient_id = ?)
                   OR (m.sender_id = ? AND m.recipient_id = ?))
                 AND m.id NOT IN (SELECT message_id FROM message_hidden WHERE user_id = ?)
               ORDER BY m.created_at ASC
               LIMIT ?""",
            (user_a, user_b, user_b, user_a, user_a, limit),
        ).fetchall()
        out = []
        for r in rows:
            item = {"id": r["id"], "senderId": r["sender_id"], "recipientId": r["recipient_id"],
                    "body": r["body"], "createdAt": r["created_at"],
                    "editedAt": r["edited_at"]}
            if r["reply_to"] and r["q_body"] is not None:
                item["replyTo"] = {
                    "id": r["reply_to"],
                    "senderId": r["q_sender"],
                    "body": r["q_body"][:160],
                }
            out.append(item)
        return out
    finally:
        conn.close()


def unread_count_from(user_id: int, other_id: int) -> int:
    """How many messages other_id sent to user_id that user_id hasn't opened."""
    conn = get_db()
    try:
        r = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE sender_id = ? AND recipient_id = ? AND read_at IS NULL",
            (other_id, user_id),
        ).fetchone()
        return int(r["n"])
    finally:
        conn.close()


def total_unread(user_id: int) -> int:
    conn = get_db()
    try:
        r = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE recipient_id = ? AND read_at IS NULL",
            (user_id,),
        ).fetchone()
        return int(r["n"])
    finally:
        conn.close()


def mark_conversation_read(user_id: int, other_id: int) -> int:
    """Mark everything other_id sent to user_id as read. Returns rows updated."""
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE messages SET read_at = ? WHERE sender_id = ? AND recipient_id = ? AND read_at IS NULL",
            (time.time(), other_id, user_id),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def last_message_with(user_id: int, other_id: int) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute(
            """SELECT body, created_at FROM messages
               WHERE ((sender_id = ? AND recipient_id = ?) OR (sender_id = ? AND recipient_id = ?))
                 AND id NOT IN (SELECT message_id FROM message_hidden WHERE user_id = ?)
               ORDER BY created_at DESC LIMIT 1""",
            (user_id, other_id, other_id, user_id, user_id),
        ).fetchone()
        return {"body": r["body"], "createdAt": r["created_at"]} if r else None
    finally:
        conn.close()


def create_user(name: str, email: str, password: str, role: str) -> dict:
    email = email.strip().lower()
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}")
    if len(password) < 6:
        raise ValueError("password must be at least 6 characters")

    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise ValueError("An account with this email already exists")
        password_hash = pbkdf2_sha256.hash(password)
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (name.strip(), email, password_hash, role, time.time()),
        )
        conn.commit()
        user_id = cur.lastrowid
        return {"id": user_id, "name": name.strip(), "email": email, "role": role}
    finally:
        conn.close()


def verify_user(email: str, password: str) -> Optional[dict]:
    email = email.strip().lower()
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not row:
            return None
        if not pbkdf2_sha256.verify(password, row["password_hash"]):
            return None
        return {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[dict]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, name, email, role, subject_id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_access_token(user: dict) -> str:
    payload = {
        "sub": str(user["id"]),
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_SECONDS,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token")


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """FastAPI dependency: extracts and validates the Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization[len("Bearer "):]
    payload = decode_access_token(token)
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user
