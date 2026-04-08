"""SQLite database module for relational storage."""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional
from contextlib import contextmanager


class Database:
    """Manages SQLite storage for documents, chunks, claims, results, feedback, and history."""

    def __init__(self, db_path: str = "./evilearn.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT DEFAULT 'processing',
                    page_count INTEGER DEFAULT 0
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    input_type TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    evidence TEXT,
                    explanation TEXT,
                    FOREIGN KEY (claim_id) REFERENCES claims(claim_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_decision TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
                )
            """)

    # --- Document Operations ---

    def insert_document(self, document_id: str, file_name: str, page_count: int = 0) -> None:
        """Insert a new document record."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO documents (document_id, file_name, upload_time, status, page_count) VALUES (?, ?, ?, ?, ?)",
                (document_id, file_name, datetime.utcnow().isoformat(), "processing", page_count),
            )

    def update_document_status(self, document_id: str, status: str) -> None:
        """Update document processing status."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE documents SET status = ? WHERE document_id = ?",
                (status, document_id),
            )

    def get_documents(self) -> list[dict]:
        """Get all documents."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY upload_time DESC").fetchall()
            return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> Optional[dict]:
        """Get a specific document."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            return dict(row) if row else None

    # --- Chunk Operations ---

    def insert_chunks(self, chunks: list[dict]) -> None:
        """Insert multiple chunks."""
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO chunks (chunk_id, document_id, chunk_text, page_number) VALUES (?, ?, ?, ?)",
                [(c["chunk_id"], c["document_id"], c["chunk_text"], c["page_number"]) for c in chunks],
            )

    # --- Session Operations ---

    def create_session(self, input_text: str, input_type: str = "answer") -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, input_text, input_type, created_at) VALUES (?, ?, ?, ?)",
                (session_id, input_text, input_type, datetime.utcnow().isoformat()),
            )
        return session_id

    def get_sessions(self) -> list[dict]:
        """Get all sessions ordered by creation time."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a specific session."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    # --- Claim Operations ---

    def insert_claims(self, session_id: str, claims: list[dict]) -> None:
        """Insert claims for a session."""
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO claims (claim_id, session_id, claim_text) VALUES (?, ?, ?)",
                [(c["claim_id"], session_id, c["claim_text"]) for c in claims],
            )

    def get_claims_by_session(self, session_id: str) -> list[dict]:
        """Get all claims for a session."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM claims WHERE session_id = ?", (session_id,)).fetchall()
            return [dict(row) for row in rows]

    # --- Result Operations ---

    def insert_results(self, session_id: str, results: list[dict]) -> None:
        """Insert verification results."""
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO results (result_id, claim_id, session_id, status, confidence_score, evidence, explanation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(uuid.uuid4()),
                        r["claim_id"],
                        session_id,
                        r["status"],
                        r["confidence_score"],
                        json.dumps(r.get("evidence", [])),
                        r.get("explanation", ""),
                    )
                    for r in results
                ],
            )

    def get_results_by_session(self, session_id: str) -> list[dict]:
        """Get all results for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT r.*, c.claim_text 
                   FROM results r 
                   JOIN claims c ON r.claim_id = c.claim_id 
                   WHERE r.session_id = ?""",
                (session_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["evidence"] = json.loads(d.get("evidence", "[]"))
                results.append(d)
            return results

    # --- Feedback Operations ---

    def insert_feedback(self, claim_id: str, session_id: str, decision: str) -> str:
        """Store user feedback on a claim."""
        feedback_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO feedback (feedback_id, claim_id, session_id, user_decision, created_at) VALUES (?, ?, ?, ?, ?)",
                (feedback_id, claim_id, session_id, decision, datetime.utcnow().isoformat()),
            )
        return feedback_id

    def get_feedback_by_session(self, session_id: str) -> list[dict]:
        """Get all feedback for a session."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM feedback WHERE session_id = ?", (session_id,)).fetchall()
            return [dict(row) for row in rows]

    # --- History Operations ---

    def get_history(self) -> list[dict]:
        """Get complete history with sessions, claims, and results."""
        with self._get_connection() as conn:
            sessions = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
            history = []
            for session in sessions:
                session_dict = dict(session)
                session_id = session_dict["session_id"]
                
                claims = conn.execute("SELECT * FROM claims WHERE session_id = ?", (session_id,)).fetchall()
                results = conn.execute(
                    "SELECT r.*, c.claim_text FROM results r JOIN claims c ON r.claim_id = c.claim_id WHERE r.session_id = ?",
                    (session_id,),
                ).fetchall()
                feedback_rows = conn.execute("SELECT * FROM feedback WHERE session_id = ?", (session_id,)).fetchall()

                session_dict["claims"] = [dict(c) for c in claims]
                result_list = []
                for r in results:
                    rd = dict(r)
                    rd["evidence"] = json.loads(rd.get("evidence", "[]"))
                    result_list.append(rd)
                session_dict["results"] = result_list
                session_dict["feedback"] = [dict(f) for f in feedback_rows]
                history.append(session_dict)

            return history
