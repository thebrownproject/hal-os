#!/usr/bin/env python3
"""HAL-OS Memory System - SQLite schema, chunking, and embedding.

Provides hybrid semantic search using:
- sqlite-vec for vector similarity (all-MiniLM-L6-v2, 384 dimensions)
- FTS5 for BM25 keyword matching
- Embedding cache to avoid re-computing identical content
"""

# Suppress noisy ML library output - must be set before imports
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_VERBOSITY'] = 'error'

import hashlib
import sqlite3
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional

# Lazy-loaded embedding model
_embedding_model = None

# Directories to index
MEMORY_DIR = Path(__file__).parent.parent / "memory"
STORAGE_DIR = Path(__file__).parent.parent / "storage"

# Chunking parameters
TARGET_TOKENS = 400  # Target tokens per chunk
OVERLAP_TOKENS = 80  # Overlap between chunks
AVG_CHARS_PER_TOKEN = 4  # Rough estimate for English text

# Database location
DB_PATH = Path(__file__).parent.parent / "storage" / "memory.sqlite"

# Schema version for migrations
SCHEMA_VERSION = 1

# Keyword boost for exact matches (helps proper nouns)
KEYWORD_BOOST = 0.15


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection with sqlite-vec loaded.

    Args:
        db_path: Optional path override for testing

    Returns:
        sqlite3.Connection with vec0 extension loaded
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Load sqlite-vec extension
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except ImportError:
        # sqlite-vec not installed, skip vector functionality
        pass
    except Exception as e:
        # Extension load failed, continue without vectors
        print(f"Warning: sqlite-vec not available: {e}")

    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the database schema.

    Creates tables for:
    - chunks: Text fragments with metadata
    - chunks_vec: Vector embeddings via sqlite-vec
    - chunks_fts: Full-text search via FTS5
    - embedding_cache: Cache to avoid re-embedding
    - schema_info: Version tracking for migrations
    """
    cursor = conn.cursor()

    # Schema version tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Check current schema version
    cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
    row = cursor.fetchone()
    current_version = int(row['value']) if row else 0

    if current_version >= SCHEMA_VERSION:
        return  # Schema is up to date

    # Core chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            hash TEXT NOT NULL,
            mtime REAL NOT NULL
        )
    """)

    # Indexes for efficient lookup
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_mtime ON chunks(mtime)
    """)

    # FTS5 virtual table for BM25 keyword search
    # content=chunks links to chunks table, content_rowid=id syncs row IDs
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, path, content='chunks', content_rowid='id'
        )
    """)

    # Triggers to keep FTS in sync with chunks table
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, path)
            VALUES (new.id, new.text, new.path);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, path)
            VALUES ('delete', old.id, old.text, old.path);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, path)
            VALUES ('delete', old.id, old.text, old.path);
            INSERT INTO chunks_fts(rowid, text, path)
            VALUES (new.id, new.text, new.path);
        END
    """)

    # Vector table via sqlite-vec (384 dimensions for all-MiniLM-L6-v2)
    # This may fail if sqlite-vec is not available
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
    except sqlite3.OperationalError as e:
        if "no such module: vec0" in str(e):
            print("Warning: sqlite-vec not available, vector search disabled")
        else:
            raise

    # Embedding cache to avoid re-computing identical content
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            hash TEXT PRIMARY KEY,
            vector BLOB NOT NULL
        )
    """)

    # Update schema version
    cursor.execute("""
        INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)
    """, (str(SCHEMA_VERSION),))

    conn.commit()


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize database and return connection.

    Convenience function combining get_connection and init_schema.

    Args:
        db_path: Optional path override for testing

    Returns:
        sqlite3.Connection with schema initialized
    """
    conn = get_connection(db_path)
    init_schema(conn)
    return conn


def verify_schema(conn: sqlite3.Connection) -> dict:
    """Verify schema is properly initialized.

    Returns:
        dict with table status and schema version
    """
    cursor = conn.cursor()

    # Check tables exist
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' OR type='virtual table'
        ORDER BY name
    """)
    tables = [row['name'] for row in cursor.fetchall()]

    # Check schema version
    try:
        cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
        row = cursor.fetchone()
        version = int(row['value']) if row else 0
    except sqlite3.OperationalError:
        version = 0

    # Check if vector table exists
    has_vectors = 'chunks_vec' in tables

    return {
        'version': version,
        'tables': tables,
        'has_vectors': has_vectors,
        'has_fts': 'chunks_fts' in tables,
        'has_cache': 'embedding_cache' in tables,
    }


# --- Chunking Functions ---

def truncate_smart(text: str, max_len: int = 200) -> str:
    """Truncate text at word boundary."""
    if len(text) <= max_len:
        return text

    # Find last space before limit
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')

    # If no space found or too short, just truncate at limit
    if last_space < max_len // 2:
        return truncated.rstrip() + '...'

    return truncated[:last_space].rstrip() + '...'


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count.

    Uses a rough heuristic: ~4 characters per token for English.
    """
    return len(text) // AVG_CHARS_PER_TOKEN


def split_into_paragraphs(lines: List[str]) -> List[Dict[str, Any]]:
    """Split lines into paragraphs based on blank lines and headers.

    Returns list of dicts with 'text', 'start_line', 'end_line'.
    Line numbers are 1-indexed.
    """
    paragraphs = []
    current_para = []
    start_line = 1

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Check if this is a section boundary
        is_header = stripped.startswith('#')
        is_blank = stripped == ''

        if is_blank or is_header:
            # Save current paragraph if it has content
            if current_para:
                paragraphs.append({
                    'text': '\n'.join(current_para),
                    'start_line': start_line,
                    'end_line': i - 1
                })
                current_para = []

            if is_header:
                # Header becomes its own paragraph
                paragraphs.append({
                    'text': line.rstrip(),
                    'start_line': i,
                    'end_line': i
                })
                start_line = i + 1
            else:
                start_line = i + 1
        else:
            if not current_para:
                start_line = i
            current_para.append(line.rstrip())

    # Don't forget the last paragraph
    if current_para:
        paragraphs.append({
            'text': '\n'.join(current_para),
            'start_line': start_line,
            'end_line': len(lines)
        })

    return paragraphs


def merge_paragraphs_into_chunks(
    paragraphs: List[Dict[str, Any]],
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS
) -> List[Dict[str, Any]]:
    """Merge paragraphs into chunks of ~target_tokens with overlap.

    Returns list of dicts with 'text', 'start_line', 'end_line', 'hash'.
    """
    if not paragraphs:
        return []

    chunks = []
    current_chunk_paras = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para['text'])

        # If single paragraph exceeds target, it becomes its own chunk
        if para_tokens > target_tokens and not current_chunk_paras:
            chunk_text = para['text']
            chunks.append({
                'text': chunk_text,
                'start_line': para['start_line'],
                'end_line': para['end_line'],
                'hash': hashlib.sha256(chunk_text.encode()).hexdigest()
            })
            continue

        # Check if adding this paragraph would exceed target
        if current_tokens + para_tokens > target_tokens and current_chunk_paras:
            # Finalize current chunk
            chunk_text = '\n\n'.join(p['text'] for p in current_chunk_paras)
            chunks.append({
                'text': chunk_text,
                'start_line': current_chunk_paras[0]['start_line'],
                'end_line': current_chunk_paras[-1]['end_line'],
                'hash': hashlib.sha256(chunk_text.encode()).hexdigest()
            })

            # Start new chunk with overlap
            # Keep paragraphs that fit within overlap_tokens
            overlap_paras = []
            overlap_tokens_count = 0
            for p in reversed(current_chunk_paras):
                p_tokens = estimate_tokens(p['text'])
                if overlap_tokens_count + p_tokens <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_tokens_count += p_tokens
                else:
                    break

            current_chunk_paras = overlap_paras
            current_tokens = overlap_tokens_count

        current_chunk_paras.append(para)
        current_tokens += para_tokens

    # Finalize last chunk
    if current_chunk_paras:
        chunk_text = '\n\n'.join(p['text'] for p in current_chunk_paras)
        chunks.append({
            'text': chunk_text,
            'start_line': current_chunk_paras[0]['start_line'],
            'end_line': current_chunk_paras[-1]['end_line'],
            'hash': hashlib.sha256(chunk_text.encode()).hexdigest()
        })

    return chunks


def chunk_file(path: Path) -> List[Dict[str, Any]]:
    """Chunk a markdown file into ~400 token pieces with 80 token overlap.

    Args:
        path: Path to the markdown file

    Returns:
        List of dicts with 'text', 'start_line', 'end_line', 'hash'

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.strip():
        return []

    lines = content.split('\n')
    paragraphs = split_into_paragraphs(lines)
    chunks = merge_paragraphs_into_chunks(paragraphs)

    return chunks


# --- Embedding Functions ---

def get_embedding_model():
    """Get or create the embedding model (lazy loaded)."""
    global _embedding_model

    if _embedding_model is None:
        try:
            import warnings
            import logging
            import sys

            warnings.filterwarnings('ignore')

            # Suppress library logging
            for logger_name in ['sentence_transformers', 'transformers', 'torch',
                                'huggingface_hub', 'filelock']:
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)

            # Redirect stderr to suppress any remaining noise
            from contextlib import redirect_stderr
            from io import StringIO

            from sentence_transformers import SentenceTransformer

            with redirect_stderr(StringIO()):
                _embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device='cpu',
                    trust_remote_code=False
                )

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    return _embedding_model


def vector_to_bytes(vector: List[float]) -> bytes:
    """Convert a vector to bytes for SQLite storage."""
    return struct.pack(f'{len(vector)}f', *vector)


def bytes_to_vector(data: bytes) -> List[float]:
    """Convert bytes back to a vector."""
    n = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f'{n}f', data))


def embed_chunks(
    chunks: List[Dict[str, Any]],
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """Add vector embeddings to chunks.

    Uses the embedding cache to avoid recomputing identical content.

    Args:
        chunks: List of chunk dicts with 'text' and 'hash'
        conn: Optional database connection for cache lookup

    Returns:
        Chunks with 'vector' key added (list of floats)
    """
    if not chunks:
        return chunks

    # Check cache for existing embeddings
    cached_vectors = {}
    uncached_indices = []

    if conn:
        cursor = conn.cursor()
        for i, chunk in enumerate(chunks):
            cursor.execute(
                "SELECT vector FROM embedding_cache WHERE hash = ?",
                (chunk['hash'],)
            )
            row = cursor.fetchone()
            if row:
                cached_vectors[i] = bytes_to_vector(row['vector'])
            else:
                uncached_indices.append(i)
    else:
        uncached_indices = list(range(len(chunks)))

    # Embed uncached chunks
    if uncached_indices:
        model = get_embedding_model()
        texts = [chunks[i]['text'] for i in uncached_indices]
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Store in cache and result
        for idx, emb_idx in enumerate(uncached_indices):
            vector = embeddings[idx].tolist()
            cached_vectors[emb_idx] = vector

            # Cache the embedding
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO embedding_cache (hash, vector) VALUES (?, ?)",
                    (chunks[emb_idx]['hash'], vector_to_bytes(vector))
                )

        if conn:
            conn.commit()

    # Add vectors to chunks
    result = []
    for i, chunk in enumerate(chunks):
        chunk_with_vector = chunk.copy()
        chunk_with_vector['vector'] = cached_vectors[i]
        result.append(chunk_with_vector)

    return result


# --- Indexing Functions ---

def index_file(
    path: Path,
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """Index a single file: chunk, embed, store in database.

    Args:
        path: Path to the markdown file
        conn: Optional database connection (creates new if not provided)

    Returns:
        Dict with 'path', 'chunks_stored', 'cached_hits'
    """
    own_conn = conn is None
    if own_conn:
        conn = init_db()

    path = Path(path).resolve()
    rel_path = str(path)  # Store absolute path for now

    # Get file mtime
    mtime = path.stat().st_mtime

    cursor = conn.cursor()

    # Check if file needs reindexing
    cursor.execute(
        "SELECT mtime FROM chunks WHERE path = ? LIMIT 1",
        (rel_path,)
    )
    row = cursor.fetchone()
    if row and row['mtime'] == mtime:
        # File unchanged, skip
        return {'path': rel_path, 'chunks_stored': 0, 'cached_hits': 0, 'skipped': True}

    # Delete old chunks for this file
    cursor.execute("SELECT id FROM chunks WHERE path = ?", (rel_path,))
    old_ids = [row['id'] for row in cursor.fetchall()]

    if old_ids:
        cursor.execute(f"DELETE FROM chunks WHERE path = ?", (rel_path,))
        # Also delete from vector table
        try:
            for old_id in old_ids:
                cursor.execute("DELETE FROM chunks_vec WHERE id = ?", (old_id,))
        except sqlite3.OperationalError:
            pass  # sqlite-vec not available

    # Chunk the file
    chunks = chunk_file(path)

    if not chunks:
        if own_conn:
            conn.close()
        return {'path': rel_path, 'chunks_stored': 0, 'cached_hits': 0}

    # Embed chunks (uses cache)
    cached_before = cursor.execute("SELECT COUNT(*) as cnt FROM embedding_cache").fetchone()['cnt']
    chunks_with_vectors = embed_chunks(chunks, conn=conn)
    cached_after = cursor.execute("SELECT COUNT(*) as cnt FROM embedding_cache").fetchone()['cnt']
    cached_hits = len(chunks) - (cached_after - cached_before)

    # Store chunks
    for chunk in chunks_with_vectors:
        cursor.execute(
            """INSERT INTO chunks (path, start_line, end_line, text, hash, mtime)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rel_path, chunk['start_line'], chunk['end_line'],
             chunk['text'], chunk['hash'], mtime)
        )
        chunk_id = cursor.lastrowid

        # Store vector
        try:
            cursor.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, vector_to_bytes(chunk['vector']))
            )
        except sqlite3.OperationalError:
            pass  # sqlite-vec not available

    conn.commit()

    if own_conn:
        conn.close()

    return {
        'path': rel_path,
        'chunks_stored': len(chunks),
        'cached_hits': cached_hits
    }


def index_all(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """Index all memory and storage files.

    Files indexed:
    - system/memory/*.md (MEMORY.md, SOUL.md, USER.md, context.md, sessions.md)
    - system/memory/daily/*.md
    - system/storage/**/*.md (all markdown in storage, recursively)

    Also cleans up orphaned chunks from deleted files.

    Args:
        conn: Optional database connection

    Returns:
        Dict with 'files_indexed', 'total_chunks', 'files', 'orphans_removed'
    """
    own_conn = conn is None
    if own_conn:
        conn = init_db()

    files_to_index = []

    # Core memory files
    if MEMORY_DIR.exists():
        files_to_index.extend(MEMORY_DIR.glob('*.md'))

    # Daily files
    daily_dir = MEMORY_DIR / 'daily'
    if daily_dir.exists():
        files_to_index.extend(daily_dir.glob('*.md'))

    # All storage markdown files (recursive)
    if STORAGE_DIR.exists():
        files_to_index.extend(STORAGE_DIR.glob('**/*.md'))

    # Resolve all paths for comparison
    files_to_index = [p.resolve() for p in files_to_index]
    indexed_paths = set(str(p) for p in files_to_index)

    # Clean up orphaned chunks (files that no longer exist)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT path FROM chunks")
    db_paths = set(row['path'] for row in cursor.fetchall())

    orphaned_paths = db_paths - indexed_paths
    orphans_removed = 0

    for orphan_path in orphaned_paths:
        # Get chunk IDs before deleting
        cursor.execute("SELECT id FROM chunks WHERE path = ?", (orphan_path,))
        orphan_ids = [row['id'] for row in cursor.fetchall()]

        # Delete from chunks table
        cursor.execute("DELETE FROM chunks WHERE path = ?", (orphan_path,))

        # Delete from vector table
        try:
            for oid in orphan_ids:
                cursor.execute("DELETE FROM chunks_vec WHERE id = ?", (oid,))
        except sqlite3.OperationalError:
            pass  # sqlite-vec not available

        orphans_removed += len(orphan_ids)

    conn.commit()

    # Index all files
    results = []
    total_chunks = 0

    for path in files_to_index:
        result = index_file(path, conn=conn)
        results.append(result)
        total_chunks += result.get('chunks_stored', 0)

    if own_conn:
        conn.close()

    return {
        'files_indexed': len(files_to_index),
        'total_chunks': total_chunks,
        'files': results,
        'orphans_removed': orphans_removed
    }


# --- Search Functions ---

def vector_search(
    query: str,
    conn: Optional[sqlite3.Connection] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """Search using vector similarity (cosine distance).

    Args:
        query: The search query text
        conn: Optional database connection
        top_n: Maximum number of results to return

    Returns:
        List of dicts with 'id', 'path', 'start_line', 'end_line', 'text', 'score'
        Scores are normalized to 0-1 range (1 = most similar)
    """
    if not query.strip():
        return []

    own_conn = conn is None
    if own_conn:
        conn = init_db()

    # Embed the query
    model = get_embedding_model()
    query_vector = model.encode([query], convert_to_numpy=True)[0].tolist()
    query_bytes = vector_to_bytes(query_vector)

    cursor = conn.cursor()

    try:
        # sqlite-vec uses distance, we want similarity
        # vec_distance_cosine returns distance (0 = identical, 2 = opposite)
        # Convert to similarity: 1 - (distance / 2)
        cursor.execute("""
            SELECT
                v.id,
                c.path,
                c.start_line,
                c.end_line,
                c.text,
                vec_distance_cosine(v.embedding, ?) as distance
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.id
            ORDER BY distance ASC
            LIMIT ?
        """, (query_bytes, top_n))

        results = []
        for row in cursor.fetchall():
            # Convert cosine distance to similarity score (0-1)
            # Cosine distance ranges 0-2, similarity = 1 - distance/2
            distance = row['distance']
            score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

            results.append({
                'id': row['id'],
                'path': row['path'],
                'start_line': row['start_line'],
                'end_line': row['end_line'],
                'text': row['text'],
                'score': score
            })

    except sqlite3.OperationalError as e:
        if "no such function: vec_distance_cosine" in str(e):
            # sqlite-vec not available
            results = []
        else:
            raise

    if own_conn:
        conn.close()

    return results


def bm25_search(
    query: str,
    conn: Optional[sqlite3.Connection] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """Search using BM25 keyword matching via FTS5.

    Args:
        query: The search query text
        conn: Optional database connection
        top_n: Maximum number of results to return

    Returns:
        List of dicts with 'id', 'path', 'start_line', 'end_line', 'text', 'score'
        Scores are normalized to 0-1 range (1 = highest relevance)
    """
    if not query.strip():
        return []

    own_conn = conn is None
    if own_conn:
        conn = init_db()

    cursor = conn.cursor()

    try:
        # FTS5 bm25() returns negative scores (more negative = more relevant)
        # We'll normalize based on the top result
        cursor.execute("""
            SELECT
                c.id,
                c.path,
                c.start_line,
                c.end_line,
                c.text,
                bm25(chunks_fts) as bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
        """, (query, top_n))

        raw_results = cursor.fetchall()

        if not raw_results:
            if own_conn:
                conn.close()
            return []

        # Normalize scores to 0-1 range
        # BM25 scores are negative, more negative = better
        # Find the range and normalize
        scores = [row['bm25_score'] for row in raw_results]
        min_score = min(scores)  # Most relevant (most negative)
        max_score = max(scores)  # Least relevant (least negative)

        # Avoid division by zero
        score_range = max_score - min_score if max_score != min_score else 1.0

        results = []
        for row in raw_results:
            # Normalize: best score -> 1.0, worst score -> close to 0
            if score_range != 0:
                normalized = 1.0 - ((row['bm25_score'] - min_score) / score_range)
            else:
                normalized = 1.0

            results.append({
                'id': row['id'],
                'path': row['path'],
                'start_line': row['start_line'],
                'end_line': row['end_line'],
                'text': row['text'],
                'score': normalized
            })

    except sqlite3.OperationalError as e:
        if "no such table: chunks_fts" in str(e) or "syntax error" in str(e):
            results = []
        else:
            raise

    if own_conn:
        conn.close()

    return results


def hybrid_search(
    query: str,
    conn: Optional[sqlite3.Connection] = None,
    top_n: int = 10,
    min_score: float = 0.35,
    vector_weight: float = 0.7,
    path_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Hybrid search combining vector similarity and BM25.

    Formula: finalScore = (vector_weight * vectorScore) + ((1 - vector_weight) * bm25Score)
    Default: 70% vector + 30% BM25

    Args:
        query: The search query text
        conn: Optional database connection
        top_n: Maximum number of results to return
        min_score: Minimum score threshold (0-1)
        vector_weight: Weight for vector search (0-1), BM25 gets 1-vector_weight
        path_filter: Optional path substring to filter results (e.g., 'vibe', 'second-brain')

    Returns:
        List of dicts with 'path', 'start_line', 'end_line', 'score', 'snippet'
        Deduplicated and sorted by combined score
    """
    if not query.strip():
        return []

    own_conn = conn is None
    if own_conn:
        conn = init_db()

    # Get results from both search methods
    vector_results = vector_search(query, conn=conn, top_n=top_n * 2)
    bm25_results = bm25_search(query, conn=conn, top_n=top_n * 2)

    # Combine results by chunk ID, tracking both scores
    combined = {}  # id -> {result data, vector_score, bm25_score}

    for r in vector_results:
        chunk_id = r['id']
        combined[chunk_id] = {
            'id': chunk_id,
            'path': r['path'],
            'start_line': r['start_line'],
            'end_line': r['end_line'],
            'text': r['text'],
            'vector_score': r['score'],
            'bm25_score': 0.0
        }

    for r in bm25_results:
        chunk_id = r['id']
        if chunk_id in combined:
            combined[chunk_id]['bm25_score'] = r['score']
        else:
            combined[chunk_id] = {
                'id': chunk_id,
                'path': r['path'],
                'start_line': r['start_line'],
                'end_line': r['end_line'],
                'text': r['text'],
                'vector_score': 0.0,
                'bm25_score': r['score']
            }

    # Calculate final scores and format output
    bm25_weight = 1.0 - vector_weight
    results = []

    for chunk_data in combined.values():
        # Apply path filter if specified
        if path_filter and path_filter.lower() not in chunk_data['path'].lower():
            continue

        final_score = (
            vector_weight * chunk_data['vector_score'] +
            bm25_weight * chunk_data['bm25_score']
        )

        # Boost score if query terms appear exactly in text (helps proper nouns)
        query_lower = query.lower()
        text_lower = chunk_data['text'].lower()
        if query_lower in text_lower:
            final_score = min(1.0, final_score + KEYWORD_BOOST)

        if final_score >= min_score:
            # Create snippet (first 200 chars, word-boundary aware)
            snippet = truncate_smart(chunk_data['text'], 200)

            results.append({
                'path': chunk_data['path'],
                'start_line': chunk_data['start_line'],
                'end_line': chunk_data['end_line'],
                'score': round(final_score, 4),
                'snippet': snippet
            })

    # Sort by score descending and limit
    results.sort(key=lambda x: x['score'], reverse=True)
    results = results[:top_n]

    if own_conn:
        conn.close()

    return results


def format_search_results(result: Dict[str, Any], fmt: str = 'json') -> str:
    """Format search results for output."""
    import json

    if fmt == 'json':
        return json.dumps(result, indent=2)

    # Human-readable text format
    lines = []
    for r in result['results']:
        # Extract filename from full path
        filename = Path(r['path']).name
        score = r['score']
        line_range = f"L{r['start_line']}-{r['end_line']}"

        lines.append(f"[{score:.2f}] {filename} ({line_range})")

        # Single-line snippet, indented
        snippet = r['snippet'].replace('\n', ' ').strip()
        lines.append(f"  > {snippet}")
        lines.append("")  # blank line between results

    if not result['results']:
        lines.append("No results found.")

    return '\n'.join(lines)


def search(
    query: str,
    conn: Optional[sqlite3.Connection] = None,
    top_n: int = 10,
    min_score: float = 0.35,
    path_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Main search function with full output format.

    Args:
        query: The search query text
        conn: Optional database connection
        top_n: Maximum number of results
        min_score: Minimum score threshold
        path_filter: Optional path substring to filter results

    Returns:
        Dict with 'results', 'query', 'model', 'path_filter'
    """
    results = hybrid_search(query, conn=conn, top_n=top_n, min_score=min_score,
                           path_filter=path_filter)

    output = {
        'results': results,
        'query': query,
        'model': 'all-MiniLM-L6-v2'
    }
    if path_filter:
        output['path_filter'] = path_filter

    return output


def get_lines(path: Path, start_line: int, num_lines: int) -> Dict[str, Any]:
    """Retrieve specific lines from a file.

    Args:
        path: Path to the file
        start_line: Starting line number (1-indexed)
        num_lines: Number of lines to retrieve

    Returns:
        Dict with 'path', 'start_line', 'end_line', 'content', 'total_lines'

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If start_line < 1 or num_lines < 1
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if num_lines < 1:
        raise ValueError("num_lines must be >= 1")

    with open(path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)

    # Convert to 0-indexed for slicing
    start_idx = start_line - 1
    end_idx = start_idx + num_lines

    # Clamp to file bounds
    start_idx = max(0, min(start_idx, total_lines))
    end_idx = max(start_idx, min(end_idx, total_lines))

    selected_lines = all_lines[start_idx:end_idx]
    content = ''.join(selected_lines)

    return {
        'path': str(path),
        'start_line': start_idx + 1,  # Back to 1-indexed
        'end_line': end_idx,
        'content': content,
        'total_lines': total_lines
    }


def index_all_full(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """Index all memory and storage files, forcing full re-index.

    Same as index_all but clears all existing chunks first.
    No orphan cleanup needed since we're wiping everything.

    Args:
        conn: Optional database connection

    Returns:
        Dict with 'files_indexed', 'total_chunks', 'files' list
    """
    own_conn = conn is None
    if own_conn:
        conn = init_db()

    cursor = conn.cursor()

    # Clear all existing chunks and vectors
    cursor.execute("DELETE FROM chunks")
    try:
        cursor.execute("DELETE FROM chunks_vec")
    except sqlite3.OperationalError:
        pass  # sqlite-vec not available
    conn.commit()

    files_to_index = []

    # Core memory files
    if MEMORY_DIR.exists():
        files_to_index.extend(MEMORY_DIR.glob('*.md'))

    # Daily files
    daily_dir = MEMORY_DIR / 'daily'
    if daily_dir.exists():
        files_to_index.extend(daily_dir.glob('*.md'))

    # All storage markdown files (recursive)
    if STORAGE_DIR.exists():
        files_to_index.extend(STORAGE_DIR.glob('**/*.md'))

    results = []
    total_chunks = 0

    for path in files_to_index:
        result = index_file(path, conn=conn)
        results.append(result)
        total_chunks += result.get('chunks_stored', 0)

    if own_conn:
        conn.close()

    return {
        'files_indexed': len(files_to_index),
        'total_chunks': total_chunks,
        'files': results,
        'full_reindex': True
    }


def main():
    """CLI entry point with argparse."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='HAL-OS Memory System - SQLite-based semantic search',
        prog='memory.py'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init command
    init_parser = subparsers.add_parser('init', help='Initialize database schema')

    # verify command
    verify_parser = subparsers.add_parser('verify', help='Check schema status')

    # index command
    index_parser = subparsers.add_parser('index', help='Index all memory files')
    index_parser.add_argument(
        '--full', action='store_true',
        help='Force full re-index (ignore mtime checks)'
    )

    # index-file command
    index_file_parser = subparsers.add_parser('index-file', help='Index a specific file')
    index_file_parser.add_argument('path', help='Path to the markdown file')

    # chunk command
    chunk_parser = subparsers.add_parser('chunk', help='Preview chunks for a file (text output)')
    chunk_parser.add_argument('path', help='Path to the markdown file')

    # search command
    search_parser = subparsers.add_parser('search', help='Search memory with hybrid scoring')
    search_parser.add_argument('query', nargs='+', help='Search query')
    search_parser.add_argument(
        '--max-results', '-n', type=int, default=10,
        help='Maximum number of results (default: 10)'
    )
    search_parser.add_argument(
        '--min-score', type=float, default=0.35,
        help='Minimum score threshold (default: 0.35)'
    )
    search_parser.add_argument(
        '--path', '-p', type=str, default=None,
        help='Filter results by path substring (e.g., "vibe", "second-brain", "networking")'
    )
    search_parser.add_argument(
        '--format', '-f', type=str, default='json',
        choices=['json', 'text'],
        help='Output format (default: json)'
    )

    # get command
    get_parser = subparsers.add_parser('get', help='Retrieve specific lines from a file')
    get_parser.add_argument('path', help='Path to the file')
    get_parser.add_argument('start_line', type=int, help='Starting line number (1-indexed)')
    get_parser.add_argument('num_lines', type=int, help='Number of lines to retrieve')

    args = parser.parse_args()

    if args.command == 'init':
        conn = init_db()
        status = verify_schema(conn)
        print(json.dumps(status, indent=2))
        conn.close()

    elif args.command == 'verify':
        conn = get_connection()
        status = verify_schema(conn)
        print(json.dumps(status, indent=2))
        conn.close()

    elif args.command == 'index':
        conn = init_db()
        if args.full:
            result = index_all_full(conn=conn)
        else:
            result = index_all(conn=conn)
        print(json.dumps(result, indent=2))
        conn.close()

    elif args.command == 'index-file':
        path = Path(args.path)
        conn = init_db()
        result = index_file(path, conn=conn)
        print(json.dumps(result, indent=2))
        conn.close()

    elif args.command == 'chunk':
        # Text output for human preview
        path = Path(args.path)
        chunks = chunk_file(path)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i + 1} (lines {chunk['start_line']}-{chunk['end_line']}) ---")
            print(chunk['text'][:200] + ('...' if len(chunk['text']) > 200 else ''))
            print(f"Hash: {chunk['hash'][:16]}...")

    elif args.command == 'search':
        query = ' '.join(args.query)
        conn = init_db()
        result = search(query, conn=conn, top_n=args.max_results, min_score=args.min_score,
                       path_filter=args.path)
        print(format_search_results(result, args.format))
        conn.close()

    elif args.command == 'get':
        path = Path(args.path)
        try:
            result = get_lines(path, args.start_line, args.num_lines)
            print(json.dumps(result, indent=2))
        except (FileNotFoundError, ValueError) as e:
            print(json.dumps({'error': str(e)}))
            exit(1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
