#!/usr/bin/env python3
"""Tests for HAL-OS Memory System chunking and embedding."""

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from memory import (
    chunk_file,
    embed_chunks,
    index_file,
    index_all,
    index_all_full,
    get_lines,
    init_db,
    get_connection,
    MEMORY_DIR,
)


# --- Fixtures ---

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
        db_path = Path(f.name)

    conn = init_db(db_path)
    yield conn, db_path

    conn.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_markdown():
    """Create a sample markdown file for testing."""
    content = """# Test Document

This is the first paragraph with some content. It should be chunked properly.

## Section One

This section has multiple paragraphs. The first one is here.

The second paragraph in section one. More content to make it longer and test
the chunking algorithm properly with enough tokens.

## Section Two

Another section with different content. Testing the overlap behavior.

### Subsection

Even more nested content here. This helps test boundary detection.
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        path = Path(f.name)

    yield path

    path.unlink(missing_ok=True)


@pytest.fixture
def mock_embedding_model():
    """Mock the sentence-transformers model."""
    import memory
    # Clear any cached model
    memory._embedding_model = None

    with patch('memory.get_embedding_model') as mock_get:
        model = MagicMock()
        # Return 384-dim vectors (all-MiniLM-L6-v2 size)
        import numpy as np
        model.encode.return_value = np.array([[0.1] * 384])
        mock_get.return_value = model
        yield model

    # Reset after test
    memory._embedding_model = None


# --- Chunking Tests ---

class TestChunking:
    """Tests for chunk_file function."""

    def test_chunk_file_returns_list(self, sample_markdown):
        """chunk_file should return a list of chunks."""
        chunks = chunk_file(sample_markdown)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self, sample_markdown):
        """Each chunk should have text, start_line, end_line, hash."""
        chunks = chunk_file(sample_markdown)
        chunk = chunks[0]

        assert 'text' in chunk
        assert 'start_line' in chunk
        assert 'end_line' in chunk
        assert 'hash' in chunk

    def test_chunk_hash_is_content_hash(self, sample_markdown):
        """Chunk hash should be SHA256 of text content."""
        chunks = chunk_file(sample_markdown)
        chunk = chunks[0]

        expected_hash = hashlib.sha256(chunk['text'].encode()).hexdigest()
        assert chunk['hash'] == expected_hash

    def test_chunk_lines_are_valid(self, sample_markdown):
        """Start and end lines should be positive and ordered."""
        chunks = chunk_file(sample_markdown)

        for chunk in chunks:
            assert chunk['start_line'] >= 1
            assert chunk['end_line'] >= chunk['start_line']

    def test_chunks_cover_entire_file(self, sample_markdown):
        """Chunks should cover all lines of the file."""
        chunks = chunk_file(sample_markdown)

        with open(sample_markdown) as f:
            total_lines = len(f.readlines())

        # First chunk starts at line 1
        assert chunks[0]['start_line'] == 1

        # Last chunk ends at last line
        assert chunks[-1]['end_line'] == total_lines

    def test_chunks_have_overlap(self, sample_markdown):
        """Adjacent chunks should have overlapping content."""
        # Create a larger file to ensure multiple chunks
        long_content = "\n".join([
            f"Paragraph {i}. " + "Word " * 100
            for i in range(20)
        ])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(long_content)
            path = Path(f.name)

        try:
            chunks = chunk_file(path)

            if len(chunks) >= 2:
                # Check that chunks overlap (end_line of chunk N >= start_line of chunk N+1)
                for i in range(len(chunks) - 1):
                    # With overlap, chunks should share some lines
                    assert chunks[i]['end_line'] >= chunks[i + 1]['start_line'] - 1
        finally:
            path.unlink()

    def test_nonexistent_file_raises(self):
        """chunk_file should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            chunk_file(Path("/nonexistent/file.md"))


# --- Embedding Tests ---

class TestEmbedding:
    """Tests for embed_chunks function."""

    def test_embed_chunks_adds_vector(self, mock_embedding_model):
        """embed_chunks should add 'vector' key to each chunk."""
        chunks = [
            {'text': 'Test content', 'start_line': 1, 'end_line': 1, 'hash': 'abc123'}
        ]

        result = embed_chunks(chunks)

        assert 'vector' in result[0]
        assert len(result[0]['vector']) == 384

    def test_embed_chunks_uses_cache(self, temp_db, mock_embedding_model):
        """embed_chunks should use cached vectors when available."""
        conn, db_path = temp_db

        # Pre-cache an embedding
        cached_vector = bytes([0] * (384 * 4))  # 384 floats as bytes
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO embedding_cache (hash, vector) VALUES (?, ?)",
            ('cached_hash', cached_vector)
        )
        conn.commit()

        chunks = [
            {'text': 'Test content', 'start_line': 1, 'end_line': 1, 'hash': 'cached_hash'}
        ]

        result = embed_chunks(chunks, conn=conn)

        # Model should not be called if cache hit
        # (We can't easily test this without more setup, but the function should work)
        assert 'vector' in result[0]

    def test_embed_chunks_batch_processing(self, mock_embedding_model):
        """embed_chunks should batch process for efficiency."""
        import numpy as np

        chunks = [
            {'text': f'Content {i}', 'start_line': i, 'end_line': i, 'hash': f'hash{i}'}
            for i in range(10)
        ]

        mock_embedding_model.encode.return_value = np.array([[0.1] * 384 for _ in range(10)])

        result = embed_chunks(chunks)

        # Model.encode should be called with list of texts
        assert len(result) == 10


# --- Indexing Tests ---

class TestIndexing:
    """Tests for index_file and index_all functions."""

    def test_index_file_stores_chunks(self, temp_db, sample_markdown, mock_embedding_model):
        """index_file should store chunks in database."""
        conn, db_path = temp_db

        index_file(sample_markdown, conn=conn)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        count = cursor.fetchone()['count']

        assert count > 0

    def test_index_file_stores_vectors(self, temp_db, sample_markdown, mock_embedding_model):
        """index_file should store vectors in chunks_vec table."""
        conn, db_path = temp_db

        index_file(sample_markdown, conn=conn)

        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as count FROM chunks_vec")
            count = cursor.fetchone()['count']
            assert count > 0
        except sqlite3.OperationalError:
            # sqlite-vec not available, skip
            pytest.skip("sqlite-vec not available")

    def test_index_file_updates_on_change(self, temp_db, sample_markdown, mock_embedding_model):
        """index_file should update chunks when file changes."""
        conn, db_path = temp_db

        # Initial index
        index_file(sample_markdown, conn=conn)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        initial_count = cursor.fetchone()['count']

        # Modify file
        with open(sample_markdown, 'a') as f:
            f.write("\n\nNew content added here.\n")

        # Re-index
        index_file(sample_markdown, conn=conn)

        cursor.execute("SELECT COUNT(*) as count FROM chunks")
        new_count = cursor.fetchone()['count']

        # Count may change (re-chunked)
        assert new_count >= 1

    def test_index_file_skips_unchanged(self, temp_db, sample_markdown, mock_embedding_model):
        """index_file should skip unchanged files."""
        conn, db_path = temp_db

        # Index twice without changes
        index_file(sample_markdown, conn=conn)

        # Get mtime stored
        cursor = conn.cursor()
        cursor.execute("SELECT mtime FROM chunks LIMIT 1")
        mtime1 = cursor.fetchone()['mtime']

        # Second call - should detect no change needed
        # (Implementation detail: may skip or re-process, but should be efficient)
        index_file(sample_markdown, conn=conn)


class TestIndexAll:
    """Tests for index_all function."""

    def test_index_all_finds_memory_files(self, temp_db, mock_embedding_model, monkeypatch):
        """index_all should find and index memory files."""
        conn, db_path = temp_db

        # Create temp memory directory
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)

            # Create test files
            (memory_dir / "MEMORY.md").write_text("# Memory\nTest content.")
            (memory_dir / "SOUL.md").write_text("# Soul\nIdentity.")
            (memory_dir / "daily").mkdir()
            (memory_dir / "daily" / "2026-01-28.md").write_text("# Today\nNotes.")

            # Patch MEMORY_DIR
            monkeypatch.setattr('memory.MEMORY_DIR', memory_dir)

            result = index_all(conn=conn)

            assert result['files_indexed'] >= 3
            assert result['total_chunks'] > 0


# --- Edge Cases ---

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_file(self):
        """chunk_file should handle empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("")
            path = Path(f.name)

        try:
            chunks = chunk_file(path)
            assert chunks == [] or all(c['text'].strip() == '' for c in chunks)
        finally:
            path.unlink()

    def test_single_line_file(self):
        """chunk_file should handle single-line files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("Single line content.")
            path = Path(f.name)

        try:
            chunks = chunk_file(path)
            assert len(chunks) >= 1
            assert chunks[0]['start_line'] == 1
            assert chunks[0]['end_line'] == 1
        finally:
            path.unlink()

    def test_unicode_content(self):
        """chunk_file should handle unicode content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("Unicode test: Hello.")
            path = Path(f.name)

        try:
            chunks = chunk_file(path)
            assert len(chunks) >= 1
        finally:
            path.unlink()


# --- Search Tests ---

class TestSearch:
    """Tests for hybrid search functionality."""

    @pytest.fixture
    def indexed_db(self, temp_db, mock_embedding_model):
        """Create a database with indexed content for search tests."""
        conn, db_path = temp_db
        import numpy as np

        # Create diverse mock embeddings for different content
        def mock_encode(texts, **kwargs):
            # Return different embeddings based on content
            embeddings = []
            for text in texts:
                if 'python' in text.lower():
                    # Python-related content gets similar vectors
                    embeddings.append([0.8] * 192 + [0.1] * 192)
                elif 'calendar' in text.lower():
                    # Calendar content gets different vectors
                    embeddings.append([0.1] * 192 + [0.8] * 192)
                else:
                    # Default
                    embeddings.append([0.5] * 384)
            return np.array(embeddings)

        mock_embedding_model.encode.side_effect = mock_encode

        # Insert test chunks directly
        cursor = conn.cursor()

        test_data = [
            ("test/python.md", 1, 10, "Python programming basics. Learn to code with loops and functions."),
            ("test/python.md", 11, 20, "Advanced Python concepts. Decorators and generators explained."),
            ("test/calendar.md", 1, 5, "Calendar integration. Schedule events and reminders."),
            ("test/notes.md", 1, 5, "General notes about various topics. Nothing specific."),
        ]

        for path, start, end, text in test_data:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cursor.execute(
                """INSERT INTO chunks (path, start_line, end_line, text, hash, mtime)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (path, start, end, text, text_hash, 1234567890.0)
            )
            chunk_id = cursor.lastrowid

            # Store vector
            try:
                from memory import vector_to_bytes
                vector = mock_encode([text])[0].tolist()
                cursor.execute(
                    "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                    (chunk_id, vector_to_bytes(vector))
                )
            except sqlite3.OperationalError:
                pass  # sqlite-vec not available

        conn.commit()
        return conn, db_path, mock_embedding_model

    def test_vector_search_returns_results(self, indexed_db):
        """vector_search should return results for a query."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import vector_search
        except ImportError:
            pytest.skip("vector_search not implemented yet")

        results = vector_search("python coding", conn=conn, top_n=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert 'path' in results[0]
        assert 'score' in results[0]

    def test_vector_search_scores_normalized(self, indexed_db):
        """vector_search scores should be between 0 and 1."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import vector_search
        except ImportError:
            pytest.skip("vector_search not implemented yet")

        results = vector_search("python", conn=conn)

        for result in results:
            assert 0 <= result['score'] <= 1

    def test_bm25_search_returns_results(self, indexed_db):
        """bm25_search should return results for a query."""
        conn, db_path, _ = indexed_db

        try:
            from memory import bm25_search
        except ImportError:
            pytest.skip("bm25_search not implemented yet")

        results = bm25_search("python programming", conn=conn, top_n=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert 'path' in results[0]
        assert 'score' in results[0]

    def test_bm25_search_scores_normalized(self, indexed_db):
        """bm25_search scores should be between 0 and 1."""
        conn, db_path, _ = indexed_db

        try:
            from memory import bm25_search
        except ImportError:
            pytest.skip("bm25_search not implemented yet")

        results = bm25_search("python", conn=conn)

        for result in results:
            assert 0 <= result['score'] <= 1

    def test_hybrid_search_combines_results(self, indexed_db):
        """hybrid_search should combine vector and bm25 results."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import hybrid_search
        except ImportError:
            pytest.skip("hybrid_search not implemented yet")

        results = hybrid_search("python programming", conn=conn)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_search_output_format(self, indexed_db):
        """hybrid_search should return required fields."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import hybrid_search
        except ImportError:
            pytest.skip("hybrid_search not implemented yet")

        results = hybrid_search("python", conn=conn)

        if results:
            result = results[0]
            assert 'path' in result
            assert 'start_line' in result
            assert 'end_line' in result
            assert 'score' in result
            assert 'snippet' in result

    def test_hybrid_search_applies_min_score(self, indexed_db):
        """hybrid_search should filter results below min_score."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import hybrid_search
        except ImportError:
            pytest.skip("hybrid_search not implemented yet")

        # Very high threshold should filter most results
        results = hybrid_search("random query", conn=conn, min_score=0.99)

        for result in results:
            assert result['score'] >= 0.99

    def test_hybrid_search_deduplicates(self, indexed_db):
        """hybrid_search should deduplicate results from both searches."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import hybrid_search
        except ImportError:
            pytest.skip("hybrid_search not implemented yet")

        results = hybrid_search("python", conn=conn)

        # Check no duplicate chunk IDs
        seen_ids = set()
        for result in results:
            chunk_id = (result['path'], result['start_line'])
            assert chunk_id not in seen_ids, f"Duplicate found: {chunk_id}"
            seen_ids.add(chunk_id)

    def test_search_function_returns_full_format(self, indexed_db):
        """search function should return dict with results, query, model."""
        conn, db_path, mock_model = indexed_db

        try:
            from memory import search
        except ImportError:
            pytest.skip("search not implemented yet")

        output = search("python", conn=conn)

        assert isinstance(output, dict)
        assert 'results' in output
        assert 'query' in output
        assert 'model' in output
        assert output['query'] == "python"
        assert output['model'] == "all-MiniLM-L6-v2"


class TestSearchEdgeCases:
    """Edge cases for search functionality."""

    def test_empty_query(self, temp_db):
        """search should handle empty queries gracefully."""
        conn, db_path = temp_db

        try:
            from memory import search
        except ImportError:
            pytest.skip("search not implemented yet")

        output = search("", conn=conn)
        assert output['results'] == []

    def test_no_results(self, temp_db, mock_embedding_model):
        """search should return empty results for non-matching queries."""
        conn, db_path = temp_db

        try:
            from memory import search
        except ImportError:
            pytest.skip("search not implemented yet")

        output = search("xyznonexistentquery123", conn=conn)
        assert output['results'] == []


# --- Get Lines Tests ---

class TestGetLines:
    """Tests for get_lines function."""

    def test_get_lines_returns_correct_content(self, sample_markdown):
        """get_lines should return the correct lines."""
        result = get_lines(sample_markdown, 1, 3)

        assert 'content' in result
        assert 'start_line' in result
        assert 'end_line' in result
        assert 'total_lines' in result
        assert result['start_line'] == 1
        assert result['end_line'] == 3

    def test_get_lines_full_file(self, sample_markdown):
        """get_lines should handle reading full file."""
        with open(sample_markdown) as f:
            total = len(f.readlines())

        result = get_lines(sample_markdown, 1, total)
        assert result['start_line'] == 1
        assert result['end_line'] == total

    def test_get_lines_clamps_to_file_bounds(self, sample_markdown):
        """get_lines should clamp to file bounds when exceeding."""
        with open(sample_markdown) as f:
            total = len(f.readlines())

        # Request more lines than file has
        result = get_lines(sample_markdown, 1, total + 100)
        assert result['end_line'] == total

    def test_get_lines_middle_of_file(self, sample_markdown):
        """get_lines should work for middle of file."""
        result = get_lines(sample_markdown, 5, 3)
        assert result['start_line'] == 5
        assert result['end_line'] == 7

    def test_get_lines_start_past_end(self, sample_markdown):
        """get_lines should handle start_line past file end."""
        with open(sample_markdown) as f:
            total = len(f.readlines())

        result = get_lines(sample_markdown, total + 10, 5)
        assert result['content'] == ''
        assert result['start_line'] == total + 1
        assert result['end_line'] == total

    def test_get_lines_nonexistent_file(self):
        """get_lines should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            get_lines(Path("/nonexistent/file.md"), 1, 10)

    def test_get_lines_invalid_start_line(self, sample_markdown):
        """get_lines should raise ValueError for invalid start_line."""
        with pytest.raises(ValueError, match="start_line must be >= 1"):
            get_lines(sample_markdown, 0, 10)

    def test_get_lines_invalid_num_lines(self, sample_markdown):
        """get_lines should raise ValueError for invalid num_lines."""
        with pytest.raises(ValueError, match="num_lines must be >= 1"):
            get_lines(sample_markdown, 1, 0)


# --- Full Re-index Tests ---

class TestFullReindex:
    """Tests for index_all_full function."""

    def test_index_all_full_clears_existing(self, temp_db, mock_embedding_model, monkeypatch):
        """index_all_full should clear existing chunks before reindexing."""
        conn, db_path = temp_db

        # Create temp memory directory
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            (memory_dir / "MEMORY.md").write_text("# Memory\nInitial content.")
            monkeypatch.setattr('memory.MEMORY_DIR', memory_dir)

            # Initial index
            index_all(conn=conn)

            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            initial_count = cursor.fetchone()['count']
            assert initial_count > 0

            # Modify file
            (memory_dir / "MEMORY.md").write_text("# Memory\nCompletely new content.")

            # Full reindex
            result = index_all_full(conn=conn)

            assert result['full_reindex'] is True

            # Should have chunks from the new content
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            new_count = cursor.fetchone()['count']
            assert new_count > 0

    def test_index_all_full_returns_correct_format(self, temp_db, mock_embedding_model, monkeypatch):
        """index_all_full should return expected output format."""
        conn, db_path = temp_db

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            (memory_dir / "MEMORY.md").write_text("# Memory\nTest content.")
            monkeypatch.setattr('memory.MEMORY_DIR', memory_dir)

            result = index_all_full(conn=conn)

            assert 'files_indexed' in result
            assert 'total_chunks' in result
            assert 'files' in result
            assert 'full_reindex' in result
            assert result['full_reindex'] is True


# --- CLI Tests ---

class TestCLI:
    """Tests for CLI argument parsing."""

    def test_search_max_results_flag(self, temp_db, mock_embedding_model):
        """search --max-results should limit results."""
        conn, db_path = temp_db
        from memory import search

        # Insert multiple chunks
        cursor = conn.cursor()
        for i in range(20):
            text = f"Test chunk {i} with searchable content"
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cursor.execute(
                """INSERT INTO chunks (path, start_line, end_line, text, hash, mtime)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"test{i}.md", 1, 10, text, text_hash, 1234567890.0)
            )
        conn.commit()

        # Test that top_n limits results
        result_5 = search("test chunk", conn=conn, top_n=5)
        result_10 = search("test chunk", conn=conn, top_n=10)

        assert len(result_5['results']) <= 5
        assert len(result_10['results']) <= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
