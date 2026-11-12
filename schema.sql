-- SQLite database schema for Document Search teaching application

CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name  TEXT    NOT NULL,
    storage_name   TEXT    NOT NULL UNIQUE,
    content_hash   TEXT    NOT NULL,
    file_type      TEXT    NOT NULL,
    parser_used TEXT NOT NULL DEFAULT '',
    extraction_warnings TEXT NOT NULL DEFAULT '',
    extracted_sections TEXT,
    extracted_text TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL DEFAULT 'pending',
    page_count     INTEGER,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS passages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    page_or_section TEXT    NOT NULL,
    embedding       TEXT,
    model_id        TEXT,
    embedding_dim   INTEGER,
    UNIQUE(document_id, seq)
);

CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text,
    content='passages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
    INSERT INTO passages_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
    INSERT INTO passages_fts(passages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS passages_au AFTER UPDATE ON passages BEGIN
    INSERT INTO passages_fts(passages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO passages_fts(rowid, text) VALUES (new.id, new.text);
END;
