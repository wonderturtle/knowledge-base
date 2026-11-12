"""Document Search – Flask application factory with SQLAlchemy and Ollama."""

import os
from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from .config import Config

db = SQLAlchemy()


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign keys and WAL mode on SQLite connections."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def create_app():
    """Application factory."""
    app = Flask(
        __name__,
        instance_path=Config.INSTANCE_DIR,
        instance_relative_config=False,
        static_folder="../static",
    )
    app.config.from_object(Config)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_SIZE_BYTES

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        from . import models  # noqa: F401
        db.create_all()

        # Ensure schema migrations without dropping data
        with db.engine.connect() as conn:
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(documents)")
            cols = [c[1] for c in cursor.fetchall()]
            if "parser_used" not in cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN parser_used TEXT DEFAULT ''"))
            if "extraction_warnings" not in cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN extraction_warnings TEXT DEFAULT ''"))

            # Create FTS5 virtual table and triggers for keyword search
            conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
                    text,
                    content='passages',
                    content_rowid='id'
                );
            """))
            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
                    INSERT INTO passages_fts(rowid, text) VALUES (new.id, new.text);
                END;
            """))
            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
                    INSERT INTO passages_fts(passages_fts, rowid, text) VALUES ('delete', old.id, old.text);
                END;
            """))
            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS passages_au AFTER UPDATE ON passages BEGIN
                    INSERT INTO passages_fts(passages_fts, rowid, text) VALUES ('delete', old.id, old.text);
                    INSERT INTO passages_fts(rowid, text) VALUES (new.id, new.text);
                END;
            """))
            conn.commit()

    # Register blueprints
    from .blueprints import documents_bp, search_bp, chat_bp
    app.register_blueprint(documents_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(chat_bp)

    @app.route("/")
    def root():
        """Redirect root to documents.index."""
        return redirect(url_for("documents.index"))

    @app.cli.command("warm-model")
    def warm_model_cli():
        """Download and warm the embedding model before class."""
        from .services import _get_embedding_model, get_embedding_error
        print(f"Loading embedding model '{Config.EMBEDDING_MODEL}'...")
        model = _get_embedding_model()
        if model is None:
            print(f"Error loading model: {get_embedding_error()}")
            return
        res = list(model.embed(["Model warmup test in English e Italiano"]))
        print(f"Model warmed successfully! Vector dimension: {res[0].shape[0]}")

    return app
