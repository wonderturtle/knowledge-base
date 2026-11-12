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


def create_app(test_config=None):
    """Application factory."""
    app = Flask(
        __name__,
        instance_path=Config.INSTANCE_DIR,
        instance_relative_config=False,
        static_folder="../static",
    )
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_SIZE_BYTES"] + 65536

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_CACHE_DIR"], exist_ok=True)

    db.init_app(app)

    with app.app_context():
        from . import models  # noqa: F401
        db.create_all()

        # Ensure schema migrations without dropping data
        with db.engine.connect() as conn:
            cols = [c[1] for c in conn.execute(text("PRAGMA table_info(documents)"))]
            if "extracted_sections" not in cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN extracted_sections TEXT"))
            if "parser_used" not in cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN parser_used TEXT DEFAULT ''"))
            if "extraction_warnings" not in cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN extraction_warnings TEXT DEFAULT ''"))

            fts_exists = conn.execute(text("SELECT 1 FROM sqlite_master WHERE name='passages_fts'")).scalar()
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
            if not fts_exists:
                conn.execute(text("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')"))
            conn.commit()

    @app.errorhandler(413)
    def upload_too_large(error):
        from flask import flash
        flash("File too large. Maximum upload size is 10 MiB.", "error")
        return redirect(url_for("documents.index"))

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
            import click
            raise click.ClickException(f"Error loading model: {get_embedding_error()}")
        res = list(model.embed(["Model warmup test in English e Italiano"]))
        print(f"Model warmed successfully! Vector dimension: {res[0].shape[0]}")

    return app
