from sqlalchemy import create_engine  # creates the DB connection pool
from sqlalchemy.orm import declarative_base, sessionmaker  # ORM base class + session factory

from backend.app.config import settings  # load DB URL and pool settings


# engine = the actual database connection pool
engine = create_engine(
    settings.DATABASE_URL,  # PostgreSQL connection string from .env
    pool_pre_ping=True,  # verify connections before using them (prevents stale conn errors)
    pool_size=5,  # keep 5 idle connections ready
    max_overflow=10,  # allow up to 10 extra connections during bursts
)

# SessionLocal = factory that gives us a new DB session when called
SessionLocal = sessionmaker(
    autocommit=False,  # we manually commit changes
    autoflush=False,  # we manually flush pending changes to DB
    bind=engine  # tie the session factory to our engine
)

# Base = parent class for all SQLAlchemy ORM models (Document, Chunk, etc.)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields one DB session per request and closes it after."""
    db = SessionLocal()  # create a new session
    try:
        yield db  # give the session to the route handler
    finally:
        db.close()  # always close the session, even if an error occurred
