import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "sqlite.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db() -> None:
    import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
