from __future__ import annotations
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from market_maker.db.models import Base


def get_engine(db_path: str) -> Engine:
    return create_engine(db_path, connect_args={"check_same_thread": False})


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session(engine: Engine) -> Session:
    return sessionmaker(bind=engine)()
