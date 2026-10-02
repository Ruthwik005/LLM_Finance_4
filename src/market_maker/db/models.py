from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    variant: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_path: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(String)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    name: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    episode_index: Mapped[int] = mapped_column(Integer)
    pnl: Mapped[float] = mapped_column(Float)


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    episode_index: Mapped[int] = mapped_column(Integer)
    side: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class Step(Base):
    __tablename__ = "steps"
    __table_args__ = (Index("ix_steps_run_episode", "run_id", "episode"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    episode: Mapped[int] = mapped_column(Integer)
    step: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[str | None] = mapped_column(String, nullable=True)
    mid: Mapped[float] = mapped_column(Float)
    bid_1: Mapped[float] = mapped_column(Float)
    bid_2: Mapped[float] = mapped_column(Float)
    ask_1: Mapped[float] = mapped_column(Float)
    ask_2: Mapped[float] = mapped_column(Float)
    bid_vol_1: Mapped[float] = mapped_column(Float)
    bid_vol_2: Mapped[float] = mapped_column(Float)
    ask_vol_1: Mapped[float] = mapped_column(Float)
    ask_vol_2: Mapped[float] = mapped_column(Float)
    inventory: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    mtm_pnl: Mapped[float] = mapped_column(Float)
    placed_volume: Mapped[float] = mapped_column(Float)
    canceled_volume: Mapped[float] = mapped_column(Float)
