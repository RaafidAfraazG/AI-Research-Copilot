"""
SQLAlchemy declarative base and shared mixins.

All ORM models must inherit from Base (directly or via TimestampMixin).
Base.metadata is used by Alembic to discover and auto-generate migrations.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to any model.

    Both columns are timezone-aware UTC timestamps.
    updated_at is automatically refreshed on every UPDATE via the onupdate hook.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
        index=False,
    )
