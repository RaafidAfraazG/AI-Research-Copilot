"""
Generic async repository base class.

All data access goes through repository classes — never directly from route handlers.
This keeps database logic out of the API layer and makes it independently testable.

Pattern:
    class SessionRepository(BaseRepository[ResearchSession]):
        model = ResearchSession

        async def find_recent(self, limit: int = 10) -> list[ResearchSession]:
            result = await self.session.execute(
                select(self.model).order_by(self.model.created_at.desc()).limit(limit)
            )
            return list(result.scalars())
"""
from __future__ import annotations

import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Generic async CRUD repository.

    Subclasses must set the `model` class attribute:
        class MyRepo(BaseRepository[MyModel]):
            model = MyModel
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, id: uuid.UUID) -> ModelT | None:
        """Fetch a single record by primary key. Returns None if not found."""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def create(self, instance: ModelT) -> ModelT:
        """Persist a new instance and refresh it from the database."""
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def delete(self, instance: ModelT) -> None:
        """Delete an instance and flush the change."""
        await self.session.delete(instance)
        await self.session.flush()

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[ModelT]:
        """Return a paginated list of all records."""
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)  # type: ignore[arg-type]
        )
        return list(result.scalars())
