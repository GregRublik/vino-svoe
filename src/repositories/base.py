from abc import ABC, abstractmethod

from pydantic import UUID4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, select, update, delete, and_
from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound

from exceptions import ModelAlreadyExistsException, ModelNotFoundException, ModelMultipleResultsFoundException

class Operators(ABC):
    operators_map = {}

    @abstractmethod
    def build_filters(self, model, filters: dict):
        raise NotImplementedError

class SQLAlchemyOperators(Operators):
    operators_map = {
        "eq": lambda field, value: field == value,
        "ne": lambda field, value: field != value,
        "gt": lambda field, value: field > value,
        "gte": lambda field, value: field >= value,
        "lt": lambda field, value: field < value,
        "lte": lambda field, value: field <= value,
        "in": lambda field, value: field.in_(value),
        "like": lambda field, value: field.like(value),
        "ilike": lambda field, value: field.ilike(value),
        "isnull": (
            lambda field, value:
            field.is_(None)
            if value
            else field.is_not(None)
        ),
    }

    def build_filters(self, model, filters: dict):

        expressions = []

        for key, value in filters.items():

            if "__" in key:
                field_name, operator = key.split("__", 1)
            else:
                field_name = key
                operator = "eq"

            if operator not in self.operators_map:
                raise ValueError(
                    f"Unknown operator: {operator}"
                )

            field = getattr(model, field_name, None)

            if field is None:
                raise ValueError(
                    f"Unknown field: {field_name}"
                )

            expression = self.operators_map[operator](
                field,
                value
            )

            expressions.append(expression)

        return and_(*expressions)


class AbstractRepository(ABC):
    """
    Абстрактный репозиторий нужен чтобы при наследовании определяли его базовые методы работы с бд
    """
    model = None
    operators: Operators = None

    @abstractmethod
    async def add_one(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def get_all(self, *args, **kwargs):
        raise NotImplementedError


class SQLAlchemyRepository(AbstractRepository):
    """
    Репозиторий для работы с sqlalchemy
    """
    model = None
    operators = SQLAlchemyOperators()

    async def add_one(self, session: AsyncSession, data: dict):
        stmt = insert(self.model).values(**data).returning(self.model)
        try:
            res = await session.execute(stmt)
            return res.scalar_one()
        except IntegrityError:
            raise ModelAlreadyExistsException

    async def change_one(self, session: AsyncSession, object_id: int | UUID4, data: dict):
        stmt = update(self.model).where(self.model.id == object_id).values(**data).returning(self.model)
        try:
            res = await session.execute(stmt)
            return res.scalar_one()
        except NoResultFound:
            raise ModelNotFoundException

    async def delete_by_id(
        self,
        session: AsyncSession,
        object_id: int | UUID4
    ):
        stmt = (
            delete(self.model)
            .where(self.model.id == object_id)
            .returning(self.model)
        )

        res = await session.execute(stmt)

        obj = res.scalar_one_or_none()

        if obj is None:
            raise ModelNotFoundException

        return obj

    async def get_all(
        self,
        session: AsyncSession,
        filters: dict = None,
        options: list = None,
    ):
        stmt = select(self.model)
        if filters:
            stmt = stmt.where(
                self.operators.build_filters(
                    self.model,
                    filters,
                )
            )
        if options:
            stmt = stmt.options(*options)
        stmt = stmt.order_by(self.model.id)
        res = await session.execute(stmt)
        return res.scalars().all()

    async def get_by_id(
        self,
        session: AsyncSession,
        object_id: int | UUID4,
        options: list = None,
    ):
        stmt = select(self.model).where(self.model.id == object_id)
        if options:
            stmt = stmt.options(*options)
        try:
            res = await session.execute(stmt)
            return res.scalar_one()
        except NoResultFound:
            raise ModelNotFoundException
        except MultipleResultsFound:
            raise ModelMultipleResultsFoundException
