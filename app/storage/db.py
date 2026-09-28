from sqlalchemy import Engine, create_engine

from storage.settings import Settings


def get_engine() -> Engine:
    return create_engine(Settings().database_url, pool_pre_ping=True)
