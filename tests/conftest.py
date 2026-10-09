import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://rag:rag@localhost:5432/rag")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
