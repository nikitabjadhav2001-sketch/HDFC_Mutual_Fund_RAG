import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    db_ok = False
    index_count = None
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
        index_count = db.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    except Exception as exc:
        logger.warning("health check degraded: %s", exc)
        db.rollback()
    return {"status": "ok" if db_ok else "degraded", "db_ok": db_ok, "index_count": index_count}
