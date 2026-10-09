from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_admin, routes_chat, routes_health, routes_metrics
from app.config import get_settings
from app.core.logging import configure_logging
from app.core.telemetry import install_request_middleware, setup_tracing


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    setup_tracing()

    application = FastAPI(title=settings.app_name, version=settings.app_version)

    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    install_request_middleware(application)

    application.include_router(routes_health.router, prefix="/api/v1", tags=["health"])
    application.include_router(routes_chat.router, prefix="/api/v1")
    application.include_router(routes_admin.router, prefix="/api/v1")
    application.include_router(routes_metrics.router)
    return application


app = create_app()
