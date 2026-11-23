"""FastAPI application factory.

Run locally:
    uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routers import jobs, models, predict, system, uploads
from api.services import build_services
from config.settings import Settings, get_settings, load_pipeline_config
from src.prediction import ModelUnavailableError
from utils.log import configure_logging

logger = logging.getLogger(__name__)

_FORM_OVERHEAD_BYTES = 1024 * 1024


def _configure_tensorflow_threads(threads: int) -> None:
    """Cap TensorFlow threads so several replicas on one machine do not oversubscribe CPUs."""
    if threads > 0:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)


def create_app(settings: Settings | None = None, config: dict | None = None) -> FastAPI:
    settings = settings or get_settings()
    config = config or load_pipeline_config()
    configure_logging(settings.log_level)
    _configure_tensorflow_threads(settings.tf_threads)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings, config)
        try:
            app.state.services.models.get()  # warm up: load the active model before taking traffic
        except ModelUnavailableError as exc:
            logger.warning("Starting without a model: %s", exc)
        yield

    app = FastAPI(
        title="Brain MRI Tumor Classifier API",
        description="Educational prototype. Four-class brain MRI classification (glioma, meningioma, pituitary, no_tumor). Not for diagnosis.",
        version="1.0.0",
        lifespan=lifespan,
    )
    max_body = int(config["uploads"]["max_batch_bytes"]) + _FORM_OVERHEAD_BYTES

    @app.middleware("http")
    async def limits_and_replica_header(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return JSONResponse({"detail": f"Request body exceeds {max_body // (1024 * 1024)} MiB."}, status.HTTP_413_CONTENT_TOO_LARGE)
        response = await call_next(request)
        response.headers["X-Served-By"] = request.app.state.services.hostname
        return response

    for router in (system.router, predict.router, uploads.router, jobs.router, models.router):
        app.include_router(router)
    return app
