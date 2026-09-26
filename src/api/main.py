from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.config import settings
from src.utils.logger import log

app = FastAPI(
    title="GreenGrid Optimizer API",
    description="API for German renewable energy forecasting.",
    version="1.0.0"
)

# CORS: explicit origins only (CORS_ORIGINS in .env). The Streamlit frontend calls
# the API server-side, so no browser origin needs access by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

if settings.ANTHROPIC_API_KEY and not settings.API_AUTH_TOKEN:
    log.warning("ANTHROPIC_API_KEY is set but API_AUTH_TOKEN is not: /chat is disabled until a token is set.")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
