import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .database import init_tables
from .scoring import load_models
from .routers import patients, dashboard, batch, genie

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Suraksha — Maternal Risk Intelligence")

app.include_router(patients.router)
app.include_router(dashboard.router)
app.include_router(batch.router)
app.include_router(genie.router)


@app.on_event("startup")
async def on_startup():
    load_models()
    init_tables()


static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    index_html = os.path.join(static_dir, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' first.")
