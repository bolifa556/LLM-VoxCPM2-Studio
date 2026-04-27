from __future__ import annotations

from app.main import create_app
from app.config import load_config

app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )
