import sys
from pathlib import Path

# add project root to sys.path so 'backend' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # ASGI server that runs the FastAPI app

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",  # module path to the FastAPI app instance
        host="0.0.0.0",  # listen on all network interfaces
        port=8000,  # default port
        reload=True  # auto-reload on code changes (disable in production)
    )
