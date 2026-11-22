"""
Vercel Serverless Function Entrypoint for Clima-Core Environmental Intelligence API.
Exposes the FastAPI ASGI application instance for zero-cold-start execution.
"""

import os
import sys

# Ensure src/ is on sys.path for Vercel Serverless Function execution
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from climacore.service.app import app

# Export ASGI app instance for Vercel Python runtime
__all__ = ["app"]
