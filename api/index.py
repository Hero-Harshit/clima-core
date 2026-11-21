"""
Vercel Serverless Function Entrypoint for Clima-Core Environmental Intelligence API.
Exposes the FastAPI ASGI application instance for zero-cold-start execution.
"""

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from climacore.service.app import app

# Export ASGI app instance for Vercel Python runtime
__all__ = ["app"]
