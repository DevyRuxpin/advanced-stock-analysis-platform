"""
Stock Analysis Platform - Main Application
Advanced AI-powered stock analysis with technical and fundamental analysis
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Stock Analysis Platform",
    description="AI-powered stock analysis with technical and fundamental analysis",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Import our modules
from app.api.routes import router

# Include API routes
app.include_router(router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze/{symbol}", response_class=HTMLResponse)
async def analyze_stock(request: Request, symbol: str):
    """Stock analysis page"""
    return templates.TemplateResponse("analysis.html", {
        "request": request, 
        "symbol": symbol.upper()
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
