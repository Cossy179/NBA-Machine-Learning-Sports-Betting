"""
🏀 NBA Prediction API for HuggingFace Space
FastAPI application that generates daily NBA predictions with parlays and confidence scores
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import json
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model_runner import NBAModelRunner
from data_provider import NBADataProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Predictions API",
    description="AI-powered NBA game predictions with parlays and Kelly criterion bet sizing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for predictions
prediction_cache = {
    'data': None,
    'generated_at': None,
    'expires_at': None
}

# Initialize model runner (lazy loading)
model_runner = None
data_provider = None

def get_model_runner():
    """Lazy load model runner"""
    global model_runner
    if model_runner is None:
        logger.info("Loading NBA prediction models...")
        model_runner = NBAModelRunner()
        logger.info("Models loaded successfully!")
    return model_runner

def get_data_provider():
    """Lazy load data provider"""
    global data_provider
    if data_provider is None:
        logger.info("Initializing data provider...")
        data_provider = NBADataProvider()
        logger.info("Data provider ready!")
    return data_provider

def is_cache_valid():
    """Check if cached predictions are still valid"""
    if prediction_cache['data'] is None or prediction_cache['expires_at'] is None:
        return False
    
    now = datetime.now(timezone.utc)
    return now < prediction_cache['expires_at']

def get_cache_expiry():
    """Calculate cache expiry (midnight UTC)"""
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight

def generate_predictions():
    """Generate fresh predictions for today's games"""
    logger.info("Generating fresh predictions...")
    
    try:
        # Get today's games
        provider = get_data_provider()
        games = provider.get_todays_games()
        
        if not games:
            logger.warning("No games found for today")
            return {
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'games': [],
                'parlays': [],
                'message': 'No games scheduled for today'
            }
        
        logger.info(f"Found {len(games)} games")
        
        # Load model runner
        runner = get_model_runner()
        
        # Generate predictions for each game
        game_predictions = []
        for game in games:
            try:
                prediction = runner.predict_game(
                    home_team=game['home_team'],
                    away_team=game['away_team'],
                    home_odds=game.get('home_odds'),
                    away_odds=game.get('away_odds'),
                    spread=game.get('spread'),
                    total=game.get('total')
                )
                
                game_predictions.append({
                    'id': f"{game['away_team']}_at_{game['home_team']}_{game['game_time']}".replace(' ', '_'),
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_time': game['game_time'],
                    'home_odds': game.get('home_odds'),
                    'away_odds': game.get('away_odds'),
                    'spread': game.get('spread'),
                    'total': game.get('total'),
                    'prediction': prediction
                })
            except Exception as e:
                logger.error(f"Error predicting game {game['home_team']} vs {game['away_team']}: {e}")
                continue
        
        # Generate parlay suggestions
        parlays = []
        try:
            parlays = runner.generate_parlays(game_predictions, max_parlays=15)
            logger.info(f"Generated {len(parlays)} parlay suggestions")
        except Exception as e:
            logger.error(f"Error generating parlays: {e}")
        
        # Build response
        response_data = {
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'games': game_predictions,
            'parlays': parlays,
            'total_games': len(game_predictions),
            'total_parlays': len(parlays)
        }
        
        # Update cache
        prediction_cache['data'] = response_data
        prediction_cache['generated_at'] = datetime.now(timezone.utc)
        prediction_cache['expires_at'] = get_cache_expiry()
        
        logger.info(f"Predictions generated successfully. Cache expires at {prediction_cache['expires_at']}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate predictions: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NBA Predictions API",
        "version": "1.0.0",
        "endpoints": [
            "/api/predictions",
            "/api/health",
            "/api/cache-status"
        ],
        "message": "FastAPI is running correctly!"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "Test endpoint working!",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_runner is not None and model_runner.model is not None
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if models are loaded
        runner = get_model_runner()
        model_status = "loaded" if runner.model is not None else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "cache_valid": is_cache_valid(),
            "last_generated": prediction_cache['generated_at'].isoformat() if prediction_cache['generated_at'] else None,
            "cache_expires": prediction_cache['expires_at'].isoformat() if prediction_cache['expires_at'] else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/cache-status")
async def cache_status():
    """Get cache status"""
    return {
        "cache_valid": is_cache_valid(),
        "has_data": prediction_cache['data'] is not None,
        "generated_at": prediction_cache['generated_at'].isoformat() if prediction_cache['generated_at'] else None,
        "expires_at": prediction_cache['expires_at'].isoformat() if prediction_cache['expires_at'] else None,
        "games_cached": len(prediction_cache['data']['games']) if prediction_cache['data'] else 0
    }

@app.get("/api/predictions")
async def get_predictions(force_refresh: bool = False):
    """
    Get NBA predictions for today's games
    
    Parameters:
    - force_refresh: Force regenerate predictions (ignore cache)
    
    Returns:
    - JSON with game predictions and parlay suggestions
    """
    try:
        # Check cache
        if not force_refresh and is_cache_valid():
            logger.info("Returning cached predictions")
            return JSONResponse(content=prediction_cache['data'])
        
        # Generate fresh predictions
        logger.info("Cache invalid or force refresh requested")
        predictions = generate_predictions()
        
        return JSONResponse(content=predictions)
        
    except Exception as e:
        logger.error(f"Error in /api/predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh")
async def refresh_predictions():
    """Manually refresh predictions (force cache invalidation)"""
    try:
        logger.info("Manual refresh requested")
        predictions = generate_predictions()
        return JSONResponse(content={
            "message": "Predictions refreshed successfully",
            "data": predictions
        })
    except Exception as e:
        logger.error(f"Error refreshing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/games")
async def get_games_only():
    """Get only game predictions (no parlays)"""
    try:
        if not is_cache_valid():
            generate_predictions()
        
        if prediction_cache['data'] is None:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        return JSONResponse(content={
            'date': prediction_cache['data']['date'],
            'generated_at': prediction_cache['data']['generated_at'],
            'games': prediction_cache['data']['games']
        })
    except Exception as e:
        logger.error(f"Error in /api/games: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/parlays")
async def get_parlays_only():
    """Get only parlay suggestions (no individual games)"""
    try:
        if not is_cache_valid():
            generate_predictions()
        
        if prediction_cache['data'] is None:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        return JSONResponse(content={
            'date': prediction_cache['data']['date'],
            'generated_at': prediction_cache['data']['generated_at'],
            'parlays': prediction_cache['data']['parlays']
        })
    except Exception as e:
        logger.error(f"Error in /api/parlays: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🏀 NBA Predictions API starting up...")
    logger.info("Loading models in background...")
    
    try:
        # Pre-load models
        get_model_runner()
        get_data_provider()
        logger.info("✅ Startup complete!")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

# Run with: uvicorn app:app --host 0.0.0.0 --port 7860
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

