"""
API Routes for Stock Analysis Platform
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, List
import logging
import pandas as pd
from app.services.stock_data_service import StockDataService
from app.models.ai_analyzer import AIStockAnalyzer

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services
stock_service = StockDataService()
ai_analyzer = AIStockAnalyzer()

@router.get("/stocks/search")
async def search_stocks(q: str = Query(..., min_length=1)):
    """Search for stocks by symbol or name"""
    try:
        results = stock_service.search_stocks(q)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/major")
async def get_major_stocks():
    """Get list of major NYSE/DOW stocks"""
    try:
        stocks = stock_service.get_major_stocks()
        return {"stocks": stocks}
    except Exception as e:
        logger.error(f"Error fetching major stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/{symbol}/info")
async def get_stock_info(symbol: str):
    """Get comprehensive stock information"""
    try:
        info = stock_service.get_stock_info(symbol.upper())
        return {"info": info}
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/{symbol}/data")
async def get_stock_data(
    symbol: str,
    period: str = Query("1y", regex="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$")
):
    """Get stock price data"""
    try:
        data = stock_service.get_stock_data(symbol.upper(), period)
        # Convert DataFrame to dict for JSON response
        data_dict = data.to_dict('index')
        # Convert Timestamp objects to strings for JSON serialization
        for date_key, row_data in data_dict.items():
            for col_key, value in row_data.items():
                if hasattr(value, 'isoformat'):  # Handle Timestamp objects
                    data_dict[date_key][col_key] = value.isoformat()
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "data": data_dict
        }
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/{symbol}/analysis")
async def analyze_stock(symbol: str):
    """Get AI-powered stock analysis"""
    try:
        # Get stock data
        data = stock_service.get_stock_data(symbol.upper(), "1y")
        
        # Get stock info
        info = stock_service.get_stock_info(symbol.upper())
        
        # Train models with current data
        training_results = ai_analyzer.train_models(data)
        
        # Make predictions
        predictions = ai_analyzer.predict(data)
        
        # Generate recommendation
        recommendation = ai_analyzer.generate_recommendation(predictions, info)
        
        return {
            "symbol": symbol.upper(),
            "analysis": {
                "predictions": predictions,
                "recommendation": recommendation,
                "training_results": training_results,
                "technical_indicators": {k: float(v) if pd.notna(v) else 0.0 for k, v in ai_analyzer.calculate_technical_indicators(data).iloc[-1].to_dict().items()}
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing stock {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/overview")
async def get_market_overview():
    """Get market overview data"""
    try:
        overview = stock_service.get_market_overview()
        return {"overview": overview}
    except Exception as e:
        logger.error(f"Error fetching market overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/report/{symbol}")
async def generate_analysis_report(symbol: str):
    """Generate comprehensive analysis report"""
    try:
        # Get all data
        data = stock_service.get_stock_data(symbol.upper(), "1y")
        info = stock_service.get_stock_info(symbol.upper())
        
        # Calculate technical indicators
        df_with_indicators = ai_analyzer.calculate_technical_indicators(data)
        
        # Get latest values
        latest = df_with_indicators.iloc[-1]
        
        # Generate predictions and recommendations
        predictions = ai_analyzer.predict(data)
        recommendation = ai_analyzer.generate_recommendation(predictions, info)
        
        # Create comprehensive report
        report = {
            "symbol": symbol.upper(),
            "company_name": info.get('name', symbol),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "current_price": info.get('current_price', latest['Close']),
            "market_cap": info.get('market_cap', 0),
            "analysis_date": predictions.get('prediction_time'),
            "recommendation": recommendation,
            "technical_analysis": {
                "rsi": float(latest['RSI']) if pd.notna(latest['RSI']) else 0.0,
                "macd": float(latest['MACD']) if pd.notna(latest['MACD']) else 0.0,
                "macd_signal": float(latest['MACD_Signal']) if pd.notna(latest['MACD_Signal']) else 0.0,
                "bollinger_upper": float(latest['BB_Upper']) if pd.notna(latest['BB_Upper']) else 0.0,
                "bollinger_lower": float(latest['BB_Lower']) if pd.notna(latest['BB_Lower']) else 0.0,
                "bollinger_middle": float(latest['BB_Middle']) if pd.notna(latest['BB_Middle']) else 0.0,
                "sma_20": float(latest['SMA_20']) if pd.notna(latest['SMA_20']) else 0.0,
                "sma_50": float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else 0.0,
                "sma_200": float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else 0.0,
                "atr": float(latest['ATR']) if pd.notna(latest['ATR']) else 0.0,
                "volume": float(latest['Volume']) if pd.notna(latest['Volume']) else 0.0
            },
            "fundamental_analysis": {
                "pe_ratio": info.get('pe_ratio', 0),
                "forward_pe": info.get('forward_pe', 0),
                "peg_ratio": info.get('peg_ratio', 0),
                "price_to_book": info.get('price_to_book', 0),
                "dividend_yield": info.get('dividend_yield', 0),
                "beta": info.get('beta', 0),
                "eps": info.get('eps', 0),
                "revenue": info.get('revenue', 0),
                "profit_margin": info.get('profit_margin', 0),
                "return_on_equity": info.get('return_on_equity', 0),
                "debt_to_equity": info.get('debt_to_equity', 0)
            },
            "ai_predictions": predictions,
            "risk_assessment": {
                "volatility": predictions.get('volatility_prediction', 0),
                "beta": info.get('beta', 1.0),
                "risk_level": recommendation.get('risk_level', 'MEDIUM')
            }
        }
        
        return {"report": report}
        
    except Exception as e:
        logger.error(f"Error generating report for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
