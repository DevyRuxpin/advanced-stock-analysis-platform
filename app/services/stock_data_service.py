"""
Stock Data Service - Handles fetching stock data from various APIs
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataService:
    """Service for fetching and managing stock data from multiple sources"""
    
    def __init__(self):
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Comprehensive NYSE, DOW, and major ETF coverage
        self.major_stocks = {
            # Technology Giants
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'NFLX': 'Netflix Inc.',
            'INTC': 'Intel Corporation',
            'IBM': 'International Business Machines Corp.',
            'ORCL': 'Oracle Corporation',
            'CSCO': 'Cisco Systems Inc.',
            'TXN': 'Texas Instruments Inc.',
            'QCOM': 'QUALCOMM Inc.',
            'AMD': 'Advanced Micro Devices Inc.',
            'AVGO': 'Broadcom Inc.',
            'TMO': 'Thermo Fisher Scientific Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'UBER': 'Uber Technologies Inc.',
            'SQ': 'Block Inc.',
            'ZM': 'Zoom Video Communications Inc.',
            'SNOW': 'Snowflake Inc.',
            'PLTR': 'Palantir Technologies Inc.',
            
            # Financial Services
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corp.',
            'WFC': 'Wells Fargo & Co.',
            'GS': 'Goldman Sachs Group Inc.',
            'MS': 'Morgan Stanley',
            'C': 'Citigroup Inc.',
            'AXP': 'American Express Co.',
            'V': 'Visa Inc.',
            'MA': 'Mastercard Inc.',
            'COF': 'Capital One Financial Corp.',
            'USB': 'U.S. Bancorp',
            'PNC': 'PNC Financial Services Group Inc.',
            'TFC': 'Truist Financial Corp.',
            'BK': 'Bank of New York Mellon Corp.',
            'SCHW': 'Charles Schwab Corp.',
            
            # Healthcare & Pharmaceuticals
            'JNJ': 'Johnson & Johnson',
            'UNH': 'UnitedHealth Group Inc.',
            'PFE': 'Pfizer Inc.',
            'ABBV': 'AbbVie Inc.',
            'LLY': 'Eli Lilly and Company',
            'MRK': 'Merck & Co. Inc.',
            'TMO': 'Thermo Fisher Scientific Inc.',
            'DHR': 'Danaher Corp.',
            'ABT': 'Abbott Laboratories',
            'BMY': 'Bristol Myers Squibb Co.',
            'AMGN': 'Amgen Inc.',
            'GILD': 'Gilead Sciences Inc.',
            'CVS': 'CVS Health Corp.',
            'CI': 'Cigna Corp.',
            'HUM': 'Humana Inc.',
            
            # Consumer & Retail
            'WMT': 'Walmart Inc.',
            'HD': 'Home Depot Inc.',
            'PG': 'Procter & Gamble Co.',
            'KO': 'The Coca-Cola Company',
            'PEP': 'PepsiCo Inc.',
            'COST': 'Costco Wholesale Corporation',
            'NKE': 'Nike Inc.',
            'SBUX': 'Starbucks Corp.',
            'MCD': 'McDonald\'s Corp.',
            'DIS': 'Walt Disney Co.',
            'CMCSA': 'Comcast Corp.',
            'T': 'AT&T Inc.',
            'VZ': 'Verizon Communications Inc.',
            'NFLX': 'Netflix Inc.',
            'AMZN': 'Amazon.com Inc.',
            
            # Energy & Utilities
            'XOM': 'Exxon Mobil Corporation',
            'CVX': 'Chevron Corporation',
            'COP': 'ConocoPhillips',
            'EOG': 'EOG Resources Inc.',
            'SLB': 'Schlumberger Ltd.',
            'KMI': 'Kinder Morgan Inc.',
            'WMB': 'Williams Companies Inc.',
            'NEE': 'NextEra Energy Inc.',
            'DUK': 'Duke Energy Corp.',
            'SO': 'Southern Co.',
            'AEP': 'American Electric Power Co.',
            'EXC': 'Exelon Corp.',
            'XEL': 'Xcel Energy Inc.',
            
            # Industrial & Materials
            'BA': 'Boeing Co.',
            'CAT': 'Caterpillar Inc.',
            'GE': 'General Electric Co.',
            'HON': 'Honeywell International Inc.',
            'MMM': '3M Co.',
            'UPS': 'United Parcel Service Inc.',
            'FDX': 'FedEx Corp.',
            'LMT': 'Lockheed Martin Corp.',
            'RTX': 'Raytheon Technologies Corp.',
            'NOC': 'Northrop Grumman Corp.',
            'GD': 'General Dynamics Corp.',
            
            # Real Estate & REITs
            'AMT': 'American Tower Corp.',
            'PLD': 'Prologis Inc.',
            'CCI': 'Crown Castle Inc.',
            'EQIX': 'Equinix Inc.',
            'PSA': 'Public Storage',
            'SPG': 'Simon Property Group Inc.',
            'O': 'Realty Income Corp.',
            'WELL': 'Welltower Inc.',
            
            # Major ETFs
            'SPY': 'SPDR S&P 500 ETF Trust',
            'QQQ': 'Invesco QQQ Trust',
            'DIA': 'SPDR Dow Jones Industrial Average ETF',
            'IWM': 'iShares Russell 2000 ETF',
            'VTI': 'Vanguard Total Stock Market ETF',
            'VOO': 'Vanguard S&P 500 ETF',
            'VEA': 'Vanguard FTSE Developed Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF',
            'BND': 'Vanguard Total Bond Market ETF',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'GLD': 'SPDR Gold Trust',
            'SLV': 'iShares Silver Trust',
            'USO': 'United States Oil Fund',
            'UNG': 'United States Natural Gas Fund',
            
            # Additional Major Companies
            'BRK.B': 'Berkshire Hathaway Inc.',
            'GOOG': 'Alphabet Inc. Class C',
            'AMZN': 'Amazon.com Inc.',
            'FB': 'Meta Platforms Inc.',
            'TWTR': 'Twitter Inc.',
            'SNAP': 'Snap Inc.',
            'PINS': 'Pinterest Inc.',
            'ROKU': 'Roku Inc.',
            'SPOT': 'Spotify Technology S.A.',
            'SHOP': 'Shopify Inc.',
            'DOCU': 'DocuSign Inc.',
            'CRWD': 'CrowdStrike Holdings Inc.',
            'OKTA': 'Okta Inc.',
            'ZM': 'Zoom Video Communications Inc.',
            'TEAM': 'Atlassian Corp.',
            'WDAY': 'Workday Inc.',
            'NOW': 'ServiceNow Inc.',
            'MDB': 'MongoDB Inc.',
            'DDOG': 'Datadog Inc.',
            'NET': 'Cloudflare Inc.'
        }
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch stock data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch fresh data (caching disabled for now to avoid JSON issues)
            logger.info(f"Fetching fresh data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error fetching data for {symbol}: {str(e)}")
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'eps': info.get('trailingEps', 0),
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'recommendation': info.get('recommendationMean', 'Hold'),
                'target_price': info.get('targetMeanPrice', 0),
                'analyst_count': info.get('numberOfAnalystOpinions', 0)
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'name': symbol, 'error': str(e)}
    
    def get_major_stocks(self) -> Dict[str, str]:
        """Get list of major NYSE/DOW stocks"""
        return self.major_stocks
    
    def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """Search for stocks by name or symbol"""
        query = query.upper()
        results = []
        
        for symbol, name in self.major_stocks.items():
            if query in symbol or query in name.upper():
                results.append({'symbol': symbol, 'name': name})
        
        return results[:10]  # Limit to 10 results
    
    def get_market_overview(self) -> Dict:
        """Get market overview data"""
        try:
            # Get major indices and ETFs
            indices = {
                'SPY': 'S&P 500',
                'QQQ': 'NASDAQ QQQ',
                'DIA': 'Dow Jones',
                'IWM': 'Russell 2000',
                'VTI': 'Total Stock Market',
                'VOO': 'S&P 500 Vanguard',
                'XLF': 'Financial Sector',
                'XLK': 'Technology Sector',
                'XLE': 'Energy Sector',
                'XLV': 'Healthcare Sector',
                'XLI': 'Industrial Sector',
                'XLY': 'Consumer Discretionary',
                'XLP': 'Consumer Staples',
                'XLU': 'Utilities',
                'XLB': 'Materials',
                'XLRE': 'Real Estate',
                'XLC': 'Communication Services'
            }
            
            overview = {}
            for symbol, name in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    data = ticker.history(period="5d")
                    
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        
                        overview[name] = {
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_percent': round(change_percent, 2),
                            'volume': info.get('volume', 0)
                        }
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {str(e)}")
                    continue
            
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            return {}
