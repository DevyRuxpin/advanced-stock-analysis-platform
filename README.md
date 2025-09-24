# Advanced Stock Analysis Platform

An AI-powered stock analysis platform that provides comprehensive investment recommendations using cutting-edge technical and fundamental analysis.

## Features

### 🤖 AI-Powered Analysis
- **Machine Learning Models**: Uses Random Forest and Gradient Boosting algorithms
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Fundamental Analysis**: P/E ratios, PEG ratios, Beta, EPS, and financial metrics
- **Risk Assessment**: Volatility prediction and risk level classification

### 📊 Comprehensive Coverage
- **NYSE & DOW Stocks**: Major stocks from both exchanges
- **Real-time Data**: Live market data via Yahoo Finance API
- **Market Overview**: S&P 500, NASDAQ, Dow Jones, Russell 2000
- **Symbol Search**: Easy stock discovery and lookup

### 🎨 Modern Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Dashboard**: Real-time market overview
- **Detailed Reports**: Comprehensive analysis with buy/sell/hold recommendations
- **Export Options**: Save reports as text files or share them

### 🔧 Technical Features
- **Free & Open Source**: No API keys required for basic functionality
- **Caching System**: Efficient data caching for better performance
- **Scalable Architecture**: FastAPI backend with modular design
- **No Authentication**: Direct access without user registration

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd stock-analysis-platform
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

5. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

### Basic Analysis
1. **Search for a stock** using the search bar (e.g., AAPL, MSFT, GOOGL)
2. **Click "Analyze"** to generate AI-powered recommendations
3. **Review the analysis** including technical indicators and fundamental metrics
4. **Save or share** the report for future reference

### Available Endpoints

#### Web Interface
- `/` - Main dashboard
- `/analyze/{symbol}` - Direct analysis page for a specific stock

#### API Endpoints
- `GET /api/stocks/search?q={query}` - Search for stocks
- `GET /api/stocks/major` - Get major NYSE/DOW stocks
- `GET /api/stocks/{symbol}/info` - Get stock information
- `GET /api/stocks/{symbol}/data?period={period}` - Get stock price data
- `GET /api/stocks/{symbol}/analysis` - Get AI analysis
- `GET /api/analysis/report/{symbol}` - Generate comprehensive report
- `GET /api/market/overview` - Get market overview

## AI Model Details

### Technical Analysis Indicators
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14, 30), Stochastic, Williams %R
- **Trend**: MACD, ADX, CCI
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, MFI, VWAP
- **Patterns**: Doji, Hammer, Shooting Star candlestick patterns

### Machine Learning Models
- **Price Direction**: Random Forest Classifier (Binary: Up/Down)
- **Trend Prediction**: Gradient Boosting Regressor (Expected return)
- **Volatility**: Gradient Boosting Regressor (Risk assessment)

### Recommendation Scoring
- **Technical Analysis**: 40% weight
- **Trend Analysis**: 30% weight
- **Fundamental Analysis**: 20% weight
- **Risk Assessment**: 10% weight

## Project Structure

```
stock-analysis-platform/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── ai_analyzer.py    # AI/ML models
│   ├── services/
│   │   ├── __init__.py
│   │   └── stock_data_service.py  # Data fetching
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       ├── index.html        # Main dashboard
│       └── analysis.html     # Analysis page
├── data/
│   ├── cache/               # Data caching
│   ├── models/             # Saved ML models
│   └── reports/            # Generated reports
├── main.py                 # Application entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Dependencies

### Core Framework
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running the application

### Data & Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance API for stock data
- **ta**: Technical analysis library
- **scikit-learn**: Machine learning algorithms

### Visualization
- **plotly**: Interactive charts and graphs
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and informational purposes only. The analysis and recommendations provided should not be considered as investment advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## Support

For support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ using Python, FastAPI, and modern web technologies**
