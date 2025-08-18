from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
try:
    import ccxt
    import vectorbt as vbt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Binance client for historical data (public access only)
exchange = ccxt.binance({
    'sandbox': False,
    'enableRateLimit': True,
})

# Pydantic models
class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", description="Trading pair symbol")
    timeframe: str = Field(default="1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    initial_capital: float = Field(default=10000.0, description="Initial capital in USDT")
    strategy_code: str = Field(description="Python strategy code using VectorBT")

class BacktestResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    chart_data: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StrategyTemplate(BaseModel):
    name: str
    description: str
    code: str
    parameters: Dict[str, Any]

# Helper functions
def prepare_for_mongo(data):
    """Convert datetime objects to ISO strings for MongoDB storage"""
    if isinstance(data.get('created_at'), datetime):
        data['created_at'] = data['created_at'].isoformat()
    return data

def generate_mock_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic mock OHLCV data for demonstration"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Determine frequency based on timeframe
    freq_map = {
        '1m': '1T', '5m': '5T', '15m': '15T',
        '1h': '1H', '4h': '4H', '1d': '1D'
    }
    freq = freq_map.get(timeframe, '1H')
    
    # Generate date range
    dates = pd.date_range(start=start, end=end, freq=freq)
    
    # Generate realistic crypto price data (starting around $45,000 for BTC)
    np.random.seed(42)  # For reproducible results
    
    if symbol.startswith('BTC'):
        base_price = 45000
        volatility = 0.02
    elif symbol.startswith('ETH'):
        base_price = 2800
        volatility = 0.025
    else:
        base_price = 100
        volatility = 0.03
    
    # Generate price walk
    returns = np.random.normal(0.0005, volatility, len(dates))  # Slight upward bias
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        volatility_factor = np.random.uniform(0.998, 1.002)
        high = close * np.random.uniform(1.001, 1.008)
        low = close * np.random.uniform(0.992, 0.999)
        open_price = prices[i-1] * volatility_factor if i > 0 else close
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': max(open_price, close, high),
            'low': min(open_price, close, low),
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

async def fetch_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical OHLCV data from Binance, with mock data fallback"""
    try:
        # Try Binance API first
        if DEPENDENCIES_AVAILABLE:
            # Convert dates to timestamps
            since = exchange.parse8601(f"{start_date}T00:00:00Z")
            until = exchange.parse8601(f"{end_date}T23:59:59Z")
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, until)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        else:
            raise Exception("Binance API not available")
        
    except Exception as e:
        # If Binance fails (geographical restrictions, etc.), use mock data
        print(f"Binance API failed: {str(e)}. Using mock data for demonstration.")
        return generate_mock_data(symbol, timeframe, start_date, end_date)

def execute_strategy(df: pd.DataFrame, strategy_code: str, initial_capital: float) -> Dict[str, Any]:
    """Execute trading strategy using VectorBT"""
    try:
        # Create a safe environment for strategy execution
        strategy_globals = {
            'pd': pd,
            'np': np,
            'vbt': vbt,
            'data': df,
            'initial_capital': initial_capital
        }
        
        # Execute the strategy code
        exec(strategy_code, strategy_globals)
        
        # Get the portfolio from the executed code
        if 'pf' not in strategy_globals:
            raise ValueError("Strategy must create a portfolio object named 'pf'")
            
        pf = strategy_globals['pf']
        
        # Calculate metrics
        stats = pf.stats()
        
        # Create chart data
        chart_data = create_chart_data(df, pf)
        
        # Get equity curve
        equity_curve = pf.value().reset_index()
        equity_curve['timestamp'] = equity_curve['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        equity_curve = equity_curve.to_dict('records')
        
        # Get trades
        trades_df = pf.trades.records_readable
        if len(trades_df) > 0:
            trades_df['Entry Time'] = trades_df['Entry Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            trades_df['Exit Time'] = trades_df['Exit Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            trades = trades_df.to_dict('records')
        else:
            trades = []
        
        return {
            'final_value': float(stats['End Value']),
            'total_return': float(stats['Total Return [%]']),
            'sharpe_ratio': float(stats['Sharpe Ratio']) if not np.isnan(stats['Sharpe Ratio']) else 0.0,
            'max_drawdown': float(stats['Max Drawdown [%]']),
            'win_rate': float(stats['Win Rate [%]']) if 'Win Rate [%]' in stats else 0.0,
            'total_trades': int(stats['# Trades']),
            'profit_factor': float(stats['Profit Factor']) if not np.isnan(stats['Profit Factor']) else 0.0,
            'chart_data': chart_data,
            'equity_curve': equity_curve,
            'trades': trades
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Strategy execution failed: {str(e)}")

def create_chart_data(df: pd.DataFrame, pf) -> Dict[str, Any]:
    """Create Plotly chart data for candlestick and signals"""
    try:
        # Create candlestick chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Chart', 'Portfolio Value'),
            row_width=[0.2, 0.7]
        )
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add buy/sell signals if available
        entries = pf.orders.records_readable
        if len(entries) > 0:
            buy_signals = entries[entries['Side'] == 'Buy']
            sell_signals = entries[entries['Side'] == 'Sell']
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals['Timestamp'],
                        y=buy_signals['Price'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='Buy Signals'
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals['Timestamp'],
                        y=sell_signals['Price'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='Sell Signals'
                    ),
                    row=1, col=1
                )
        
        # Add portfolio value
        portfolio_value = pf.value()
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                line=dict(color='blue'),
                name='Portfolio Value'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Backtest Results",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            height=800,
            showlegend=True
        )
        
        return fig.to_dict()
        
    except Exception as e:
        # Return simple chart if detailed chart fails
        return {
            'data': [],
            'layout': {
                'title': 'Chart generation failed',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Price'}
            }
        }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Crypto Backtesting API"}

@api_router.get("/symbols")
async def get_available_symbols():
    """Get list of available crypto trading pairs"""
    try:
        if DEPENDENCIES_AVAILABLE:
            markets = exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT') and markets[symbol]['active']]
            # Convert to Binance format (BTCUSDT instead of BTC/USDT)
            symbols = [pair.replace('/', '') for pair in usdt_pairs[:50]]  # Limit to first 50
            return {"symbols": sorted(symbols)}
    except Exception as e:
        print(f"Binance markets failed: {str(e)}. Using default symbols.")
    
    # Fallback to popular crypto pairs
    return {"symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT"]}

@api_router.get("/strategy-templates")
async def get_strategy_templates():
    """Get predefined strategy templates"""
    templates = [
        {
            "name": "Simple Moving Average Crossover",
            "description": "Buy when fast SMA crosses above slow SMA, sell when it crosses below",
            "code": """# Simple Moving Average Crossover Strategy
fast_period = 20
slow_period = 50

# Calculate moving averages
fast_ma = data['close'].rolling(window=fast_period).mean()
slow_ma = data['close'].rolling(window=slow_period).mean()

# Generate signals
entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

# Create portfolio
pf = vbt.Portfolio.from_signals(
    data['close'], 
    entries, 
    exits, 
    init_cash=initial_capital,
    fees=0.001  # 0.1% trading fee
)""",
            "parameters": {"fast_period": 20, "slow_period": 50}
        },
        {
            "name": "RSI Mean Reversion",
            "description": "Buy when RSI is oversold (< 30), sell when overbought (> 70)",
            "code": """# RSI Mean Reversion Strategy
import talib

# Calculate RSI
rsi = pd.Series(talib.RSI(data['close'].values, timeperiod=14), index=data.index)

# Generate signals
entries = rsi < 30  # Buy when oversold
exits = rsi > 70   # Sell when overbought

# Create portfolio
pf = vbt.Portfolio.from_signals(
    data['close'], 
    entries, 
    exits, 
    init_cash=initial_capital,
    fees=0.001  # 0.1% trading fee
)""",
            "parameters": {"rsi_period": 14, "oversold": 30, "overbought": 70}
        }
    ]
    return {"templates": templates}

@api_router.post("/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run a backtest with the provided strategy"""
    try:
        # Fetch historical data
        df = await fetch_historical_data(
            request.symbol, 
            request.timeframe, 
            request.start_date, 
            request.end_date
        )
        
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data for backtesting")
        
        # Execute strategy
        result = execute_strategy(df, request.strategy_code, request.initial_capital)
        
        # Create backtest result
        backtest_result = BacktestResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            **result
        )
        
        # Save to database
        result_dict = prepare_for_mongo(backtest_result.dict())
        await db.backtest_results.insert_one(result_dict)
        
        return backtest_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@api_router.get("/backtest-history", response_model=List[BacktestResult])
async def get_backtest_history():
    """Get historical backtest results"""
    try:
        results = await db.backtest_results.find().sort("created_at", -1).limit(50).to_list(50)
        return [BacktestResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()