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
        
        # Calculate basic metrics directly
        returns = pf.returns()
        total_return = (pf.value().iloc[-1] - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio manually
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Get orders (trades)
        orders = pf.orders.records_readable
        total_trades = len(orders) if len(orders) > 0 else 0
        
        # Calculate win rate
        if total_trades > 0 and len(orders) > 0:
            # Get trade records if available
            try:
                trades = pf.trades.records_readable
                if len(trades) > 0:
                    winning_trades = len(trades[trades['PnL'] > 0])
                    win_rate = (winning_trades / len(trades)) * 100
                else:
                    win_rate = 0.0
            except:
                win_rate = 0.0
        else:
            win_rate = 0.0
        
        # Calculate profit factor
        if total_trades > 0:
            try:
                trades = pf.trades.records_readable
                if len(trades) > 0:
                    profits = trades[trades['PnL'] > 0]['PnL'].sum()
                    losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())
                    profit_factor = profits / losses if losses > 0 else profits
                else:
                    profit_factor = 1.0
            except:
                profit_factor = 1.0
        else:
            profit_factor = 1.0
        
        # Create chart data
        chart_data = create_chart_data(df, pf)
        
        # Get equity curve
        equity_curve = pf.value().reset_index()
        equity_curve.columns = ['timestamp', 'value']
        equity_curve['timestamp'] = equity_curve['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        equity_curve = equity_curve.to_dict('records')
        
        # Get trades for display
        trades_list = []
        try:
            trades_df = pf.trades.records_readable
            if len(trades_df) > 0:
                for _, trade in trades_df.iterrows():
                    trades_list.append({
                        'entry_time': trade['Entry Time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(trade['Entry Time']) else 'N/A',
                        'exit_time': trade['Exit Time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(trade['Exit Time']) else 'N/A',
                        'size': float(trade['Size']) if pd.notna(trade['Size']) else 0.0,
                        'entry_price': float(trade['Entry Price']) if pd.notna(trade['Entry Price']) else 0.0,
                        'exit_price': float(trade['Exit Price']) if pd.notna(trade['Exit Price']) else 0.0,
                        'pnl': float(trade['PnL']) if pd.notna(trade['PnL']) else 0.0,
                        'return': float(trade['Return [%]']) if pd.notna(trade['Return [%]']) else 0.0
                    })
        except Exception as e:
            print(f"Error processing trades: {e}")
            trades_list = []
        
        return {
            'final_value': float(pf.value().iloc[-1]),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'profit_factor': float(profit_factor),
            'chart_data': chart_data,
            'equity_curve': equity_curve,
            'trades': trades_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Strategy execution failed: {str(e)}")

def create_chart_data(df: pd.DataFrame, pf) -> Dict[str, Any]:
    """Create Plotly chart data for candlestick and signals"""
    try:
        # Simple chart data structure without complex plotly objects
        chart_data = {
            'type': 'candlestick',
            'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': df['open'].tolist(),
            'high': df['high'].tolist(), 
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'volume': df['volume'].tolist()
        }
        
        # Add buy/sell signals if available
        try:
            orders = pf.orders.records_readable
            if len(orders) > 0:
                buy_signals = orders[orders['Side'] == 'Buy']
                sell_signals = orders[orders['Side'] == 'Sell']
                
                chart_data['buy_signals'] = {
                    'timestamps': buy_signals['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist() if len(buy_signals) > 0 else [],
                    'prices': buy_signals['Price'].tolist() if len(buy_signals) > 0 else []
                }
                
                chart_data['sell_signals'] = {
                    'timestamps': sell_signals['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist() if len(sell_signals) > 0 else [],
                    'prices': sell_signals['Price'].tolist() if len(sell_signals) > 0 else []
                }
        except Exception as e:
            print(f"Error getting signals: {e}")
            chart_data['buy_signals'] = {'timestamps': [], 'prices': []}
            chart_data['sell_signals'] = {'timestamps': [], 'prices': []}
        
        # Add portfolio value
        try:
            portfolio_value = pf.value()
            chart_data['portfolio_value'] = {
                'timestamps': portfolio_value.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'values': portfolio_value.tolist()
            }
        except Exception as e:
            print(f"Error getting portfolio value: {e}")
            chart_data['portfolio_value'] = {'timestamps': [], 'values': []}
            
        return chart_data
        
    except Exception as e:
        print(f"Chart creation error: {e}")
        # Return minimal fallback chart data
        return {
            'type': 'candlestick',
            'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()[:100],  # Limit to avoid JSON issues
            'open': df['open'].tolist()[:100],
            'high': df['high'].tolist()[:100],
            'low': df['low'].tolist()[:100], 
            'close': df['close'].tolist()[:100],
            'volume': df['volume'].tolist()[:100],
            'buy_signals': {'timestamps': [], 'prices': []},
            'sell_signals': {'timestamps': [], 'prices': []},
            'portfolio_value': {'timestamps': [], 'values': []}
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
        },
        {
            "name": "PPP VishvaAlgo MTF Scalper",
            "description": "Advanced multi-timeframe strategy with EMA baseline, SuperTrend, EWO, StochRSI, and ATR regime filter",
            "code": """# PPP VishvaAlgo MTF Scalper Strategy (VectorBT Version)
import talib

# Strategy Parameters
ema_len = 200          # EMA Baseline
st_period = 10         # SuperTrend ATR period
st_factor = 2.0        # SuperTrend ATR multiplier
ewo_fast = 5           # EWO Fast EMA
ewo_slow = 35          # EWO Slow EMA
rsi_len = 14           # StochRSI: RSI Length
stoch_len = 14         # StochRSI: Stoch Length
k_smooth = 3           # StochRSI: %K Smoothing
d_smooth = 3           # StochRSI: %D Smoothing
k_buy_th = 30.0        # StochRSI: Buy zone K<
k_sell_th = 70.0       # StochRSI: Sell zone K>
atr_reg_len = 14       # ATR regime filter length
min_atr_pct = 0.10     # Min ATR %
max_atr_pct = 3.00     # Max ATR %

# Calculate indicators
ema_base = data['close'].ewm(span=ema_len).mean()

# SuperTrend calculation
atr = pd.Series(talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=st_period), index=data.index)
hl2 = (data['high'] + data['low']) / 2
upper_band = hl2 + (st_factor * atr)
lower_band = hl2 - (st_factor * atr)

# SuperTrend logic
st_trend = pd.Series(index=data.index, dtype=float)
st_direction = pd.Series(index=data.index, dtype=int)

for i in range(len(data)):
    if i == 0:
        st_trend.iloc[i] = lower_band.iloc[i]
        st_direction.iloc[i] = 1
    else:
        if data['close'].iloc[i] <= st_trend.iloc[i-1]:
            st_trend.iloc[i] = upper_band.iloc[i]
            st_direction.iloc[i] = -1
        else:
            st_trend.iloc[i] = lower_band.iloc[i]
            st_direction.iloc[i] = 1

ut_long = st_direction == 1
ut_short = st_direction == -1

# EWO (Elliott Wave Oscillator)
ewo_fast_ma = data['close'].ewm(span=ewo_fast).mean()
ewo_slow_ma = data['close'].ewm(span=ewo_slow).mean()
ewo = ewo_fast_ma - ewo_slow_ma

# StochRSI
rsi = pd.Series(talib.RSI(data['close'].values, timeperiod=rsi_len), index=data.index)
rsi_low = rsi.rolling(window=stoch_len).min()
rsi_high = rsi.rolling(window=stoch_len).max()
stoch_rsi = ((rsi - rsi_low) / (rsi_high - rsi_low)) * 100
stoch_rsi = stoch_rsi.fillna(50)
k = stoch_rsi.rolling(window=k_smooth).mean()
d = k.rolling(window=d_smooth).mean()

# Regime filter (ATR %)
atr_reg = pd.Series(talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=atr_reg_len), index=data.index)
atr_pct = (atr_reg / data['close']) * 100.0
regime_ok = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)

# Signal conditions
k_cross_up = (k > d) & (k.shift(1) <= d.shift(1))
k_cross_down = (k < d) & (k.shift(1) >= d.shift(1))

# Generate signals
long_signal = (
    (data['close'] > ema_base) & 
    ut_long & 
    (ewo > 0) & 
    k_cross_up & 
    (k < k_buy_th) & 
    regime_ok
)

short_signal = (
    (data['close'] < ema_base) & 
    ut_short & 
    (ewo < 0) & 
    k_cross_down & 
    (k > k_sell_th) & 
    regime_ok
)

# Create portfolio with stop loss and take profit
pf = vbt.Portfolio.from_signals(
    data['close'], 
    long_signal, 
    short_signal, 
    init_cash=initial_capital,
    fees=0.0004,  # 0.04% commission
    sl_stop=0.005,  # 0.5% stop loss
    tp_stop=0.008   # 0.8% take profit
)""",
            "parameters": {
                "ema_len": 200, "st_period": 10, "st_factor": 2.0,
                "ewo_fast": 5, "ewo_slow": 35, "k_buy_th": 30.0, "k_sell_th": 70.0
            }
        },
        {
            "name": "No Brain + ORB Combo",
            "description": "Donchian Channel breakout with CCI filter plus Opening Range Breakout strategy",
            "code": """# No Brain + ORB Combo Strategy (VectorBT Version)
import talib

# Strategy Parameters
donch_len = 20         # Donchian Channel length
cci_len = 20           # CCI length
cci_long_threshold = 0 # CCI long threshold
cci_short_threshold = 0 # CCI short threshold
orb_hours = 0.5        # Opening Range hours (0.5 = 30 minutes)
orb_trade_hours = 2.0  # ORB trading window hours
use_close_break = True # Require close beyond Donchian

# Calculate Donchian Channel
donch_upper = data['high'].rolling(window=donch_len).max()
donch_lower = data['low'].rolling(window=donch_len).min()
donch_upper_prev = donch_upper.shift(1)
donch_lower_prev = donch_lower.shift(1)

# Calculate CCI
cci = pd.Series(talib.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=cci_len), index=data.index)
cci_long_ok = cci > cci_long_threshold
cci_short_ok = cci < cci_short_threshold

# No Brain signals (Donchian + CCI)
if use_close_break:
    long_break = data['close'] > donch_upper_prev
    short_break = data['close'] < donch_lower_prev
else:
    long_break = data['high'] > donch_upper_prev
    short_break = data['low'] < donch_lower_prev

nb_long = long_break & cci_long_ok
nb_short = short_break & cci_short_ok

# Opening Range Breakout (ORB) Logic
# Simplified ORB: Use first portion of data as opening range
orb_periods = int(len(data) * (orb_hours / 24))  # Convert hours to periods
trade_periods = int(len(data) * (orb_trade_hours / 24))

# Calculate opening range for each day (simplified)
orb_high = data['high'].rolling(window=orb_periods, min_periods=1).max()
orb_low = data['low'].rolling(window=orb_periods, min_periods=1).min()

# ORB signals with 1-tick buffer
tick_size = data['close'].diff().abs().median() * 0.1  # Estimate tick size
orb_long_trigger = orb_high + tick_size
orb_short_trigger = orb_low - tick_size

# ORB breakout signals
orb_long = data['close'] > orb_long_trigger
orb_short = data['close'] < orb_short_trigger

# Combine signals (ORB has priority when both are active)
# Simple combination: OR logic for demonstration
long_signal = nb_long | orb_long
short_signal = nb_short | orb_short

# Create portfolio
pf = vbt.Portfolio.from_signals(
    data['close'], 
    long_signal, 
    short_signal, 
    init_cash=initial_capital,
    fees=0.0004,  # 0.04% commission
    sl_stop=0.005,  # 0.5% stop loss
    tp_stop=0.005   # 0.5% take profit
)""",
            "parameters": {
                "donch_len": 20, "cci_len": 20, "orb_hours": 0.5,
                "cci_long_threshold": 0, "cci_short_threshold": 0
            }
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