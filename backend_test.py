import requests
import sys
import json
from datetime import datetime

class CryptoBacktesterAPITester:
    def __init__(self, base_url="https://82ba9ddf-ec66-4583-96eb-bba80382f071.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timed out after {timeout} seconds")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "", 200)

    def test_symbols_endpoint(self):
        """Test symbols endpoint"""
        success, response = self.run_test("Symbols Endpoint", "GET", "symbols", 200)
        if success and isinstance(response, dict):
            symbols = response.get('symbols', [])
            print(f"   Found {len(symbols)} symbols")
            if len(symbols) > 0:
                print(f"   Sample symbols: {symbols[:5]}")
        return success

    def test_strategy_templates_endpoint(self):
        """Test strategy templates endpoint"""
        success, response = self.run_test("Strategy Templates Endpoint", "GET", "strategy-templates", 200)
        if success and isinstance(response, dict):
            templates = response.get('templates', [])
            print(f"   Found {len(templates)} templates")
            for i, template in enumerate(templates):
                print(f"   Template {i+1}: {template.get('name', 'Unknown')}")
        return success

    def test_backtest_endpoint(self):
        """Test backtest endpoint with sample strategy"""
        # Sample strategy from the request
        strategy_code = """# Simple Moving Average Crossover Strategy
fast_period = 10
slow_period = 20

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
    fees=0.001
)"""

        backtest_data = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "initial_capital": 10000.0,
            "strategy_code": strategy_code
        }

        print(f"   Testing with strategy: Simple Moving Average Crossover")
        print(f"   Parameters: {backtest_data['symbol']}, {backtest_data['timeframe']}, {backtest_data['start_date']} to {backtest_data['end_date']}")
        
        success, response = self.run_test("Backtest Endpoint", "POST", "backtest", 200, backtest_data, timeout=60)
        
        if success and isinstance(response, dict):
            # Check required fields in response
            required_fields = ['id', 'symbol', 'final_value', 'total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"   ⚠️  Missing fields in response: {missing_fields}")
            else:
                print(f"   📊 Results Summary:")
                print(f"      Total Return: {response.get('total_return', 0):.2f}%")
                print(f"      Sharpe Ratio: {response.get('sharpe_ratio', 0):.2f}")
                print(f"      Max Drawdown: {response.get('max_drawdown', 0):.2f}%")
                print(f"      Total Trades: {response.get('total_trades', 0)}")
                print(f"      Final Value: ${response.get('final_value', 0):.2f}")
        
        return success

    def test_backtest_history_endpoint(self):
        """Test backtest history endpoint"""
        success, response = self.run_test("Backtest History Endpoint", "GET", "backtest-history", 200)
        if success and isinstance(response, list):
            print(f"   Found {len(response)} historical backtests")
            if len(response) > 0:
                latest = response[0]
                print(f"   Latest backtest: {latest.get('symbol', 'Unknown')} - {latest.get('total_return', 0):.2f}% return")
        return success

    def test_invalid_backtest(self):
        """Test backtest with invalid data"""
        invalid_data = {
            "symbol": "INVALID",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "initial_capital": 10000.0,
            "strategy_code": "invalid code"
        }

        success, response = self.run_test("Invalid Backtest (Error Handling)", "POST", "backtest", 400, invalid_data)
        # For this test, we expect it to fail (400 status), so success means proper error handling
        return not success  # Invert because we expect this to fail

def main():
    print("🚀 Starting Crypto Backtester API Tests")
    print("=" * 50)
    
    tester = CryptoBacktesterAPITester()
    
    # Run all tests
    tests = [
        tester.test_root_endpoint,
        tester.test_symbols_endpoint,
        tester.test_strategy_templates_endpoint,
        tester.test_backtest_endpoint,
        tester.test_backtest_history_endpoint,
        tester.test_invalid_backtest
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            tester.tests_run += 1

    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed! Backend API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())