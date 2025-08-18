import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { Alert, AlertDescription } from './components/ui/alert';
import { Play, TrendingUp, DollarSign, BarChart3, Activity, Clock, Target } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [symbols, setSymbols] = useState([]);
  const [templates, setTemplates] = useState([]);
  const [backtestHistory, setBacktestHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  
  // Form state
  const [formData, setFormData] = useState({
    symbol: 'BTCUSDT',
    timeframe: '1h',
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    initial_capital: 10000,
    strategy_code: ''
  });

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [symbolsRes, templatesRes, historyRes] = await Promise.all([
        axios.get(`${API}/symbols`),
        axios.get(`${API}/strategy-templates`),
        axios.get(`${API}/backtest-history`)
      ]);
      
      setSymbols(symbolsRes.data.symbols || []);
      setTemplates(templatesRes.data.templates || []);
      setBacktestHistory(historyRes.data || []);
      
      // Set default strategy code if templates are available
      if (templatesRes.data.templates && templatesRes.data.templates.length > 0) {
        setFormData(prev => ({ ...prev, strategy_code: templatesRes.data.templates[0].code }));
      }
    } catch (err) {
      console.error('Failed to load initial data:', err);
      setError('Failed to load initial data');
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const runBacktest = async () => {
    if (!formData.strategy_code.trim()) {
      setError('Please enter a strategy code');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${API}/backtest`, formData);
      setResult(response.data);
      
      // Refresh backtest history
      const historyRes = await axios.get(`${API}/backtest-history`);
      setBacktestHistory(historyRes.data || []);
      
    } catch (err) {
      console.error('Backtest failed:', err);
      setError(err.response?.data?.detail || 'Backtest failed');
    } finally {
      setIsLoading(false);
    }
  };

  const loadTemplate = (template) => {
    setFormData(prev => ({ ...prev, strategy_code: template.code }));
  };

  const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US', { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    }).format(num);
  };

  const formatPercent = (num) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'percent', 
      minimumFractionDigits: 2 
    }).format(num / 100);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">CryptoBacktester</h1>
                <p className="text-sm text-slate-600">Professional Cryptocurrency Strategy Backtesting</p>
              </div>
            </div>
            <Badge variant="secondary" className="px-3 py-1">
              Powered by VectorBT
            </Badge>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Panel - Strategy Configuration */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="border-slate-200 shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold text-slate-900">Strategy Configuration</CardTitle>
                <CardDescription className="text-slate-600">Configure your backtesting parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                
                {/* Symbol Selection */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">Trading Pair</label>
                  <Select value={formData.symbol} onValueChange={(value) => handleInputChange('symbol', value)}>
                    <SelectTrigger className="bg-white">
                      <SelectValue placeholder="Select symbol" />
                    </SelectTrigger>
                    <SelectContent>
                      {symbols.map(symbol => (
                        <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Timeframe */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">Timeframe</label>
                  <Select value={formData.timeframe} onValueChange={(value) => handleInputChange('timeframe', value)}>
                    <SelectTrigger className="bg-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1m">1 Minute</SelectItem>
                      <SelectItem value="5m">5 Minutes</SelectItem>
                      <SelectItem value="15m">15 Minutes</SelectItem>
                      <SelectItem value="1h">1 Hour</SelectItem>
                      <SelectItem value="4h">4 Hours</SelectItem>
                      <SelectItem value="1d">1 Day</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Date Range */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700">Start Date</label>
                    <Input
                      type="date"
                      value={formData.start_date}
                      onChange={(e) => handleInputChange('start_date', e.target.value)}
                      className="bg-white"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700">End Date</label>
                    <Input
                      type="date"
                      value={formData.end_date}
                      onChange={(e) => handleInputChange('end_date', e.target.value)}
                      className="bg-white"
                    />
                  </div>
                </div>

                {/* Initial Capital */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">Initial Capital (USDT)</label>
                  <Input
                    type="number"
                    value={formData.initial_capital}
                    onChange={(e) => handleInputChange('initial_capital', parseFloat(e.target.value))}
                    className="bg-white"
                    min="100"
                  />
                </div>

              </CardContent>
            </Card>

            {/* Strategy Templates */}
            <Card className="border-slate-200 shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-semibold text-slate-900">Strategy Templates</CardTitle>
                <CardDescription className="text-slate-600">Quick start with predefined strategies</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {templates.map((template, index) => (
                  <div key={index} className="p-3 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors">
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h4 className="font-medium text-slate-900 text-sm">{template.name}</h4>
                        <p className="text-xs text-slate-600 mt-1">{template.description}</p>
                      </div>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => loadTemplate(template)}
                        className="ml-2 h-7 text-xs"
                      >
                        Use
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Main Panel */}
          <div className="lg:col-span-2 space-y-6">
            
            <Tabs defaultValue="strategy" className="w-full">
              <TabsList className="grid w-full grid-cols-3 bg-slate-100">
                <TabsTrigger value="strategy" className="data-[state=active]:bg-white">Strategy Code</TabsTrigger>
                <TabsTrigger value="results" className="data-[state=active]:bg-white">Results</TabsTrigger>
                <TabsTrigger value="history" className="data-[state=active]:bg-white">History</TabsTrigger>
              </TabsList>

              {/* Strategy Code Tab */}
              <TabsContent value="strategy" className="space-y-4">
                <Card className="border-slate-200 shadow-sm">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg font-semibold text-slate-900">Python Strategy Code</CardTitle>
                    <CardDescription className="text-slate-600">
                      Write your trading strategy using VectorBT. Use 'data' DataFrame for OHLCV data.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Textarea
                      value={formData.strategy_code}
                      onChange={(e) => handleInputChange('strategy_code', e.target.value)}
                      placeholder="# Enter your VectorBT strategy code here..."
                      className="min-h-[400px] font-mono text-sm bg-slate-50"
                    />
                    
                    <div className="flex justify-between items-center mt-4">
                      <div className="text-xs text-slate-500">
                        Available variables: data (OHLCV DataFrame), initial_capital (float)
                      </div>
                      <Button 
                        onClick={runBacktest}
                        disabled={isLoading}
                        className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                      >
                        {isLoading ? (
                          <>
                            <Clock className="h-4 w-4 mr-2 animate-spin" />
                            Running...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4 mr-2" />
                            Run Backtest
                          </>
                        )}
                      </Button>
                    </div>

                    {error && (
                      <Alert className="mt-4 border-red-200 bg-red-50">
                        <AlertDescription className="text-red-700">{error}</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Results Tab */}
              <TabsContent value="results" className="space-y-6">
                {result ? (
                  <>
                    {/* Metrics Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <Card className="border-slate-200 shadow-sm">
                        <CardContent className="p-4">
                          <div className="flex items-center space-x-2">
                            <DollarSign className="h-5 w-5 text-green-600" />
                            <div>
                              <p className="text-xs font-medium text-slate-600">Total Return</p>
                              <p className={`text-lg font-bold ${result.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                {formatPercent(result.total_return)}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="border-slate-200 shadow-sm">
                        <CardContent className="p-4">
                          <div className="flex items-center space-x-2">
                            <BarChart3 className="h-5 w-5 text-blue-600" />
                            <div>
                              <p className="text-xs font-medium text-slate-600">Sharpe Ratio</p>
                              <p className="text-lg font-bold text-slate-900">{formatNumber(result.sharpe_ratio)}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="border-slate-200 shadow-sm">
                        <CardContent className="p-4">
                          <div className="flex items-center space-x-2">
                            <Activity className="h-5 w-5 text-red-600" />
                            <div>
                              <p className="text-xs font-medium text-slate-600">Max Drawdown</p>
                              <p className="text-lg font-bold text-red-600">{formatPercent(result.max_drawdown)}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="border-slate-200 shadow-sm">
                        <CardContent className="p-4">
                          <div className="flex items-center space-x-2">
                            <Target className="h-5 w-5 text-purple-600" />
                            <div>
                              <p className="text-xs font-medium text-slate-600">Win Rate</p>
                              <p className="text-lg font-bold text-purple-600">{formatPercent(result.win_rate)}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Additional Metrics */}
                    <Card className="border-slate-200 shadow-sm">
                      <CardHeader className="pb-4">
                        <CardTitle className="text-lg font-semibold text-slate-900">Performance Summary</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
                          <div>
                            <p className="text-slate-600 font-medium">Initial Capital</p>
                            <p className="text-lg font-semibold text-slate-900">${formatNumber(result.initial_capital)}</p>
                          </div>
                          <div>
                            <p className="text-slate-600 font-medium">Final Value</p>
                            <p className="text-lg font-semibold text-slate-900">${formatNumber(result.final_value)}</p>
                          </div>
                          <div>
                            <p className="text-slate-600 font-medium">Total Trades</p>
                            <p className="text-lg font-semibold text-slate-900">{result.total_trades}</p>
                          </div>
                          <div>
                            <p className="text-slate-600 font-medium">Profit Factor</p>
                            <p className="text-lg font-semibold text-slate-900">{formatNumber(result.profit_factor)}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Chart Placeholder */}
                    <Card className="border-slate-200 shadow-sm">
                      <CardHeader className="pb-4">
                        <CardTitle className="text-lg font-semibold text-slate-900">Price Chart & Signals</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-96 bg-slate-50 rounded-lg flex items-center justify-center">
                          <div className="text-center">
                            <BarChart3 className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                            <p className="text-slate-600">Interactive chart would be rendered here</p>
                            <p className="text-xs text-slate-500 mt-2">Chart data available in API response</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </>
                ) : (
                  <Card className="border-slate-200 shadow-sm">
                    <CardContent className="p-12 text-center">
                      <BarChart3 className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold text-slate-900 mb-2">No Results Yet</h3>
                      <p className="text-slate-600">Run a backtest to see detailed performance metrics and charts</p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              {/* History Tab */}
              <TabsContent value="history" className="space-y-4">
                <Card className="border-slate-200 shadow-sm">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg font-semibold text-slate-900">Backtest History</CardTitle>
                    <CardDescription className="text-slate-600">Previous backtest results</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {backtestHistory.length > 0 ? (
                      <div className="space-y-3">
                        {backtestHistory.slice(0, 10).map((item, index) => (
                          <div key={item.id} className="p-4 border border-slate-200 rounded-lg">
                            <div className="flex justify-between items-start">
                              <div>
                                <div className="flex items-center space-x-2">
                                  <Badge variant="outline">{item.symbol}</Badge>
                                  <Badge variant="outline">{item.timeframe}</Badge>
                                  <span className="text-xs text-slate-500">
                                    {new Date(item.created_at).toLocaleDateString()}
                                  </span>
                                </div>
                                <p className="text-sm text-slate-600 mt-1">
                                  {item.start_date} to {item.end_date}
                                </p>
                              </div>
                              <div className="text-right">
                                <p className={`font-semibold ${item.total_return >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {formatPercent(item.total_return)}
                                </p>
                                <p className="text-xs text-slate-500">{item.total_trades} trades</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <Clock className="h-12 w-12 text-slate-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-slate-900 mb-2">No History Yet</h3>
                        <p className="text-slate-600">Your completed backtests will appear here</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;