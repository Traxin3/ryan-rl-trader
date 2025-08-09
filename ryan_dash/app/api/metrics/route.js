import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/metrics`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Metrics API error:', error);
    return NextResponse.json({ 
      error: error.message,
      metrics: {
        is_training: false,
        progress: 0,
        steps: 0,
        episodes: 0,
        current_reward: 0,
        win_rate: 0,
        sharpe_ratio: 0,
        balance: 10000,
        equity: 10000,
        active_positions: 0,
        peak_equity: 10000,
        max_drawdown: 0,
        total_trades: 0,
        profit_factor: 0,
        sortino_ratio: 0,
        volatility: 0,
        avg_trade_profit: 0,
        avg_holding_time: 0
      },
      system: {
        gpuUsage: 0,
        cpuUsage: 0,
        memoryUsage: 0,
        timestamp: new Date().toISOString()
      },
      history: []
    }, { status: 500 });
  }
} 