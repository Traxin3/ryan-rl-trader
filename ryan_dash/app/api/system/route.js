import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/system`, {
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
    console.error('System API error:', error);
    return NextResponse.json({ 
      error: error.message,
      status: 'offline',
      timestamp: new Date().toISOString(),
      metrics: {
        gpuUsage: 0,
        cpuUsage: 0,
        memoryUsage: 0,
        timestamp: new Date().toISOString()
      }
    }, { status: 500 });
  }
} 