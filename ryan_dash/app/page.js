'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  BarChart3, 
  TrendingUp, 
  Cpu, 
  HardDrive,
  Zap,
  Target,
  DollarSign,
  Activity,
  Clock,
  Users,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RotateCcw,
  Save,
  Download,
  Upload,
  Eye,
  EyeOff,
  ChevronLeft,
  ChevronRight,
  Home,
  Database,
  Code,
  Monitor,
  Maximize,
  Minimize,
  Filter,
  Layers,
  GitCompare,
  HardDriveDownload,
  HardDriveUpload
} from 'lucide-react';

// Utility function for class merging
const cn = (...classes) => classes.filter(Boolean).join(' ');

// Helper functions
const formatUptime = (seconds) => {
  if (!seconds) return '00:00:00';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const computeLinePath = (data, metric) => {
  if (!data.length) return '';
  const points = data.map((point, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - (point[metric] * 100);
    return `${x}% ${y}%`;
  });
  return `M ${points.join(' L ')}`;
};

// Model training state - all values start at 0/null
const initialMetrics = {
  // Model training metrics
  isTraining: false,
  progress: 0,
  steps: 0,
  episodes: 0,
  currentReward: 0,
  loss: 0,
  accuracy: 0,
  learningRate: 0,
  
  // System metrics
  gpuUsage: null, // Initialize as null to detect GPU presence
  cpuUsage: 0,
  memoryUsage: 0,
  systemLoad: '0.0, 0.0, 0.0',
  threadCount: 0,
  pythonVersion: '',
  uptime: 0,
  
  // Model state
  modelVersion: 'v1.0',
  lastSaved: '',
  trainingDuration: '0h 0m',
  
  // History data
  history: []
};

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState(initialMetrics);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [config, setConfig] = useState({
    symbols: [],
    timeframes: [],
    windowSize: 0,
    maxLeverage: 0,
    learningRate: 0,
    nSteps: 0,
    batchSize: 0,
    nEpochs: 0,
    gamma: 0,
    gaeLambda: 0,
    clipRange: 0,
    entCoef: 0,
    maxGradNorm: 0,
    vfCoef: 0
  });
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [backendConnected, setBackendConnected] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);

  // Load configuration on mount
  useEffect(() => {
    loadConfig();
    loadTrainingHistory();
  }, []);

  // Initial load only - no automatic updates to prevent refresh loops
  useEffect(() => {
    loadConfig();
    loadTrainingHistory();
    updateMetrics();
  }, []);

  // Manual refresh function
  const handleRefresh = useCallback(async () => {
    await updateMetrics();
  }, []);

  const loadConfig = async () => {
    try {
      const response = await fetch('/api/config');
      if (response.ok) {
        const data = await response.json();
        if (data.ppo && data.env) {
          setConfig({
            symbols: data.env.symbols || ['EURUSD'],
            timeframes: data.env.timeframes || [15],
            windowSize: data.env.window_size || 50,
            maxLeverage: data.env.max_leverage || 2.0,
            learningRate: data.ppo.learning_rate || 0.0004,
            nSteps: data.ppo.n_steps || 2048,
            batchSize: data.ppo.batch_size || 256,
            nEpochs: data.ppo.n_epochs || 5,
            gamma: data.ppo.gamma || 0.99,
            gaeLambda: data.ppo.gae_lambda || 0.95,
            clipRange: data.ppo.clip_range || 0.2,
            entCoef: data.ppo.ent_coef || 0.01,
            maxGradNorm: data.ppo.max_grad_norm || 0.5,
            vfCoef: data.ppo.vf_coef || 0.25
          });
        }
      }
    } catch (error) {
      console.error('Error loading config:', error);
    }
  };

  const loadTrainingHistory = async () => {
    try {
      const response = await fetch('/api/history');
      if (response.ok) {
        const data = await response.json();
        setTrainingHistory(data);
      }
    } catch (error) {
      console.error('Error loading training history:', error);
    }
  };

  const updateMetrics = async () => {
    if (isUpdating) return;
    
    try {
      setIsUpdating(true);
      const response = await fetch('/api/metrics');
      if (response.ok) {
        const data = await response.json();
        setBackendConnected(true);
        
        if (data.metrics) {
          setMetrics(prev => ({
            ...prev,
            ...data.metrics,
            currentTime: new Date().toLocaleTimeString(),
            history: [...(prev.history || []), {
              timestamp: Date.now(),
              reward: data.metrics.currentReward || 0,
              loss: data.metrics.loss || 0,
              accuracy: data.metrics.accuracy || 0,
              learningRate: data.metrics.learningRate || 0,
              step: data.metrics.steps || 0
            }].slice(-1000) // Keep last 1000 data points
          }));
          setIsTraining(data.metrics.is_training || false);
        }
        
        if (data.system) {
          setMetrics(prev => ({
            ...prev,
            gpuUsage: data.system.gpuUsage,
            cpuUsage: data.system.cpuUsage || 0,
            memoryUsage: data.system.memoryUsage || 0,
            systemLoad: data.system.load || '0.0, 0.0, 0.0',
            threadCount: data.system.threads || 0,
            pythonVersion: data.system.python || '',
            uptime: data.system.uptime || 0
          }));
        }
      } else {
        setBackendConnected(false);
      }
    } catch (error) {
      console.error('Error updating metrics:', error);
      setBackendConnected(false);
    } finally {
      setIsUpdating(false);
    }
  };

  const startTraining = async () => {
    try {
      const response = await fetch('/api/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'start',
          config: {
            symbols: config.symbols,
            timeframes: config.timeframes,
            windowSize: config.windowSize,
            maxLeverage: config.maxLeverage,
            learningRate: config.learningRate,
            nSteps: config.nSteps,
            batchSize: config.batchSize,
            nEpochs: config.nEpochs,
            gamma: config.gamma,
            gaeLambda: config.gaeLambda,
            clipRange: config.clipRange,
            entCoef: config.entCoef,
            maxGradNorm: config.maxGradNorm,
            vfCoef: config.vfCoef
          }
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Training started:', data);
      } else {
        console.error('Failed to start training');
      }
    } catch (error) {
      console.error('Error starting training:', error);
    }
  };

  const stopTraining = async () => {
    try {
      const response = await fetch('/api/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'stop'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Training stopped:', data);
      } else {
        console.error('Failed to stop training');
      }
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  const pauseTraining = async () => {
    try {
      const response = await fetch('/api/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'pause'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Training paused:', data);
      } else {
        console.error('Failed to pause training');
      }
    } catch (error) {
      console.error('Error pausing training:', error);
    }
  };

  const saveModel = async () => {
    try {
      const response = await fetch('/api/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'save'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Model save triggered:', data);
      } else {
        console.error('Failed to save model');
      }
    } catch (error) {
      console.error('Error saving model:', error);
    }
  };

  // Enhanced Components
  const EnhancedTrainingChart = ({ data, backendConnected }) => {
    const [viewMode, setViewMode] = useState('reward');
    const [timeRange, setTimeRange] = useState('1h');
    const [hoveredPoint, setHoveredPoint] = useState(null);
    const [fullscreen, setFullscreen] = useState(false);
    
    const filteredData = useMemo(() => {
      if (!data.length) return [];
      const now = Date.now();
      const cutoff = timeRange === '1h' ? now - 3600000 :
                    timeRange === '6h' ? now - 21600000 :
                    timeRange === '24h' ? now - 86400000 : 0;
      return data.filter(point => point.timestamp >= cutoff);
    }, [data, timeRange]);

    const chartConfig = {
      reward: {
        color: '#4fd1c5',
        label: 'Reward',
        formatter: v => v.toFixed(3)
      },
      loss: {
        color: '#f56565',
        label: 'Loss',
        formatter: v => v.toFixed(4)
      },
      accuracy: {
        color: '#48bb78',
        label: 'Accuracy',
        formatter: v => `${(v * 100).toFixed(1)}%`
      }
    };

    if (!backendConnected) {
      return (
        <div className="h-64 flex items-center justify-center text-gray-500">
          Backend not connected
        </div>
      );
    }

    return (
      <div className={`${fullscreen ? 'fixed inset-0 z-50 bg-gray-900 p-8' : 'bg-gray-800/50 rounded-xl p-4 border border-gray-700/50'}`}>
        <div className="flex justify-between items-center mb-4">
          <div className="flex space-x-2">
            {Object.keys(chartConfig).map(key => (
              <button
                key={key}
                onClick={() => setViewMode(key)}
                className={`px-3 py-1 rounded-md text-sm ${
                  viewMode === key 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {chartConfig[key].label}
              </button>
            ))}
          </div>
          <div className="flex space-x-2">
            {['1h', '6h', '24h', 'all'].map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-2 py-1 rounded-md text-xs ${
                  timeRange === range 
                    ? 'bg-purple-500 text-white' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {range}
              </button>
            ))}
            <button 
              onClick={() => setFullscreen(!fullscreen)}
              className="p-1 rounded-md hover:bg-gray-700 text-gray-300"
            >
              {fullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
            </button>
          </div>
        </div>
        
        <div className={`relative ${fullscreen ? 'h-[calc(100vh-160px)]' : 'h-64'}`}>
          <svg width="100%" height="100%" className="rounded-lg">
            <g fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1">
              {[...Array(5)].map((_, i) => (
                <line 
                  key={`hgrid-${i}`} 
                  x1="0" 
                  y1={`${(i + 1) * 20}%`} 
                  x2="100%" 
                  y2={`${(i + 1) * 20}%`} 
                />
              ))}
            </g>
            
            <path
              d={computeLinePath(filteredData, viewMode)}
              stroke={chartConfig[viewMode].color}
              strokeWidth="2"
              fill="none"
            />
            
            {filteredData.map((point, i) => (
              <circle
                key={i}
                cx={`${(i / (filteredData.length - 1)) * 100}%`}
                cy={`${100 - (point[viewMode] * 100)}%`}
                r="4"
                fill={chartConfig[viewMode].color}
                onMouseEnter={() => setHoveredPoint(point)}
                onMouseLeave={() => setHoveredPoint(null)}
                className="cursor-pointer"
              />
            ))}
            
            {hoveredPoint && (
              <g>
                <line
                  x1={`${(filteredData.indexOf(hoveredPoint) / (filteredData.length - 1)) * 100}%`}
                  y1="0%"
                  x2={`${(filteredData.indexOf(hoveredPoint) / (filteredData.length - 1)) * 100}%`}
                  y2="100%"
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth="1"
                  strokeDasharray="4 2"
                />
                <circle
                  cx={`${(filteredData.indexOf(hoveredPoint) / (filteredData.length - 1)) * 100}%`}
                  cy={`${100 - (hoveredPoint[viewMode] * 100)}%`}
                  r="8"
                  fill="transparent"
                  stroke={chartConfig[viewMode].color}
                  strokeWidth="2"
                />
              </g>
            )}
          </svg>
          
          {hoveredPoint && (
            <div 
              className="absolute bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg"
              style={{
                left: `${(filteredData.indexOf(hoveredPoint) / (filteredData.length - 1)) * 100}%`,
                top: '10%',
                transform: 'translateX(-50%)'
              }}
            >
              <div className="text-sm text-white">
                <div>Time: {new Date(hoveredPoint.timestamp).toLocaleTimeString()}</div>
                <div>{chartConfig[viewMode].label}: {chartConfig[viewMode].formatter(hoveredPoint[viewMode])}</div>
                <div className="text-gray-400 text-xs mt-1">
                  Step: {hoveredPoint.step.toLocaleString()}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const ModelMetricsVisualization = ({ metrics }) => {
    const [selectedMetric, setSelectedMetric] = useState('loss');
    const [compareMode, setCompareMode] = useState(false);
    const [historyRange, setHistoryRange] = useState('session');
    
    const metricOptions = [
      { id: 'loss', name: 'Loss', color: '#f56565' },
      { id: 'accuracy', name: 'Accuracy', color: '#48bb78' },
      { id: 'reward', name: 'Reward', color: '#4fd1c5' },
      { id: 'learningRate', name: 'Learning Rate', color: '#9f7aea' }
    ];
    
    const MiniSparkline = ({ data, color }) => {
      if (!data || !data.length) return null;
      
      const maxVal = Math.max(...data);
      const minVal = Math.min(...data);
      const range = maxVal - minVal;
      
      return (
        <svg width="100%" height="100%">
          <path
            d={data.map((val, i) => {
              const x = (i / (data.length - 1)) * 100;
              const y = 100 - ((val - minVal) / (range || 1) * 100);
              return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
            }).join(' ')}
            stroke={color}
            strokeWidth="1.5"
            fill="none"
          />
        </svg>
      );
    };
    
    const DetailedMetricChart = ({ metric, data, color }) => {
      const filteredData = data.slice(-50); // Show last 50 points for detail
      
      return (
        <div className="relative h-full w-full">
          <svg width="100%" height="100%">
            <g fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1">
              {[...Array(5)].map((_, i) => (
                <line 
                  key={`hgrid-${i}`} 
                  x1="0" 
                  y1={`${(i + 1) * 20}%`} 
                  x2="100%" 
                  y2={`${(i + 1) * 20}%`} 
                />
              ))}
            </g>
            
            <path
              d={computeLinePath(filteredData, metric)}
              stroke={color}
              strokeWidth="2"
              fill="none"
            />
          </svg>
        </div>
      );
    };
    
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-white">Model Performance Metrics</h3>
          <div className="flex space-x-4">
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-3 py-1 rounded-md text-sm ${
                compareMode 
                  ? 'bg-purple-500 text-white' 
                  : 'bg-gray-700 text-gray-300'
              }`}
            >
              {compareMode ? 'Comparison Mode' : 'Single Metric'}
            </button>
            <select
              value={historyRange}
              onChange={(e) => setHistoryRange(e.target.value)}
              className="bg-gray-700 text-gray-300 rounded-md px-3 py-1 text-sm"
            >
              <option value="session">Current Session</option>
              <option value="day">Last 24 Hours</option>
              <option value="week">Last 7 Days</option>
            </select>
          </div>
        </div>
        
        {compareMode ? (
          <div className="grid grid-cols-2 gap-6">
            {metricOptions.map(metric => (
              <div key={metric.id} className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300 mb-3">{metric.name}</h4>
                <div className="h-40">
                  <MiniSparkline 
                    data={metrics.history.map(m => m[metric.id])}
                    color={metric.color}
                  />
                </div>
                <div className="mt-2 text-right">
                  <span className="text-xl font-bold text-white">
                    {metrics[metric.id]?.toFixed(metric.id === 'accuracy' ? 2 : 4)}
                  </span>
                  <span className="text-xs text-gray-400 ml-1">
                    {metric.id === 'accuracy' ? '%' : ''}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex space-x-4">
              {metricOptions.map(metric => (
                <button
                  key={metric.id}
                  onClick={() => setSelectedMetric(metric.id)}
                  className={`px-3 py-1 rounded-md text-sm ${
                    selectedMetric === metric.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {metric.name}
                </button>
              ))}
            </div>
            
            <div className="h-64">
              <DetailedMetricChart 
                metric={selectedMetric}
                data={metrics.history}
                color={metricOptions.find(m => m.id === selectedMetric)?.color || '#4fd1c5'}
              />
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400">Current Value</div>
                <div className="text-xl font-bold text-white">
                  {metrics[selectedMetric]?.toFixed(4)}
                </div>
              </div>
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400">Min</div>
                <div className="text-xl font-bold text-white">
                  {Math.min(...metrics.history.map(m => m[selectedMetric]))?.toFixed(4)}
                </div>
              </div>
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400">Max</div>
                <div className="text-xl font-bold text-white">
                  {Math.max(...metrics.history.map(m => m[selectedMetric]))?.toFixed(4)}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const SystemResourceMonitor = ({ metrics }) => {
    const hasGPU = metrics.gpuUsage !== undefined && metrics.gpuUsage !== null;
    
    const getCpuInfo = () => {
      return `Cores: ${navigator.hardwareConcurrency || 'Unknown'}`;
    };
    
    const getGpuInfo = () => {
      return hasGPU ? 'NVIDIA GPU detected' : 'No GPU detected';
    };
    
    const getMemoryInfo = () => {
      return `Total: ${(metrics.memoryTotal || 0).toFixed(1)}GB`;
    };
    
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
        <h3 className="text-lg font-semibold text-white mb-6">System Resources</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* CPU Monitor */}
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-blue-400" />
                <span className="text-sm font-medium text-gray-300">CPU Usage</span>
              </div>
              <span className="text-sm font-bold text-white">
                {metrics.cpuUsage || 0}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-400 to-blue-600 h-2 rounded-full"
                style={{ width: `${metrics.cpuUsage || 0}%` }}
              />
            </div>
            <div className="mt-2 text-xs text-gray-400">
              {getCpuInfo()}
            </div>
          </div>
          
          {/* GPU Monitor - only shown if GPU is available */}
          {hasGPU && (
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Cpu className="w-5 h-5 text-purple-400" />
                  <span className="text-sm font-medium text-gray-300">GPU Usage</span>
                </div>
                <span className="text-sm font-bold text-white">
                  {metrics.gpuUsage}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-purple-400 to-purple-600 h-2 rounded-full"
                  style={{ width: `${metrics.gpuUsage}%` }}
                />
              </div>
              <div className="mt-2 text-xs text-gray-400">
                {getGpuInfo()}
              </div>
            </div>
          )}
          
          {/* Memory Monitor */}
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <HardDrive className="w-5 h-5 text-green-400" />
                <span className="text-sm font-medium text-gray-300">Memory Usage</span>
              </div>
              <span className="text-sm font-bold text-white">
                {metrics.memoryUsage || 0}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-green-400 to-green-600 h-2 rounded-full"
                style={{ width: `${metrics.memoryUsage || 0}%` }}
              />
            </div>
            <div className="mt-2 text-xs text-gray-400">
              {getMemoryInfo()}
            </div>
          </div>
        </div>
        
        {/* Additional system info */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-900/50 rounded-lg p-3">
            <div className="text-xs text-gray-400">Process Uptime</div>
            <div className="text-sm font-medium text-white">
              {formatUptime(metrics.uptime)}
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-3">
            <div className="text-xs text-gray-400">System Load</div>
            <div className="text-sm font-medium text-white">
              {metrics.systemLoad || '--'}
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-3">
            <div className="text-xs text-gray-400">Threads</div>
            <div className="text-sm font-medium text-white">
              {metrics.threadCount || '--'}
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-3">
            <div className="text-xs text-gray-400">Python Version</div>
            <div className="text-sm font-medium text-white">
              {metrics.pythonVersion || '--'}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const MetricCard = ({ title, value, icon: Icon, color = 'blue', trend = null, backendConnected }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn(
        "bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl",
        "border border-gray-700/50 rounded-xl p-6",
        "hover:border-gray-600/50 transition-all duration-300",
        "group cursor-pointer"
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={cn(
            "p-2 rounded-lg",
            color === 'blue' && "bg-blue-500/20 text-blue-400",
            color === 'green' && "bg-green-500/20 text-green-400",
            color === 'red' && "bg-red-500/20 text-red-400",
            color === 'purple' && "bg-purple-500/20 text-purple-400",
            color === 'yellow' && "bg-yellow-500/20 text-yellow-400"
          )}>
            <Icon className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-medium text-gray-300">{title}</h3>
        </div>
        {trend && (
          <div className={cn(
            "flex items-center text-xs",
            trend > 0 ? "text-green-400" : "text-red-400"
          )}>
            {trend > 0 ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingUp className="w-3 h-3 mr-1 rotate-180" />}
            {Math.abs(trend)}%
          </div>
        )}
      </div>
      <div className="text-2xl font-bold text-white mb-2">
        {!backendConnected ? 'No Data' :
         typeof value === 'string' ? value :
         typeof value === 'number' && value >= 1000 ? value.toLocaleString() : 
         typeof value === 'number' && value > 0 ? value : 
         value === 0 ? '0' : '--'}
      </div>
      <div className="text-xs text-gray-400">
        {title === 'Training Progress' && value > 0 ? `${value.toFixed(1)}%` : '--'}
        {title === 'Steps' && value > 0 ? value.toLocaleString() : '--'}
        {title === 'Episodes' && value > 0 ? value.toLocaleString() : '--'}
        {title === 'Current Reward' && value > 0 ? value.toFixed(3) : '--'}
        {title === 'Loss' && value > 0 ? value.toFixed(4) : '--'}
        {title === 'Accuracy' && value > 0 ? `${(value * 100).toFixed(1)}%` : '--'}
        {title === 'Learning Rate' && value > 0 ? value.toFixed(6) : '--'}
        {title === 'GPU Usage' && value > 0 ? `${value}%` : '--'}
        {title === 'CPU Usage' && value > 0 ? `${value}%` : '--'}
        {title === 'Memory Usage' && value > 0 ? `${value}%` : '--'}
      </div>
    </motion.div>
  );

  const Sidebar = () => (
    <motion.div
      initial={{ x: -300 }}
      animate={{ x: sidebarOpen ? 0 : -300 }}
      transition={{ type: "spring", damping: 20 }}
      className="fixed left-0 top-0 h-full w-80 bg-gradient-to-b from-gray-900/95 to-gray-800/95 backdrop-blur-xl border-r border-gray-700/50 z-50"
    >
      <div className="p-6">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-xl font-bold text-white flex items-center">
            <Zap className="w-6 h-6 mr-2 text-blue-400" />
            Ryan RL
          </h1>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
          >
            <XCircle className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        <nav className="space-y-2">
          {[
            { id: 'overview', label: 'Model Control', icon: Home },
            { id: 'training', label: 'Training Status', icon: Activity },
            { id: 'metrics', label: 'Model Metrics', icon: BarChart3 },
            { id: 'config', label: 'Model Config', icon: Settings },
            { id: 'history', label: 'Training History', icon: Clock },
            { id: 'system', label: 'System Monitor', icon: Monitor }
          ].map((item) => (
            <motion.button
              key={item.id}
              whileHover={{ x: 5 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200",
                activeTab === item.id
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                  : "text-gray-300 hover:bg-gray-700/50 hover:text-white"
              )}
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </motion.button>
          ))}
        </nav>
      </div>
    </motion.div>
  );

  const OverviewTab = () => (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">RL Model Control</h2>
          <p className="text-gray-400">Model training and monitoring</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm text-gray-400">
            <div className={cn(
              "w-2 h-2 rounded-full",
              backendConnected ? "bg-green-400 animate-pulse" : "bg-red-400"
            )}></div>
            <span>{backendConnected ? "Backend Connected" : "Backend Disconnected"}</span>
          </div>
          <div className="text-sm text-gray-400">
            {backendConnected ? metrics.currentTime : 'No Data'}
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRefresh}
            disabled={isUpdating}
            className={cn(
              "flex items-center space-x-2 px-3 py-1 rounded-lg text-xs transition-colors",
              isUpdating 
                ? "bg-gray-500/20 text-gray-400 cursor-not-allowed"
                : "bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
            )}
          >
            <RotateCcw className={cn("w-3 h-3", isUpdating && "animate-spin")} />
            <span>{isUpdating ? "Updating..." : "Refresh"}</span>
          </motion.button>
        </div>
      </div>

      {/* Control Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">Model Control Panel</h3>
          <div className="flex items-center space-x-2">
            <div className={cn(
              "w-3 h-3 rounded-full",
              !backendConnected ? "bg-red-400" :
              isTraining ? "bg-green-400 animate-pulse" : "bg-gray-500"
            )}></div>
            <span className="text-sm text-gray-400">
              {!backendConnected ? "Backend Disconnected" :
               isTraining ? "Model Training Active" : "Model Idle"}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={startTraining}
            disabled={isTraining}
            className={cn(
              "flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200",
              isTraining
                ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                : "bg-green-500 hover:bg-green-600 text-white"
            )}
          >
            <Play className="w-4 h-4" />
            <span>Start Training</span>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={pauseTraining}
            disabled={!isTraining}
            className={cn(
              "flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200",
              !isTraining
                ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                : "bg-yellow-500 hover:bg-yellow-600 text-white"
            )}
          >
            <Pause className="w-4 h-4" />
            <span>Pause Training</span>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={stopTraining}
            disabled={!isTraining}
            className={cn(
              "flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200",
              !isTraining
                ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                : "bg-red-500 hover:bg-red-600 text-white"
            )}
          >
            <Square className="w-4 h-4" />
            <span>Stop Training</span>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={saveModel}
            disabled={!isTraining}
            className={cn(
              "flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200",
              !isTraining
                ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                : "bg-blue-500 hover:bg-blue-600 text-white"
            )}
          >
            <Save className="w-4 h-4" />
            <span>Save Model</span>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center space-x-2 px-6 py-3 rounded-lg bg-purple-500 hover:bg-purple-600 text-white font-medium transition-all duration-200"
          >
            <Download className="w-4 h-4" />
            <span>Export Data</span>
          </motion.button>
        </div>
      </motion.div>

      {/* Model Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Training Progress"
          value={metrics.progress}
          icon={Activity}
          color="blue"
          backendConnected={backendConnected}
        />
        <MetricCard
          title="Steps"
          value={metrics.steps}
          icon={TrendingUp}
          color="green"
          backendConnected={backendConnected}
        />
        <MetricCard
          title="Episodes"
          value={metrics.episodes}
          icon={Clock}
          color="purple"
          backendConnected={backendConnected}
        />
        <MetricCard
          title="Current Reward"
          value={metrics.currentReward}
          icon={Target}
          color="yellow"
          backendConnected={backendConnected}
        />
        <MetricCard
          title="Loss"
          value={metrics.loss}
          icon={AlertTriangle}
          color="red"
          backendConnected={backendConnected}
        />
        <MetricCard
          title="Accuracy"
          value={metrics.accuracy}
          icon={CheckCircle}
          color="green"
          backendConnected={backendConnected}
        />
        {metrics.gpuUsage !== null && (
          <MetricCard
            title="GPU Usage"
            value={metrics.gpuUsage}
            icon={Cpu}
            color="yellow"
            backendConnected={backendConnected}
          />
        )}
        <MetricCard
          title="Memory Usage"
          value={metrics.memoryUsage}
          icon={HardDrive}
          color="purple"
          backendConnected={backendConnected}
        />
      </div>

      {/* Enhanced Training Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <EnhancedTrainingChart 
          data={metrics.history} 
          backendConnected={backendConnected} 
        />
      </motion.div>

      {/* Model Metrics Visualization */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <ModelMetricsVisualization metrics={metrics} />
      </motion.div>
    </div>
  );

  const TrainingTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">Model Training Status</h2>
      
      {/* Training Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Model Training Status</h3>
          <div className="flex items-center space-x-2">
            <div className={cn(
              "w-3 h-3 rounded-full",
              isTraining ? "bg-green-400 animate-pulse" : "bg-gray-500"
            )}></div>
            <span className="text-sm text-gray-400">
              {isTraining ? "Model Training Active" : "Model Training Inactive"}
            </span>
          </div>
        </div>

        {!backendConnected ? (
          <div className="space-y-4">
            <div className="text-center text-gray-400">
              <p>Backend not connected</p>
              <p className="text-sm">Start the backend server to see training data</p>
            </div>
          </div>
        ) : isTraining && (
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Progress</span>
              <span className="text-white">{metrics.progress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${metrics.progress}%` }}
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
              />
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Steps</span>
                <div className="text-white font-medium">{metrics.steps.toLocaleString()}</div>
              </div>
              <div>
                <span className="text-gray-400">Episodes</span>
                <div className="text-white font-medium">{metrics.episodes.toLocaleString()}</div>
              </div>
              <div>
                <span className="text-gray-400">Reward</span>
                <div className="text-white font-medium">{metrics.currentReward.toFixed(3)}</div>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* Model Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">Model Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(config).map(([key, value]) => (
            <div key={key} className="space-y-2">
              <label className="text-sm text-gray-400 capitalize">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </label>
              <input
                type="text"
                value={value}
                onChange={(e) => setConfig(prev => ({ ...prev, [key]: e.target.value }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );

  const MetricsTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">Enhanced Model Metrics</h2>
      
      {/* Futuristic Metrics Dashboard */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-8"
      >
        <div className="text-center mb-8">
          <h3 className="text-xl font-semibold text-white mb-2">Model Performance Analytics</h3>
          <p className="text-gray-400">Real-time model training metrics and system monitoring</p>
        </div>

        {/* Main Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-blue-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <TrendingUp className="w-5 h-5 text-blue-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {backendConnected ? (metrics.learningRate || 0).toFixed(6) : 'No Data'}
                </div>
                <div className="text-sm text-gray-400">Learning Rate</div>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((metrics.learningRate || 0) * 1000000, 100)}%` }}
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <Code className="w-5 h-5 text-green-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {backendConnected ? (metrics.modelVersion || 'v1.0') : 'No Data'}
                </div>
                <div className="text-sm text-gray-400">Model Version</div>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full"
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-br from-yellow-500/20 to-orange-500/20 border border-yellow-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-yellow-500/20 rounded-lg">
                <Clock className="w-5 h-5 text-yellow-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {backendConnected ? (metrics.trainingDuration || '0h 0m') : 'No Data'}
                </div>
                <div className="text-sm text-gray-400">Training Duration</div>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "75%" }}
                className="bg-gradient-to-r from-yellow-500 to-orange-500 h-2 rounded-full"
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-500/30 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-purple-500/20 rounded-lg">
                <Cpu className="w-5 h-5 text-purple-400" />
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">
                  {backendConnected ? `${metrics.cpuUsage || 0}%` : 'No Data'}
                </div>
                <div className="text-sm text-gray-400">CPU Usage</div>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${metrics.cpuUsage || 0}%` }}
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
              />
            </div>
          </motion.div>
        </div>

        {/* System Resources Chart */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {metrics.gpuUsage !== null && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-gray-400">GPU Usage</span>
                <span className="text-sm text-white">{backendConnected ? `${metrics.gpuUsage || 0}%` : 'No Data'}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${metrics.gpuUsage || 0}%` }}
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full"
                />
              </div>
            </motion.div>
          )}

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-gray-400">Memory Usage</span>
              <span className="text-sm text-white">{backendConnected ? `${metrics.memoryUsage || 0}%` : 'No Data'}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${metrics.memoryUsage || 0}%` }}
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full"
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
            className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-gray-400">Training Progress</span>
              <span className="text-sm text-white">{backendConnected ? `${metrics.progress || 0}%` : 'No Data'}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${metrics.progress || 0}%` }}
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full"
              />
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );

  const ConfigTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">Model Configuration</h2>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">Model Training Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400">Symbols</label>
              <input
                type="text"
                value={config.symbols.join(', ')}
                onChange={(e) => setConfig(prev => ({ ...prev, symbols: e.target.value.split(', ') }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">Timeframes</label>
              <input
                type="text"
                value={config.timeframes.join(', ')}
                onChange={(e) => setConfig(prev => ({ ...prev, timeframes: e.target.value.split(', ').map(Number) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">Window Size</label>
              <input
                type="number"
                value={config.windowSize}
                onChange={(e) => setConfig(prev => ({ ...prev, windowSize: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">Max Leverage</label>
              <input
                type="number"
                step="0.1"
                value={config.maxLeverage}
                onChange={(e) => setConfig(prev => ({ ...prev, maxLeverage: parseFloat(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                value={config.learningRate}
                onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">Batch Size</label>
              <input
                type="number"
                value={config.batchSize}
                onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">N Steps</label>
              <input
                type="number"
                value={config.nSteps}
                onChange={(e) => setConfig(prev => ({ ...prev, nSteps: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
            <div>
              <label className="text-sm text-gray-400">N Epochs</label>
              <input
                type="number"
                value={config.nEpochs}
                onChange={(e) => setConfig(prev => ({ ...prev, nEpochs: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-blue-500/50"
              />
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-4 mt-6">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center space-x-2 px-6 py-3 rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-medium transition-all duration-200"
          >
            <Save className="w-4 h-4" />
            <span>Save Configuration</span>
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center space-x-2 px-6 py-3 rounded-lg bg-gray-600 hover:bg-gray-700 text-white font-medium transition-all duration-200"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset to Default</span>
          </motion.button>
        </div>
      </motion.div>
    </div>
  );

  const HistoryTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">Model Training History</h2>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl border border-gray-700/50 rounded-xl p-6"
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700/50">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Model Version</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Date</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Duration</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Final Reward</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Steps</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Actions</th>
              </tr>
            </thead>
            <tbody>
              {trainingHistory.length > 0 ? (
                trainingHistory.map((item) => (
                  <tr key={item.id} className="border-b border-gray-700/30 hover:bg-gray-800/30">
                    <td className="py-3 px-4 text-white font-medium">{item.name}</td>
                    <td className="py-3 px-4 text-gray-300">{item.date}</td>
                    <td className="py-3 px-4 text-gray-300">{item.duration}</td>
                    <td className="py-3 px-4 text-gray-300">{item.finalReward ? item.finalReward.toFixed(3) : '--'}</td>
                    <td className="py-3 px-4 text-gray-300">{item.steps ? item.steps.toLocaleString() : '--'}</td>
                    <td className="py-3 px-4">
                      <span className={cn(
                        "px-2 py-1 rounded-full text-xs font-medium",
                        item.status === 'completed' ? "bg-green-500/20 text-green-400" : "bg-yellow-500/20 text-yellow-400"
                      )}>
                        {item.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        <button className="p-1 rounded hover:bg-gray-700/50 transition-colors">
                          <Eye className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-1 rounded hover:bg-gray-700/50 transition-colors">
                          <Download className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-1 rounded hover:bg-gray-700/50 transition-colors">
                          <RotateCcw className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="7" className="py-8 px-4 text-center text-gray-400">
                    No training history available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );

  const SystemTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">System Monitor</h2>
      
      <SystemResourceMonitor metrics={metrics} />
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab />;
      case 'training':
        return <TrainingTab />;
      case 'metrics':
        return <MetricsTab />;
      case 'config':
        return <ConfigTab />;
      case 'history':
        return <HistoryTab />;
      case 'system':
        return <SystemTab />;
      default:
        return <OverviewTab />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && <Sidebar />}
      </AnimatePresence>

      {/* Main Content */}
      <div className={cn(
        "transition-all duration-300",
        sidebarOpen ? "ml-80" : "ml-0"
      )}>
        {/* Top Bar */}
        <div className="bg-gradient-to-r from-gray-900/95 to-gray-800/95 backdrop-blur-xl border-b border-gray-700/50 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {!sidebarOpen && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setSidebarOpen(true)}
                  className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
                >
                  <ChevronRight className="w-5 h-5 text-gray-400" />
                </motion.button>
              )}
              <h1 className="text-xl font-bold text-white">Ryan-RL Trading Dashboard</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live</span>
              </div>
              <div className="text-sm text-gray-400">
                {metrics.currentTime}
              </div>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderTabContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
