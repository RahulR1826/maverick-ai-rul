import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Target, Activity, Zap, ShieldAlert, BarChart3 } from 'lucide-react';
import { cn } from '../utils';

const FLEET_TREND = Array.from({ length: 30 }, (_, i) => ({
  day: i + 1,
  health: 85 + Math.sin(i / 3) * 5 + Math.random() * 2,
}));

const FL_CONVERGENCE = Array.from({ length: 20 }, (_, i) => ({
  round: i + 1,
  federated_loss: 50 * Math.exp(-i / 4) + 15 + Math.random() * 2,
  centralized_loss: 50 * Math.exp(-i / 6) + 18 + Math.random() * 3,
}));

const RUL_DISTRIBUTION = [
  { range: '0-50', count: 12, fill: '#FF0055' },
  { range: '51-100', count: 45, fill: '#FFB300' },
  { range: '101-150', count: 85, fill: '#00E676' },
  { range: '151+', count: 32, fill: '#00BFFF' },
];

const RISK_PIE = [
  { name: 'Critical', value: 12, color: '#FF0055', dropShadow: 'drop-shadow(0 0 10px #FF0055)' },
  { name: 'High', value: 24, color: '#FF4444', dropShadow: 'drop-shadow(0 0 10px #FF4444)' },
  { name: 'Medium', value: 45, color: '#FFB300', dropShadow: 'drop-shadow(0 0 10px #FFB300)' },
  { name: 'Low', value: 85, color: '#00E676', dropShadow: 'drop-shadow(0 0 10px #00E676)' },
];

const STATS = [
  { label: 'Global RMSE', value: '14.52', icon: Target, color: 'text-[#00E676]', glow: 'shadow-[0_0_15px_rgba(0,230,118,0.3)]', border: 'border-[#00E676]/30', bg: 'bg-[#00E676]/20' },
  { label: 'Global MAE', value: '10.15', icon: Activity, color: 'text-[#00BFFF]', glow: 'shadow-[0_0_15px_rgba(0,191,255,0.3)]', border: 'border-[#00BFFF]/30', bg: 'bg-[#00BFFF]/20' },
  { label: 'R² Score', value: '0.87', icon: Zap, color: 'text-[#FFB300]', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]', border: 'border-[#FFB300]/30', bg: 'bg-[#FFB300]/20' },
  { label: 'FL Rounds', value: '20', icon: ShieldAlert, color: 'text-[#FF0055]', glow: 'shadow-[0_0_15px_rgba(255,0,85,0.3)]', border: 'border-[#FF0055]/30', bg: 'bg-[#FF0055]/20' },
];

const GlassTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-panel rounded p-3 text-[10px] font-mono text-white shadow-[0_0_15px_rgba(0,191,255,0.15)] uppercase tracking-widest">
        <p className="mb-2 text-primary font-bold">{label}</p>
        {payload.map((p: any, i: number) => (
          <p key={i} style={{ color: p.color }}>{p.name}: <span className="font-bold tabular-nums ml-2 text-white">{p.value.toFixed ? p.value.toFixed(2) : p.value}</span></p>
        ))}
      </div>
    );
  }
  return null;
};

// Custom Glowing Bar
const GlowingBar = (props: any) => {
  const { fill, x, y, width, height } = props;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} opacity={0.4} rx={2} />
      <rect x={x} y={y} width={width} height={4} fill={fill} rx={2} style={{ filter: `drop-shadow(0 0 8px ${fill})` }} />
    </g>
  );
};

export function Analytics() {
  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      <div className="relative z-10 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
            <BarChart3 className="w-8 h-8 mr-3 text-secondary" /> Data Analytics
          </h2>
          <p className="text-muted text-sm font-mono uppercase tracking-widest">Model Performance & Fleet Telemetry Insights</p>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 relative z-10">
        {STATS.map((stat, i) => (
          <motion.div key={stat.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
            className="glass-panel p-5 flex items-center rounded-xl relative overflow-hidden group hover:border-primary/40 transition-colors">
            <div className={`absolute top-0 left-0 w-1 h-full ${stat.bg} ${stat.glow}`} />
            <div className={`relative w-12 h-12 rounded-full border ${stat.border} flex items-center justify-center mr-4 shrink-0`}>
              <div className={`absolute inset-0 rounded-full ${stat.bg} blur-md opacity-50 group-hover:opacity-100 transition-opacity duration-300`} />
              <stat.icon className={`relative z-10 w-5 h-5 ${stat.color}`} />
            </div>
            <div>
              <p className="text-[9px] font-bold text-muted uppercase tracking-widest">{stat.label}</p>
              <div className={`text-2xl font-mono font-bold mt-0.5 tabular-nums drop-shadow-md ${stat.color}`}>{stat.value}</div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
        
        {/* Fleet Health Trend */}
        <div className="glass-panel rounded-xl p-6 shadow-lg">
          <h3 className="text-[10px] font-bold text-white uppercase tracking-widest mb-6 flex items-center">
            <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" /> Fleet Health Trend (30 Days)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={FLEET_TREND}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,191,255,0.05)" vertical={false} />
                <XAxis dataKey="day" stroke="rgba(255,255,255,0.2)" tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis domain={[75, 100]} stroke="rgba(255,255,255,0.2)" tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} />
                <Tooltip content={<GlassTooltip />} />
                <Line type="monotone" dataKey="health" stroke="#00BFFF" strokeWidth={3} dot={false} style={{ filter: 'drop-shadow(0 0 8px rgba(0,191,255,0.5))' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* FL Convergence */}
        <div className="glass-panel rounded-xl p-6 shadow-lg">
          <h3 className="text-[10px] font-bold text-white uppercase tracking-widest mb-6 flex items-center">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00E676] mr-2 shadow-[0_0_5px_#00E676]" /> FL Convergence vs Centralized
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={FL_CONVERGENCE}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,191,255,0.05)" vertical={false} />
                <XAxis dataKey="round" stroke="rgba(255,255,255,0.2)" tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis stroke="rgba(255,255,255,0.2)" tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} />
                <Tooltip content={<GlassTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace', textTransform: 'uppercase' }} />
                <Line type="monotone" name="Federated Loss" dataKey="federated_loss" stroke="#00E676" strokeWidth={2} dot={false} style={{ filter: 'drop-shadow(0 0 8px rgba(0,230,118,0.5))' }} />
                <Line type="monotone" name="Centralized Loss" dataKey="centralized_loss" stroke="#FF4444" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* RUL Distribution */}
        <div className="glass-panel rounded-xl p-6 shadow-lg">
          <h3 className="text-[10px] font-bold text-white uppercase tracking-widest mb-6 flex items-center">
            <span className="w-1.5 h-1.5 rounded-full bg-[#FFB300] mr-2 shadow-[0_0_5px_#FFB300]" /> RUL Distribution (Cycles)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={RUL_DISTRIBUTION} margin={{ top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,191,255,0.05)" vertical={false} />
                <XAxis dataKey="range" stroke="rgba(255,255,255,0.2)" tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis stroke="rgba(255,255,255,0.2)" tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} />
                <Tooltip cursor={{ fill: 'rgba(0,191,255,0.05)' }} content={<GlassTooltip />} />
                <Bar dataKey="count" shape={<GlowingBar />}>
                  {RUL_DISTRIBUTION.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk Pie */}
        <div className="glass-panel rounded-xl p-6 shadow-lg">
          <h3 className="text-[10px] font-bold text-white uppercase tracking-widest mb-2 flex items-center">
            <span className="w-1.5 h-1.5 rounded-full bg-[#FF0055] mr-2 shadow-[0_0_5px_#FF0055]" /> Fleet Risk Profile
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={RISK_PIE} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={5} dataKey="value" stroke="none">
                  {RISK_PIE.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} style={{ filter: entry.dropShadow }} opacity={0.8} />
                  ))}
                </Pie>
                <Tooltip content={<GlassTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace', textTransform: 'uppercase' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
