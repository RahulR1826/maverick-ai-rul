import { motion, useAnimation } from 'framer-motion';
import { BrainCircuit, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useStore } from '../store/useStore';
import { useEffect, useState } from 'react';
import { cn } from '../utils';

const XAI_DATA = [
  { sensor: 'HPC Efficiency', importance: 22.4 },
  { sensor: 'Fuel Flow', importance: 18.1 },
  { sensor: 'Burner F/A Ratio', importance: 14.6 },
  { sensor: 'HPT Coolant', importance: 12.3 },
  { sensor: 'LPC Temp (T24)', importance: 10.8 },
  { sensor: 'HPC Temp (T30)', importance: 9.2 },
  { sensor: 'Press. Ratio', importance: 7.9 },
  { sensor: 'Bypass Ratio', importance: 4.7 },
];

const COLORS = ['#FF0055','#FF4444','#FF8C00','#FFB300','#00E676','#00DDCC','#00CCDD','#00BFFF'];

// Custom Bar Shape with Glowing Tip for Recharts
const GlowingBar = (props: any) => {
  const { fill, x, y, width, height } = props;
  return (
    <g>
      {/* Main Bar Body - slightly transparent */}
      <rect x={x} y={y + 4} width={width} height={height - 8} fill={fill} opacity={0.3} rx={2} />
      {/* Glowing Tip */}
      <rect x={x + width - 4} y={y + 2} width={4} height={height - 4} fill={fill} rx={2} style={{ filter: `drop-shadow(0 0 5px ${fill})` }} />
    </g>
  );
};

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.length) {
    const color = payload[0].payload.fill || payload[0].color;
    return (
      <div className="glass-panel rounded px-4 py-3 text-xs text-white shadow-[0_0_15px_rgba(0,191,255,0.15)] font-mono">
        <p className="font-bold tracking-widest uppercase mb-1" style={{ color }}>{payload[0].payload.sensor}</p>
        <p>IMPACT SCORE: <strong className="text-white drop-shadow-md">{payload[0].value}%</strong></p>
      </div>
    );
  }
  return null;
};

// Typewriter Effect Component
const Typewriter = ({ text, delay = 0 }: { text: string, delay?: number }) => {
  const [displayedText, setDisplayedText] = useState('');
  
  useEffect(() => {
    let i = 0;
    setDisplayedText('');
    const startDelay = setTimeout(() => {
      const interval = setInterval(() => {
        setDisplayedText(text.slice(0, i));
        i++;
        if (i > text.length) clearInterval(interval);
      }, 20); // typing speed
      return () => clearInterval(interval);
    }, delay * 1000);
    return () => clearTimeout(startDelay);
  }, [text, delay]);

  return <span className="font-mono">{displayedText}<span className="animate-pulse bg-primary/70 w-1.5 h-3.5 inline-block ml-1 align-middle" /></span>;
};

export function ExplainableAI() {
  const { fleet, selectedAircraftId } = useStore();
  const aircraft = fleet.find(a => a.id === selectedAircraftId) || fleet[1];

  const top3 = XAI_DATA.slice(0, 3).map(d => d.sensor);
  const confidenceScore = 94.2;
  const isCritical = aircraft.risk === 'Critical' || aircraft.risk === 'High';

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      <div className="relative z-10">
        <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
          <BrainCircuit className="w-8 h-8 mr-3 text-secondary" /> Explainable AI (XAI)
        </h2>
        <p className="text-muted text-sm font-mono uppercase tracking-widest">SHAP Feature Attribution • Neural Network Logic Trace</p>
      </div>

      {/* Prediction context HUD Strip */}
      <div className="glass-panel p-5 rounded-xl border-l-4 border-l-primary relative z-10 shadow-[0_0_20px_rgba(0,191,255,0.05)]">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[10px] text-primary font-bold uppercase tracking-widest mb-1 drop-shadow-md">Diagnostic Target</p>
            <h3 className="text-2xl font-mono font-bold text-white tracking-wider">{aircraft.id} <span className="text-muted mx-2">|</span> RUL: <span className={cn("drop-shadow-md", aircraft.rul < 50 ? 'text-[#FFB300]' : 'text-primary')}>{aircraft.rul} CYC</span></h3>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-muted font-bold uppercase tracking-widest mb-1">Model Confidence</p>
            <p className="text-2xl font-mono font-bold text-[#00E676] drop-shadow-[0_0_8px_#00E676]">{confidenceScore}%</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
        
        {/* SHAP Bar Chart */}
        <div className="glass-panel rounded-xl p-6 shadow-lg">
          <h3 className="text-[10px] font-bold text-white mb-6 flex items-center uppercase tracking-widest">
            <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" /> Signal Attribution Meter
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={XAI_DATA} layout="vertical" margin={{ left: 120, right: 20, top: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(0,191,255,0.1)" horizontal={false} />
              <XAxis type="number" domain={[0, 25]} tick={{ fill: '#6B8CAE', fontSize: 10, fontFamily: 'monospace' }} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="sensor" tick={{ fill: '#E8F4FF', fontSize: 10, fontFamily: 'monospace', fontWeight: 'bold' }} width={115} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: 'rgba(0,191,255,0.05)' }} content={<CustomTooltip />} />
              <Bar dataKey="importance" shape={<GlowingBar />}>
                {XAI_DATA.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Diagnostic Outputs */}
        <div className="space-y-6">
          
          {/* Top 3 Contributing Sensors - Glass Cards */}
          <div className="grid grid-cols-3 gap-3">
            {top3.map((name, i) => (
              <motion.div key={name} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.15 }}
                className="glass-panel rounded-lg p-4 relative overflow-hidden group">
                <div className="absolute top-0 left-0 w-full h-[2px]" style={{ backgroundColor: COLORS[i], boxShadow: `0 0 10px ${COLORS[i]}` }} />
                <div className="text-[24px] font-mono font-bold opacity-20 absolute -right-1 -bottom-2" style={{ color: COLORS[i] }}>0{i+1}</div>
                <div className="text-[10px] text-muted uppercase tracking-widest mb-2 font-bold">{name}</div>
                <div className="text-xl font-mono font-bold text-white tabular-nums drop-shadow-md">
                  {XAI_DATA[i].importance}%
                </div>
              </motion.div>
            ))}
          </div>

          {/* Typewriter Diagnostic Terminal */}
          <div className="bg-[#02050A] border border-primary/30 rounded-xl p-6 shadow-[inset_0_0_30px_rgba(0,191,255,0.05)] relative overflow-hidden">
            {/* Scanline overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,191,255,0.05)_1px,transparent_1px)] bg-[size:100%_4px] pointer-events-none" />
            
            <h3 className="text-[10px] font-bold text-primary mb-4 flex items-center uppercase tracking-widest drop-shadow-[0_0_5px_#00BFFF]">
              <BrainCircuit className="w-3.5 h-3.5 mr-2" /> System Diagnostic Readout
            </h3>
            
            <div className="text-[11px] leading-relaxed text-gray-300 font-mono space-y-4 relative z-10">
              <p>
                <span className="text-primary mr-2">&gt;</span>
                <Typewriter text={`INITIALIZING EXPLANATION FOR TARGET: ${aircraft.id}`} delay={0} />
              </p>
              
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}>
                <span className="text-primary mr-2">&gt;</span>
                <Typewriter text={`PRIMARY DEGRADATION DRIVER: [${top3[0].toUpperCase()}]`} delay={1.5} />
              </motion.p>
              
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 3 }}>
                <span className={cn("mr-2", isCritical ? 'text-[#FF0055]' : 'text-[#00E676]')}>&gt;</span>
                {isCritical ? (
                  <span className="text-[#FF0055] drop-shadow-[0_0_5px_#FF0055]">
                    <Typewriter text="CRITICAL ALERT: Degradation pattern detected. Efficiency has dropped significantly below baseline. Immediate borescope inspection recommended to prevent catastrophic failure." delay={3} />
                  </span>
                ) : (
                  <span className="text-[#00E676] drop-shadow-[0_0_5px_#00E676]">
                    <Typewriter text="NOMINAL: Engine health within standard operating range. Continue standard monitoring cadence. No immediate action required." delay={3} />
                  </span>
                )}
              </motion.div>
              
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 5.5 }} className="text-muted pt-2 border-t border-white/10">
                <span className="mr-2">~</span>
                <Typewriter text="FEDERATED MODEL VALIDATION: Trained on >1,200 engine cycles across 5 isolated airline fleets. Confidence matrix: STABLE." delay={5.5} />
              </motion.p>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
