import { motion } from 'framer-motion';
import { Server, Cpu, Clock, HardDrive, WifiOff, Shield } from 'lucide-react';
import { useStore } from '../store/useStore';
import { useEffect, useState } from 'react';
import { cn } from '../utils';

// --- Ping Animation Hook for Live Edge Inference Feel ---
function usePing(isActive: boolean) {
  const [ping, setPing] = useState(false);
  const [count, setCount] = useState(Math.floor(Math.random() * 5000 + 1000));

  useEffect(() => {
    if (!isActive) return;
    // Random interval between 500ms and 3000ms to simulate live predictions arriving
    const triggerPing = () => {
      setPing(true);
      setCount(c => c + 1);
      setTimeout(() => setPing(false), 300);
      setTimeout(triggerPing, Math.random() * 2500 + 500);
    };
    const initTimer = setTimeout(triggerPing, Math.random() * 1000);
    return () => clearTimeout(initTimer);
  }, [isActive]);

  return { ping, count };
}

// --- Node Card Component ---
const NodeCard = ({ aircraft }: { aircraft: any }) => {
  const isActive = aircraft.edgeStatus === 'Active';
  const { ping, count } = usePing(isActive);
  
  const statusColor = isActive ? 'bg-[#00E676] text-[#00E676] shadow-[0_0_10px_#00E676]' : 'bg-[#FFB300] text-[#FFB300] shadow-[0_0_10px_#FFB300]';
  const borderColor = isActive ? 'border-[#00E676]/30' : 'border-[#FFB300]/30';

  return (
    <div className={cn("relative glass-panel rounded-xl p-5 overflow-hidden transition-colors", borderColor)}>
      {/* Node Ping Overlay */}
      <motion.div
        initial={false}
        animate={{ opacity: ping ? 0.15 : 0 }}
        className="absolute inset-0 bg-white pointer-events-none"
        transition={{ duration: 0.3 }}
      />
      
      {/* Top Edge Indicator */}
      <div className={cn("absolute top-0 left-0 right-0 h-[2px]", isActive ? 'bg-[#00E676]' : 'bg-[#FFB300]')} />

      <div className="flex justify-between items-start mb-4">
        <div>
          <span className="font-mono text-white font-bold text-lg drop-shadow-md">{aircraft.id} <span className="text-muted text-xs font-sans font-normal ml-1">NODE</span></span>
          <div className="text-[9px] font-bold text-muted uppercase tracking-widest mt-1">{aircraft.type}</div>
        </div>
        <div className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest font-bold">
          <span className={cn("w-2 h-2 rounded-full", statusColor, isActive ? 'animate-pulse' : '')} />
          <span className={cn(isActive ? 'text-[#00E676]' : 'text-[#FFB300]')}>{aircraft.edgeStatus}</span>
        </div>
      </div>
      
      <div className="space-y-3">
        <div className="flex justify-between items-center border-b border-primary/10 pb-2">
          <span className="text-[10px] text-muted font-bold uppercase tracking-widest">Model</span>
          <span className="text-primary font-mono text-sm">{aircraft.modelVersion}</span>
        </div>
        <div className="flex justify-between items-center border-b border-primary/10 pb-2">
          <span className="text-[10px] text-muted font-bold uppercase tracking-widest">Latency</span>
          <span className="text-[#00E676] font-mono text-sm">{(Math.random() * 2 + 1).toFixed(2)} ms</span>
        </div>
        <div className="flex justify-between items-center pt-1">
          <span className="text-[10px] text-muted font-bold uppercase tracking-widest">Predictions</span>
          <span className={cn("font-mono text-lg font-bold tabular-nums drop-shadow-md", ping ? 'text-white' : 'text-gray-300')}>
            {count.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
};

export function EdgeAI() {
  const fleet = useStore(state => state.fleet);
  const activeNodes = fleet.filter(a => a.edgeStatus === 'Active').length;

  const FEATURES = [
    { title: 'Cloud Independent', desc: 'Models run entirely on local hardware, ensuring uninterrupted predictions even without internet connectivity.', icon: WifiOff, color: 'text-[#00BFFF]', bg: 'bg-[#00BFFF]/20', border: 'border-[#00BFFF]/30', glow: 'shadow-[0_0_15px_rgba(0,191,255,0.3)]' },
    { title: 'Instant Decisions', desc: 'Sub-millisecond inference latency allows for immediate fault detection and safety protocol engagement.', icon: Cpu, color: 'text-[#00E676]', bg: 'bg-[#00E676]/20', border: 'border-[#00E676]/30', glow: 'shadow-[0_0_15px_rgba(0,230,118,0.3)]' },
    { title: 'Data Privacy', desc: 'Telemetry data is processed locally. Only encrypted model weight updates are sent to the global federated server.', icon: Shield, color: 'text-[#FFB300]', bg: 'bg-[#FFB300]/20', border: 'border-[#FFB300]/30', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]' },
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      <div className="relative z-10">
        <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
          <Server className="w-8 h-8 mr-3 text-secondary" /> Edge AI Infrastructure
        </h2>
        <p className="text-muted text-sm font-mono uppercase tracking-widest">Distributed Local Inference • Zero Raw Data Exfiltration</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
        {FEATURES.map((feat, i) => (
          <motion.div 
            key={feat.title}
            initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.1 }}
            className="glass-panel p-6 rounded-xl hover:border-primary/40 transition-all group relative overflow-hidden"
          >
            <div className={`absolute top-0 left-0 w-1 h-full ${feat.bg} ${feat.glow}`} />
            <div className="absolute right-0 top-0 opacity-[0.03] transform translate-x-4 -translate-y-4 group-hover:scale-110 transition-transform duration-500">
              <feat.icon className="w-32 h-32 text-white" />
            </div>
            
            <div className={`relative w-12 h-12 rounded-full border ${feat.border} flex items-center justify-center mb-4 shrink-0`}>
              <div className={`absolute inset-0 rounded-full ${feat.bg} blur-md opacity-50 group-hover:opacity-100 transition-opacity duration-300`} />
              <feat.icon className={`relative z-10 w-5 h-5 ${feat.color}`} />
            </div>
            
            <h4 className={`font-bold uppercase tracking-widest text-xs mb-2 ${feat.color}`}>{feat.title}</h4>
            <p className="text-gray-400 text-sm leading-relaxed">{feat.desc}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 relative z-10">
        
        {/* Node Status Grid */}
        <div className="xl:col-span-3 glass-panel rounded-xl p-6 shadow-lg">
          <h3 className="text-sm font-bold text-white flex items-center mb-6 tracking-widest uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" /> Active Edge Nodes
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {fleet.map((aircraft) => (
              <NodeCard key={aircraft.id} aircraft={aircraft} />
            ))}
          </div>
        </div>

        {/* Global Metrics */}
        <div className="space-y-6">
          {[
            { label: 'Avg Inference Time', value: '1.4', unit: 'ms', icon: Cpu, color: 'text-[#00BFFF]', bg: 'bg-[#00BFFF]/20', border: 'border-[#00BFFF]/30', glow: 'shadow-[0_0_15px_rgba(0,191,255,0.3)]' },
            { label: 'Nodes Online', value: `${activeNodes}`, unit: `/ ${fleet.length}`, icon: Server, color: 'text-[#00E676]', bg: 'bg-[#00E676]/20', border: 'border-[#00E676]/30', glow: 'shadow-[0_0_15px_rgba(0,230,118,0.3)]' },
            { label: 'Total Predictions', value: '1.2M+', unit: '', icon: HardDrive, color: 'text-[#FFB300]', bg: 'bg-[#FFB300]/20', border: 'border-[#FFB300]/30', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]' },
          ].map((metric, i) => (
            <motion.div 
              key={metric.label}
              initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
              className="glass-panel rounded-xl p-6 text-center hover:border-primary/40 transition-all relative overflow-hidden group"
            >
              <div className={`absolute top-0 left-0 w-full h-1 ${metric.bg} ${metric.glow}`} />
              <div className="flex justify-center mb-3">
                <div className={`relative w-12 h-12 rounded-full border ${metric.border} flex items-center justify-center shrink-0`}>
                  <div className={`absolute inset-0 rounded-full ${metric.bg} blur-md opacity-50 group-hover:opacity-100 transition-opacity duration-300`} />
                  <metric.icon className={`relative z-10 w-5 h-5 ${metric.color}`} />
                </div>
              </div>
              <h4 className="text-[10px] text-muted uppercase font-bold tracking-widest mb-1">{metric.label}</h4>
              <div className={`text-4xl font-mono font-bold drop-shadow-md tabular-nums ${metric.color}`}>
                {metric.value} <span className="text-sm font-sans font-normal text-muted ml-1">{metric.unit}</span>
              </div>
            </motion.div>
          ))}
        </div>

      </div>
    </div>
  );
}
