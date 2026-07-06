import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';
import { Shield, Plane, Server, Network, Activity, Cpu } from 'lucide-react';
import { cn } from '../utils';
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Custom hook for value pulse
function usePulse(value: string | number) {
  const [pulse, setPulse] = useState(false);
  const prev = useRef(value);
  useEffect(() => {
    if (value !== prev.current) {
      setPulse(true);
      const timer = setTimeout(() => setPulse(false), 300);
      prev.current = value;
      return () => clearTimeout(timer);
    }
  }, [value]);
  return pulse;
}

const AnimatedValue = ({ value, className }: { value: string | number, className?: string }) => {
  const pulse = usePulse(value);
  return (
    <div className="relative inline-block">
      <motion.div
        initial={false}
        animate={{ opacity: pulse ? 0.8 : 0 }}
        className="absolute inset-0 bg-white blur-md rounded"
        transition={{ duration: 0.3 }}
      />
      <span className={cn("relative z-10 tabular-nums font-mono", className)}>{value}</span>
    </div>
  );
};

const STATS = [
  { label: 'Fleet Health', value: '88.5%', icon: Activity, color: 'text-[#00E676]', glow: 'shadow-[0_0_15px_rgba(0,230,118,0.3)]', border: 'border-[#00E676]/30', bg: 'bg-[#00E676]/20' },
  { label: 'Aircraft Connected', value: '1,245', icon: Plane, color: 'text-[#00BFFF]', glow: 'shadow-[0_0_15px_rgba(0,191,255,0.3)]', border: 'border-[#00BFFF]/30', bg: 'bg-[#00BFFF]/20' },
  { label: 'Edge Nodes', value: '412', icon: Server, color: 'text-[#FFB300]', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]', border: 'border-[#FFB300]/30', bg: 'bg-[#FFB300]/20' },
  { label: 'Federated Clients', value: '14', icon: Network, color: 'text-[#FF0055]', glow: 'shadow-[0_0_15px_rgba(255,0,85,0.3)]', border: 'border-[#FF0055]/30', bg: 'bg-[#FF0055]/20' },
];

export function Dashboard() {
  const { fleet, selectAircraft } = useStore();
  const navigate = useNavigate();

  const handleSelect = (id: string) => {
    selectAircraft(id);
    navigate('/aircraft');
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      {/* Hero Section */}
      <div className="glass-panel p-10 relative overflow-hidden">
        {/* Holographic Grid/Scanline Texture */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,191,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(0,191,255,0.05)_1px,transparent_1px)] bg-[size:30px_30px] pointer-events-none opacity-50" />
        <div className="absolute inset-0 bg-gradient-to-r from-background via-transparent to-background pointer-events-none" />
        
        <div className="absolute right-0 top-0 w-96 h-96 bg-primary/20 blur-[100px] rounded-full pointer-events-none -translate-y-1/2 translate-x-1/3" />
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative z-10"
        >
          <div className="inline-flex items-center px-3 py-1.5 rounded bg-primary/10 text-primary text-[10px] font-bold tracking-widest uppercase border border-primary/30 shadow-[0_0_10px_rgba(0,191,255,0.2)] mb-4">
            <Shield className="w-3.5 h-3.5 mr-2" />
            Enterprise Aerospace Software
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white mb-4 drop-shadow-md">
            MAVERICK <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">AI</span>
          </h1>
          <p className="text-lg text-gray-400 max-w-2xl leading-relaxed">
            Privacy-Preserving Edge AI Platform for Aircraft Predictive Maintenance. 
            Local inference with global federated intelligence.
          </p>
        </motion.div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">
        {STATS.map((stat, i) => (
          <motion.div 
            key={stat.label}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1 }}
            className="glass-panel p-6 flex items-center hover:-translate-y-1 transition-all duration-300 group cursor-default hover:shadow-[0_0_20px_rgba(0,191,255,0.25)] hover:border-primary/40 relative overflow-hidden"
          >
            <div className={`absolute top-0 left-0 w-1 h-full ${stat.bg} ${stat.glow}`} />
            <div className={`relative w-14 h-14 rounded-full border ${stat.border} flex items-center justify-center mr-5 shrink-0`}>
              <div className={`absolute inset-0 rounded-full ${stat.bg} blur-md opacity-50 group-hover:opacity-100 transition-opacity duration-300`} />
              <stat.icon className={`relative z-10 w-6 h-6 ${stat.color}`} />
            </div>
            <div>
              <p className="text-[10px] font-bold text-muted uppercase tracking-widest">{stat.label}</p>
              <div className={`text-3xl font-mono font-bold mt-1 ${stat.color} drop-shadow-md`}>
                <AnimatedValue value={stat.value} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Live Fleet Preview */}
      <div className="glass-panel p-6 relative z-10">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-sm font-bold text-white flex items-center tracking-widest uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" /> Live Fleet Status
          </h2>
          <div className="flex items-center text-[10px] uppercase tracking-widest font-bold text-muted bg-black/40 px-3 py-1.5 rounded border border-white/5">
            <motion.span 
              animate={{ opacity: [1, 0.4, 1], scale: [1, 1.2, 1] }} 
              transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
              className="relative flex h-2 w-2 mr-2 rounded-full bg-[#00E676] shadow-[0_0_8px_#00E676]" 
            />
            Real-time Telemetry Active
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-primary/20 text-[10px] font-bold uppercase tracking-widest text-muted">
                <th className="pb-3">Aircraft</th>
                <th className="pb-3">Type</th>
                <th className="pb-3">Health Score</th>
                <th className="pb-3">RUL (Cycles)</th>
                <th className="pb-3">Risk Level</th>
                <th className="pb-3">Edge AI</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {fleet.map((aircraft) => (
                <tr key={aircraft.id} onClick={() => handleSelect(aircraft.id)} className="border-b border-primary/10 hover:bg-primary/5 transition-colors cursor-pointer group">
                  <td className="py-4 font-mono text-primary font-bold">{aircraft.id}</td>
                  <td className="py-4 text-gray-300 font-medium">{aircraft.type}</td>
                  <td className="py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-24 bg-gray-900 rounded-full h-[3px] relative overflow-hidden shadow-[inset_0_1px_3px_rgba(0,0,0,0.5)]">
                        <div 
                          className={cn("absolute top-0 left-0 h-full rounded-full shadow-[0_0_8px_currentColor]",
                            aircraft.health > 70 ? 'bg-[#00E676] text-[#00E676]' : 
                            aircraft.health > 40 ? 'bg-[#FFB300] text-[#FFB300]' : 
                            'bg-[#FF0055] text-[#FF0055]'
                          )}
                          style={{ width: `${aircraft.health}%` }} 
                        />
                      </div>
                      <span className="text-white">
                        <AnimatedValue value={`${aircraft.health}%`} />
                      </span>
                    </div>
                  </td>
                  <td className="py-4 text-gray-300">
                    <AnimatedValue value={aircraft.rul} className={cn(
                      aircraft.rul > 70 ? 'text-[#00BFFF]' : 
                      aircraft.rul > 30 ? 'text-[#FFB300]' : 
                      'text-[#FF0055]'
                    )} />
                  </td>
                  <td className="py-4">
                    <span className={cn("px-2.5 py-1 rounded border text-[10px] font-bold uppercase tracking-widest",
                      aircraft.risk === 'Critical' ? 'bg-[#FF0055]/10 text-[#FF0055] border-[#FF0055]/30 shadow-[0_0_10px_rgba(255,0,85,0.1)]' :
                      aircraft.risk === 'High' ? 'bg-[#FF4444]/10 text-[#FF4444] border-[#FF4444]/30 shadow-[0_0_10px_rgba(255,68,68,0.1)]' :
                      aircraft.risk === 'Medium' ? 'bg-[#FFB300]/10 text-[#FFB300] border-[#FFB300]/30 shadow-[0_0_10px_rgba(255,179,0,0.1)]' :
                      'bg-[#00E676]/10 text-[#00E676] border-[#00E676]/30 shadow-[0_0_10px_rgba(0,230,118,0.1)]'
                    )}>
                      {aircraft.risk}
                    </span>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center text-[10px] uppercase font-bold tracking-widest text-muted group-hover:text-gray-300 transition-colors">
                      <Cpu className={cn("w-3.5 h-3.5 mr-1.5", aircraft.edgeStatus === 'Active' ? 'text-[#00E676]' : 'text-primary')} />
                      {aircraft.edgeStatus}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
