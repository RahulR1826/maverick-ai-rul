import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../store/useStore';
import { Plane, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { cn } from '../utils';
import { useState } from 'react';

const TYPES = ["Boeing 737-800", "Boeing 777-300", "Airbus A320neo", "Airbus A350-900", "Boeing 787-9"];

export function Fleet() {
  const { fleet, selectAircraft } = useStore();
  const navigate = useNavigate();
  const [lockingId, setLockingId] = useState<string | null>(null);

  const handleSelect = (id: string) => {
    setLockingId(id);
    setTimeout(() => {
      selectAircraft(id);
      navigate('/aircraft');
    }, 400); // Wait for targeting lock animation
  };

  const criticalCount = fleet.filter(a => a.risk === 'Critical' || a.risk === 'High').length;
  const activeCount = fleet.filter(a => a.edgeStatus === 'Active').length;
  const maintenanceCount = fleet.filter(a => a.maintenanceStatus !== 'OK').length;

  const STATS = [
    { label: 'Total Aircraft', value: fleet.length, color: 'text-white', glow: 'shadow-[0_0_15px_rgba(255,255,255,0.1)]', border: 'border-white/20' },
    { label: 'Critical / High Risk', value: criticalCount, color: 'text-[#FF0055]', glow: 'shadow-[0_0_15px_rgba(255,0,85,0.3)]', border: 'border-[#FF0055]/30' },
    { label: 'Edge AI Active', value: activeCount, color: 'text-[#00E676]', glow: 'shadow-[0_0_15px_rgba(0,230,118,0.3)]', border: 'border-[#00E676]/30' },
    { label: 'Maintenance Due', value: maintenanceCount, color: 'text-[#FFB300]', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]', border: 'border-[#FFB300]/30' },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      <div className="relative z-10">
        <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
          <Plane className="w-8 h-8 mr-3 text-primary" /> Live Fleet Monitor
        </h2>
        <p className="text-muted text-sm font-mono uppercase tracking-widest">Select target to initialize Digital Twin diagnostics scan</p>
      </div>

      {/* Summary bar */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
        {STATS.map((s, i) => (
          <motion.div key={s.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}
            className="glass-panel p-5 text-center hover:border-primary/40 transition-all cursor-default relative overflow-hidden group hover:shadow-[0_0_25px_rgba(0,191,255,0.2)]">
            <div className={`absolute top-0 left-0 w-1 h-full ${s.border.replace('border-', 'bg-')} ${s.glow}`} />
            <div className={`text-3xl font-mono font-bold tabular-nums drop-shadow-md ${s.color}`}>{s.value}</div>
            <div className="text-[10px] font-bold text-muted uppercase tracking-widest mt-1">{s.label}</div>
          </motion.div>
        ))}
      </div>

      {/* Fleet table */}
      <div className="glass-panel p-6 overflow-x-auto relative z-10">
        <table className="w-full text-left border-collapse text-sm">
          <thead>
            <tr className="border-b border-primary/20 text-[10px] font-bold uppercase tracking-widest text-muted">
              <th className="pb-3">Aircraft ID</th>
              <th className="pb-3">Type</th>
              <th className="pb-3">Health Score</th>
              <th className="pb-3">RUL (Cycles)</th>
              <th className="pb-3">Risk</th>
              <th className="pb-3">Flight Phase</th>
              <th className="pb-3">Edge AI</th>
              <th className="pb-3">Maintenance</th>
              <th className="pb-3"></th>
            </tr>
          </thead>
          <tbody>
            {fleet.map((a, i) => {
              const isLocking = lockingId === a.id;
              
              return (
                <motion.tr
                  key={a.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => handleSelect(a.id)}
                  className={cn("border-b border-primary/10 transition-all cursor-pointer group relative", 
                    isLocking 
                      ? "bg-primary/20 shadow-[inset_0_0_30px_rgba(0,191,255,0.5)] z-20" 
                      : "hover:bg-primary/5 hover:shadow-[inset_0_0_15px_rgba(0,191,255,0.15)]"
                  )}
                >
                  {/* Targeting Lock Overlay Animation */}
                  <AnimatePresence>
                    {isLocking && (
                      <motion.td 
                        initial={{ opacity: 0, scaleY: 0 }}
                        animate={{ opacity: 1, scaleY: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 border-y border-primary pointer-events-none flex items-center justify-center bg-primary/10 backdrop-blur-[2px]"
                      >
                        <motion.div 
                          initial={{ scale: 2, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ type: "spring", stiffness: 300, damping: 15 }}
                          className="w-full h-full flex items-center px-4"
                        >
                          <div className="w-4 h-4 border-t-2 border-l-2 border-primary absolute top-2 left-2" />
                          <div className="w-4 h-4 border-b-2 border-l-2 border-primary absolute bottom-2 left-2" />
                          <div className="w-4 h-4 border-t-2 border-r-2 border-primary absolute top-2 right-2" />
                          <div className="w-4 h-4 border-b-2 border-r-2 border-primary absolute bottom-2 right-2" />
                          <span className="text-primary font-mono font-bold tracking-[0.3em] mx-auto animate-pulse">TARGET LOCKED // INITIALIZING SCAN</span>
                        </motion.div>
                      </motion.td>
                    )}
                  </AnimatePresence>

                  <td className={cn("py-4 font-mono font-bold text-primary", isLocking ? "opacity-0" : "")}>{a.id}</td>
                  <td className={cn("py-4 text-gray-300 font-medium", isLocking ? "opacity-0" : "")}>{TYPES[i % TYPES.length]}</td>
                  <td className={cn("py-4", isLocking ? "opacity-0" : "")}>
                    <div className="flex items-center gap-3">
                      <div className="w-20 bg-gray-900 rounded-full h-[3px] relative overflow-hidden shadow-[inset_0_1px_3px_rgba(0,0,0,0.5)]">
                        <div 
                          className={cn("absolute top-0 left-0 h-full rounded-full shadow-[0_0_8px_currentColor]",
                            a.health > 70 ? 'bg-[#00E676] text-[#00E676]' : 
                            a.health > 40 ? 'bg-[#FFB300] text-[#FFB300]' : 
                            'bg-[#FF0055] text-[#FF0055]'
                          )}
                          style={{ width: `${a.health}%` }} 
                        />
                      </div>
                      <span className="text-white font-mono font-bold tabular-nums">{a.health}%</span>
                    </div>
                  </td>
                  <td className={cn("py-4 font-mono font-bold tabular-nums", isLocking ? "opacity-0" : "text-white")}>{a.rul} cyc</td>
                  <td className={cn("py-4", isLocking ? "opacity-0" : "")}>
                    <span className={cn("px-2.5 py-1 rounded border text-[10px] font-bold uppercase tracking-widest",
                      a.risk === 'Critical' ? 'bg-[#FF0055]/10 text-[#FF0055] border-[#FF0055]/30 shadow-[0_0_10px_rgba(255,0,85,0.1)]' :
                      a.risk === 'High' ? 'bg-[#FF4444]/10 text-[#FF4444] border-[#FF4444]/30 shadow-[0_0_10px_rgba(255,68,68,0.1)]' :
                      a.risk === 'Medium' ? 'bg-[#FFB300]/10 text-[#FFB300] border-[#FFB300]/30 shadow-[0_0_10px_rgba(255,179,0,0.1)]' :
                      'bg-[#00E676]/10 text-[#00E676] border-[#00E676]/30 shadow-[0_0_10px_rgba(0,230,118,0.1)]'
                    )}>
                      {a.risk}
                    </span>
                  </td>
                  <td className={cn("py-4 text-gray-400 font-medium", isLocking ? "opacity-0" : "")}>{a.phase}</td>
                  <td className={cn("py-4", isLocking ? "opacity-0" : "")}>
                    <div className="flex items-center gap-2 text-[10px] uppercase font-bold tracking-widest text-muted">
                      {a.edgeStatus === 'Syncing' ? (
                        <motion.span 
                          animate={{ opacity: [1, 0.4, 1], scale: [1, 1.2, 1] }} 
                          transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
                          className="w-2 h-2 rounded-full bg-[#FFB300] shadow-[0_0_8px_#FFB300]"
                        />
                      ) : a.edgeStatus === 'Active' ? (
                        <span className="w-2 h-2 rounded-full bg-[#00E676] shadow-[0_0_8px_#00E676]" />
                      ) : (
                        <span className="w-2 h-2 rounded-full bg-[#FF0055]" />
                      )}
                      {a.edgeStatus}
                    </div>
                  </td>
                  <td className={cn("py-4", isLocking ? "opacity-0" : "")}>
                    <span className={cn("text-[10px] font-bold uppercase tracking-widest",
                      a.maintenanceStatus === 'Overdue' ? 'text-[#FF0055] drop-shadow-[0_0_5px_#FF0055]' :
                      a.maintenanceStatus === 'Scheduled' ? 'text-[#FFB300]' : 'text-[#00E676]')}>
                      {a.maintenanceStatus === 'OK' ? '✓ OK' : a.maintenanceStatus === 'Scheduled' ? '⚠ SCHED' : '🔴 OVERDUE'}
                    </span>
                  </td>
                  <td className={cn("py-4", isLocking ? "opacity-0" : "")}>
                    <ChevronRight className="w-4 h-4 text-muted group-hover:text-primary transition-colors group-hover:translate-x-1 duration-300" />
                  </td>
                </motion.tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
