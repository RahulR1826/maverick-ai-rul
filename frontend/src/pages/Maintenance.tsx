import { motion } from 'framer-motion';
import { Wrench, Calendar, DollarSign, UserCog, Settings } from 'lucide-react';
import { useStore } from '../store/useStore';
import { useState } from 'react';
import { cn } from '../utils';

export function Maintenance() {
  const fleet = useStore(state => state.fleet);
  const [activeTab, setActiveTab] = useState(fleet[0]?.id);
  const aircraft = fleet.find(a => a.id === activeTab) || fleet[0];

  const getPriorityInfo = (risk: string) => {
    switch(risk) {
      case 'Critical': return { level: 'URGENT', color: 'text-[#FF0055]', bg: 'bg-[#FF0055]', glow: 'shadow-[0_0_15px_rgba(255,0,85,0.4)]', border: 'border-[#FF0055]/30' };
      case 'High': return { level: 'HIGH', color: 'text-[#FF4444]', bg: 'bg-[#FF4444]', glow: 'shadow-[0_0_15px_rgba(255,68,68,0.3)]', border: 'border-[#FF4444]/30' };
      case 'Medium': return { level: 'MEDIUM', color: 'text-[#FFB300]', bg: 'bg-[#FFB300]', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]', border: 'border-[#FFB300]/30' };
      default: return { level: 'LOW', color: 'text-[#00E676]', bg: 'bg-[#00E676]', glow: 'shadow-[0_0_15px_rgba(0,230,118,0.3)]', border: 'border-[#00E676]/30' };
    }
  };

  const priority = getPriorityInfo(aircraft.risk);

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      <div className="relative z-10 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
            <Wrench className="w-8 h-8 mr-3 text-secondary" /> Maintenance Control
          </h2>
          <p className="text-muted text-sm font-mono uppercase tracking-widest">AI-Generated Service Directives</p>
        </div>
      </div>

      {/* Glowing Tab Selector */}
      <div className="flex space-x-2 overflow-x-auto pb-2 relative z-10 scrollbar-hide">
        {fleet.slice(0, 8).map(a => {
          const isActive = a.id === activeTab;
          const p = getPriorityInfo(a.risk);
          return (
            <button
              key={a.id}
              onClick={() => setActiveTab(a.id)}
              className={cn("px-4 py-2 text-[11px] font-bold uppercase tracking-widest rounded transition-all whitespace-nowrap relative border",
                isActive ? `bg-[#050B14] ${p.border} text-white` : `bg-transparent border-transparent text-muted hover:text-white`
              )}
            >
              {isActive && (
                <motion.div layoutId="activetab" className={cn("absolute bottom-0 left-0 right-0 h-[2px]", p.bg, p.glow)} />
              )}
              {a.id}
            </button>
          );
        })}
      </div>

      <motion.div
        key={aircraft.id}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel rounded-xl p-8 relative overflow-hidden shadow-lg z-10"
      >
        {/* Left Glowing Edge */}
        <div className={cn("absolute top-0 left-0 bottom-0 w-2", priority.bg, priority.glow)} />

        <div className="flex items-center justify-between mb-8 pl-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h3 className="text-2xl font-mono font-bold text-white tracking-widest">{aircraft.id}</h3>
              <span className={cn("px-2 py-0.5 rounded border text-[10px] font-bold uppercase tracking-widest", priority.bg.replace('bg-', 'bg-')+'/10', priority.color, priority.border, priority.glow)}>
                PRIORITY: {priority.level}
              </span>
            </div>
            <p className="text-xs text-muted uppercase tracking-widest">Suggested action based on RUL {aircraft.rul} cycles</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] font-bold text-muted uppercase tracking-widest">Confidence Level</p>
            <p className="text-xl font-mono font-bold text-[#00E676] drop-shadow-[0_0_8px_#00E676]">92.8%</p>
          </div>
        </div>

        {/* 4-Column HUD Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8 pl-4">
          {[
            { label: 'Action Window', value: aircraft.rul < 50 ? 'Immediate' : 'Next 15 Days', icon: Calendar, color: 'text-primary', shadow: 'shadow-[0_0_8px_rgba(0,191,255,0.4)]' },
            { label: 'Est. Downtime', value: '48 Hrs', icon: Settings, color: 'text-primary', shadow: 'shadow-[0_0_8px_rgba(0,191,255,0.4)]' },
            { label: 'Proj. Cost', value: '$12,500', icon: DollarSign, color: 'text-[#FFB300]', shadow: 'shadow-[0_0_8px_rgba(255,179,0,0.4)]' },
            { label: 'Tech Level', value: 'Level 4+', icon: UserCog, color: 'text-primary', shadow: 'shadow-[0_0_8px_rgba(0,191,255,0.4)]' },
          ].map(col => (
            <div key={col.label} className="bg-black/30 border border-primary/10 rounded-lg p-4 flex flex-col justify-center relative overflow-hidden group">
              <div className="flex items-center mb-2">
                <div className={cn("w-6 h-6 rounded-full flex items-center justify-center border border-white/10 bg-white/5 shrink-0 mr-2", col.shadow)}>
                  <col.icon className={cn("w-3 h-3", col.color)} />
                </div>
                <span className="text-[9px] font-bold text-muted uppercase tracking-widest">{col.label}</span>
              </div>
              <span className="text-xl font-mono font-bold text-white tracking-widest tabular-nums ml-8">{col.value}</span>
            </div>
          ))}
        </div>

        {/* Parts Sub-Panel */}
        <div className="pl-4">
          <h4 className="text-[10px] font-bold text-white uppercase tracking-widest mb-3 flex items-center">
            <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" /> Required Logistics
          </h4>
          <div className="bg-black/40 backdrop-blur-md border border-primary/20 rounded-lg p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { id: 'PT-9921', name: 'HPC Blade Set', stock: 'In Stock', color: 'text-[#00E676]' },
              { id: 'GSK-404', name: 'Thermal Gasket', stock: 'In Transit', color: 'text-[#FFB300]' },
              { id: 'SEN-88', name: 'Temp Probe (T24)', stock: 'In Stock', color: 'text-[#00E676]' },
            ].map(part => (
              <div key={part.id} className="flex justify-between items-center border-b border-white/5 last:border-0 pb-2 last:pb-0">
                <div>
                  <div className="text-primary font-mono text-sm font-bold tracking-widest">{part.id}</div>
                  <div className="text-[9px] text-gray-400 uppercase tracking-widest">{part.name}</div>
                </div>
                <div className={cn("text-[9px] font-bold uppercase tracking-widest", part.color)}>{part.stock}</div>
              </div>
            ))}
          </div>
        </div>

      </motion.div>
    </div>
  );
}
