import { motion } from 'framer-motion';
import { ShieldAlert, AlertTriangle, AlertCircle, Info, Activity } from 'lucide-react';
import { useStore } from '../store/useStore';
import { cn } from '../utils';

export function Alerts() {
  const fleet = useStore(state => state.fleet);

  // Generate some realistic alerts based on the real fleet data
  const alerts = fleet.flatMap(aircraft => {
    const list = [];
    if (aircraft.risk === 'Critical') {
      list.push({ id: `${aircraft.id}-1`, aircraft: aircraft.id, type: 'RUL Threshold Violation', severity: 'Critical', message: `Remaining useful life critically low (${aircraft.rul} cycles). Immediate grounding advised.`, time: 'Just now', icon: ShieldAlert });
    }
    if (aircraft.health < 60) {
      list.push({ id: `${aircraft.id}-2`, aircraft: aircraft.id, type: 'HPC Efficiency Drop', severity: 'High', message: 'High pressure compressor efficiency deviated by 14% from baseline.', time: '12m ago', icon: AlertTriangle });
    }
    if (aircraft.edgeStatus === 'Syncing') {
      list.push({ id: `${aircraft.id}-3`, aircraft: aircraft.id, type: 'Edge Node Latency', severity: 'Medium', message: 'Model weight synchronization delayed by > 200ms.', time: '1h ago', icon: AlertCircle });
    }
    if (aircraft.maintenanceStatus === 'Scheduled') {
      list.push({ id: `${aircraft.id}-4`, aircraft: aircraft.id, type: 'Maintenance Approaching', severity: 'Low', message: 'Routine borescope inspection scheduled in 48 hours.', time: '3h ago', icon: Info });
    }
    return list;
  }).sort(() => Math.random() - 0.5).slice(0, 15); // shuffle and limit

  const counts = {
    Critical: alerts.filter(a => a.severity === 'Critical').length,
    High: alerts.filter(a => a.severity === 'High').length,
    Medium: alerts.filter(a => a.severity === 'Medium').length,
    Low: alerts.filter(a => a.severity === 'Low').length,
  };

  const getSeverityColor = (sev: string) => {
    switch(sev) {
      case 'Critical': return { text: 'text-[#FF0055]', bg: 'bg-[#FF0055]', glow: 'shadow-[0_0_15px_rgba(255,0,85,0.4)]', pulse: true };
      case 'High': return { text: 'text-[#FF4444]', bg: 'bg-[#FF4444]', glow: 'shadow-[0_0_15px_rgba(255,68,68,0.3)]', pulse: false };
      case 'Medium': return { text: 'text-[#FFB300]', bg: 'bg-[#FFB300]', glow: 'shadow-[0_0_15px_rgba(255,179,0,0.3)]', pulse: false };
      default: return { text: 'text-[#00BFFF]', bg: 'bg-[#00BFFF]', glow: 'shadow-[0_0_15px_rgba(0,191,255,0.3)]', pulse: false };
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      <div className="relative z-10 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
            <Activity className="w-8 h-8 mr-3 text-[#FF0055]" /> Alert Center
          </h2>
          <p className="text-muted text-sm font-mono uppercase tracking-widest">Anomaly Detection • System Notifications</p>
        </div>
      </div>

      {/* KPI Summary Bar */}
      <div className="grid grid-cols-4 gap-4 relative z-10">
        {Object.entries(counts).map(([sev, count], i) => {
          const colors = getSeverityColor(sev);
          return (
            <motion.div key={sev} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
              className="glass-panel p-4 text-center rounded-xl relative overflow-hidden group">
              <div className={`absolute top-0 left-0 w-1 h-full ${colors.bg} ${colors.glow}`} />
              <div className={`text-3xl font-mono font-bold tabular-nums drop-shadow-md ${colors.text}`}>{count}</div>
              <div className="text-[10px] font-bold text-muted uppercase tracking-widest mt-1">{sev}</div>
            </motion.div>
          );
        })}
      </div>

      <div className="space-y-4 relative z-10">
        {alerts.map((alert, i) => {
          const style = getSeverityColor(alert.severity);
          return (
            <motion.div 
              key={alert.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass-panel rounded-xl p-5 flex items-center gap-6 relative overflow-hidden group hover:border-primary/40 transition-colors"
            >
              {/* Full glowing edge */}
              <div className={cn("absolute top-0 left-0 bottom-0 w-1.5", style.bg, style.glow, style.pulse ? "animate-pulse" : "")} />
              
              {/* Glowing Icon Bezel */}
              <div className={cn("relative w-12 h-12 rounded-full border flex items-center justify-center shrink-0 ml-2", style.bg.replace('bg-', 'border-') + '/30')}>
                <div className={cn("absolute inset-0 rounded-full blur-md opacity-30", style.bg)} />
                <alert.icon className={cn("relative z-10 w-5 h-5 drop-shadow-md", style.text)} />
              </div>
              
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <span className={cn("text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border", style.text, style.bg.replace('bg-', 'bg-') + '/10', style.bg.replace('bg-', 'border-') + '/30', style.glow)}>
                    {alert.severity}
                  </span>
                  <span className="text-white font-mono font-bold text-sm">{alert.aircraft}</span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-primary drop-shadow-[0_0_5px_#00BFFF]">{alert.type}</span>
                </div>
                <p className="text-gray-300 text-sm mt-1.5 leading-relaxed">{alert.message}</p>
              </div>
              
              <div className="text-right shrink-0">
                <p className="text-[10px] text-muted font-bold uppercase tracking-widest">{alert.time}</p>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
