import { motion, AnimatePresence } from 'framer-motion';
import { Settings2, Settings as SettingsIcon, Wrench, ShieldAlert, Cpu, Network, CheckCircle } from 'lucide-react';
import { useState } from 'react';
import { cn } from '../utils';

const SETTINGS_SECTIONS = [
  { id: 'fl', title: 'Federated Learning Global Config', icon: Network },
  { id: 'model', title: 'LSTM Model Hyperparameters', icon: BrainCircuitIcon },
  { id: 'edge', title: 'Edge Node Telemetry Policy', icon: Cpu },
  { id: 'privacy', title: 'Privacy & Security Protocol', icon: ShieldAlert },
];

function BrainCircuitIcon(props: any) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M9 13a4.5 4.5 0 1 0 4.3 6.35"/><path d="M12 21v-4"/><path d="M15 13a4.5 4.5 0 1 0-4.3-6.35"/><path d="M12 7V3"/><path d="M12 15a3 3 0 1 0 0-6"/></svg>
}

// Custom Avionics Toggle
const AvionicsToggle = ({ label, enabled, onChange, color = "#00BFFF" }: any) => (
  <div className="flex items-center justify-between py-3 border-b border-white/5 last:border-0 group cursor-pointer" onClick={() => onChange(!enabled)}>
    <span className="text-[11px] font-bold text-gray-300 uppercase tracking-widest group-hover:text-white transition-colors">{label}</span>
    <div className={cn("relative w-12 h-6 rounded-full transition-colors duration-300 border", enabled ? `border-[${color}]/50` : "border-white/20")} style={{ backgroundColor: enabled ? `${color}33` : 'rgba(255,255,255,0.05)' }}>
      <motion.div 
        layout
        className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full"
        style={{ 
          backgroundColor: enabled ? color : '#6B7280',
          boxShadow: enabled ? `0 0 10px ${color}` : 'none'
        }}
        initial={false}
        animate={{ x: enabled ? 24 : 0 }}
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      />
    </div>
  </div>
);

// Glass Input
const GlassInput = ({ label, value, type = "text" }: any) => (
  <div className="flex flex-col gap-1.5">
    <label className="text-[9px] font-bold text-muted uppercase tracking-widest">{label}</label>
    <input 
      type={type} 
      defaultValue={value}
      className="bg-[#02050A]/80 backdrop-blur-md border border-primary/20 text-white font-mono text-sm rounded-lg px-3 py-2 outline-none focus:border-primary focus:ring-1 focus:ring-primary/50 transition-all shadow-[inset_0_0_10px_rgba(0,0,0,0.5)] focus:shadow-[0_0_15px_rgba(0,191,255,0.15)]"
    />
  </div>
);

export function Settings() {
  const [flEnabled, setFlEnabled] = setFlEnabledState(true);
  const [encrypt, setEncrypt] = setFlEnabledState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  function setFlEnabledState(initial: boolean) {
    return useState(initial);
  }

  const handleSave = () => {
    setSaving(true);
    setTimeout(() => {
      setSaving(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }, 800);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      <div className="relative z-10 flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
            <Settings2 className="w-8 h-8 mr-3 text-primary" /> System Configuration
          </h2>
          <p className="text-muted text-sm font-mono uppercase tracking-widest">Global parameters • Access level: COMMAND</p>
        </div>
        
        {/* Save Button */}
        <button 
          onClick={handleSave}
          disabled={saving || saved}
          className={cn("relative overflow-hidden px-6 py-3 rounded-lg border text-[11px] font-bold uppercase tracking-widest transition-all duration-300 outline-none flex items-center",
            saved 
              ? "bg-[#00E676]/20 border-[#00E676] text-[#00E676] shadow-[0_0_20px_rgba(0,230,118,0.4)]"
              : "bg-primary/10 border-primary/50 text-primary hover:bg-primary/20 hover:border-primary hover:shadow-[0_0_20px_rgba(0,191,255,0.3)] hover:-translate-y-0.5 active:translate-y-0"
          )}
        >
          <AnimatePresence mode="wait">
            {saved ? (
              <motion.div key="saved" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} className="flex items-center">
                <CheckCircle className="w-4 h-4 mr-2" /> CONFIG SAVED
              </motion.div>
            ) : saving ? (
              <motion.div key="saving" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center">
                <SettingsIcon className="w-4 h-4 mr-2 animate-spin" /> ARMING...
              </motion.div>
            ) : (
              <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center">
                ARM CONFIGURATION
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Press flash effect */}
          {saving && <motion.div initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ duration: 0.5 }} className="absolute inset-0 bg-white pointer-events-none" />}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative z-10">
        
        {/* FL Config */}
        <div className="glass-panel rounded-xl overflow-hidden shadow-lg group hover:border-primary/40 transition-colors">
          <div className="bg-primary/10 border-b border-primary/20 p-4 flex items-center">
            <Network className="w-5 h-5 text-primary mr-3 drop-shadow-[0_0_5px_#00BFFF]" />
            <h3 className="text-xs font-bold text-white uppercase tracking-widest drop-shadow-md">Federated Learning Hub</h3>
          </div>
          <div className="p-6 space-y-4">
            <AvionicsToggle label="Global Aggregation Engine" enabled={flEnabled} onChange={setFlEnabled} color="#00BFFF" />
            <div className="grid grid-cols-2 gap-4 pt-2">
              <GlassInput label="Max Communication Rounds" value="50" type="number" />
              <GlassInput label="Clients Per Round" value="5" type="number" />
            </div>
            <GlassInput label="Aggregation Strategy" value="FedAvgRUL (Custom)" />
          </div>
        </div>

        {/* Model Config */}
        <div className="glass-panel rounded-xl overflow-hidden shadow-lg group hover:border-primary/40 transition-colors">
          <div className="bg-primary/10 border-b border-primary/20 p-4 flex items-center">
            <BrainCircuitIcon className="w-5 h-5 text-[#00E676] mr-3 drop-shadow-[0_0_5px_#00E676]" />
            <h3 className="text-xs font-bold text-white uppercase tracking-widest drop-shadow-md">LSTM Hyperparameters</h3>
          </div>
          <div className="p-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <GlassInput label="Sequence Length (Cycles)" value="30" type="number" />
              <GlassInput label="Hidden Layers" value="4" type="number" />
              <GlassInput label="Hidden Size" value="128" type="number" />
              <GlassInput label="Dropout Rate" value="0.2" type="number" />
            </div>
            <GlassInput label="Optimizer" value="Adam (LR: 0.001)" />
          </div>
        </div>

        {/* Privacy */}
        <div className="glass-panel rounded-xl overflow-hidden shadow-lg group hover:border-primary/40 transition-colors">
          <div className="bg-primary/10 border-b border-primary/20 p-4 flex items-center">
            <ShieldAlert className="w-5 h-5 text-[#FFB300] mr-3 drop-shadow-[0_0_5px_#FFB300]" />
            <h3 className="text-xs font-bold text-white uppercase tracking-widest drop-shadow-md">Privacy & Security</h3>
          </div>
          <div className="p-6 space-y-4">
            <AvionicsToggle label="Homomorphic Encryption" enabled={encrypt} onChange={setEncrypt} color="#FFB300" />
            <AvionicsToggle label="Differential Privacy Noise" enabled={true} onChange={() => {}} color="#FFB300" />
            <GlassInput label="Noise Multiplier (Epsilon)" value="0.01" type="number" />
          </div>
        </div>
        
        {/* Edge Node */}
        <div className="glass-panel rounded-xl overflow-hidden shadow-lg group hover:border-primary/40 transition-colors">
          <div className="bg-primary/10 border-b border-primary/20 p-4 flex items-center">
            <Cpu className="w-5 h-5 text-[#FF0055] mr-3 drop-shadow-[0_0_5px_#FF0055]" />
            <h3 className="text-xs font-bold text-white uppercase tracking-widest drop-shadow-md">Edge Node Policy</h3>
          </div>
          <div className="p-6 space-y-4">
            <AvionicsToggle label="Allow Offline Inference" enabled={true} onChange={() => {}} color="#FF0055" />
            <GlassInput label="Local Training Epochs" value="5" type="number" />
            <GlassInput label="Max Sync Latency (ms)" value="500" type="number" />
          </div>
        </div>

      </div>
    </div>
  );
}
