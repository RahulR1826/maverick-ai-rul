import ReactFlow, { Background, Controls, Position } from 'reactflow';
import type { Edge, Node } from 'reactflow';
import 'reactflow/dist/style.css';
import { motion } from 'framer-motion';
import { ShieldCheck, Network, Cpu, Lock } from 'lucide-react';
import { ArcGauge } from '../components/ArcGauge';
import { cn } from '../utils';

// --- Custom Nodes & Edges ---
// We can use standard nodes but styled heavily with tailwind classes
const initialNodes: Node[] = [
  { id: 'agg', type: 'output', data: { label: 'GLOBAL AGGREGATOR' }, position: { x: 400, y: 50 }, 
    className: 'glass-panel border-2 border-primary text-white font-bold font-mono tracking-widest text-[10px] p-6 w-72 shadow-[0_0_30px_rgba(0,191,255,0.3)]' },
  { id: 'c1', data: { label: 'NODE 01: Airline A' }, position: { x: 100, y: 250 }, 
    className: 'glass-panel border-[#00E676]/30 text-white font-mono tracking-widest text-[9px] p-4 w-40 shadow-[0_0_15px_rgba(0,230,118,0.1)]' },
  { id: 'c2', data: { label: 'NODE 02: Airline B' }, position: { x: 300, y: 250 }, 
    className: 'glass-panel border-[#00E676]/30 text-white font-mono tracking-widest text-[9px] p-4 w-40 shadow-[0_0_15px_rgba(0,230,118,0.1)]' },
  { id: 'c3', data: { label: 'NODE 03: Airline C' }, position: { x: 500, y: 250 }, 
    className: 'glass-panel border-[#00E676]/30 text-white font-mono tracking-widest text-[9px] p-4 w-40 shadow-[0_0_15px_rgba(0,230,118,0.1)]' },
  { id: 'c4', data: { label: 'NODE 04: Airline D' }, position: { x: 700, y: 250 }, 
    className: 'glass-panel border-[#00E676]/30 text-white font-mono tracking-widest text-[9px] p-4 w-40 shadow-[0_0_15px_rgba(0,230,118,0.1)]' },
];

// React Flow supports animated edges natively via CSS stroke-dasharray animation
const initialEdges: Edge[] = [
  { id: 'e1', source: 'c1', target: 'agg', animated: true, style: { stroke: '#00BFFF', strokeWidth: 2, filter: 'drop-shadow(0 0 5px #00BFFF)' } },
  { id: 'e2', source: 'c2', target: 'agg', animated: true, style: { stroke: '#00BFFF', strokeWidth: 2, filter: 'drop-shadow(0 0 5px #00BFFF)' } },
  { id: 'e3', source: 'c3', target: 'agg', animated: true, style: { stroke: '#00BFFF', strokeWidth: 2, filter: 'drop-shadow(0 0 5px #00BFFF)' } },
  { id: 'e4', source: 'c4', target: 'agg', animated: true, style: { stroke: '#00BFFF', strokeWidth: 2, filter: 'drop-shadow(0 0 5px #00BFFF)' } },
];

export function FederatedLearning() {
  return (
    <div className="space-y-6 h-full flex flex-col relative min-h-[calc(100vh-6rem)] pb-20">
      
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="relative z-10">
        <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
          <Network className="w-8 h-8 mr-3 text-primary" /> Federated Learning Hub
        </h2>
        <p className="text-muted text-sm font-mono uppercase tracking-widest">Collaborative model training across isolated fleets</p>
      </motion.div>

      {/* Privacy Guarantee HUD Alert Strip */}
      <motion.div 
        initial={{ opacity: 0, scaleX: 0 }} animate={{ opacity: 1, scaleX: 1 }}
        className="relative z-10 bg-[#00E676]/10 border border-[#00E676]/40 backdrop-blur-md p-3 rounded-lg flex items-center justify-center gap-3 shadow-[0_0_20px_rgba(0,230,118,0.15)]"
      >
        <ShieldCheck className="w-5 h-5 text-[#00E676] drop-shadow-[0_0_5px_#00E676]" />
        <span className="text-[#00E676] font-bold tracking-[0.2em] uppercase text-xs">Privacy Guarantee: Raw Sensor Data NEVER Shared • End-to-End Encryption Active</span>
        <Lock className="w-4 h-4 text-[#00E676] ml-2 opacity-70" />
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
        <div className="glass-panel p-6 flex flex-col items-center justify-center relative">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
          <ArcGauge value={24} max={50} label="Current FL Round" color="#00BFFF" format={v => `${v}/50`} />
        </div>
        
        <div className="glass-panel p-6 flex flex-col items-center justify-center relative">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#00E676]/50 to-transparent" />
          <ArcGauge value={94.2} max={100} label="Global Accuracy" color="#00E676" format={v => `${v}%`} />
        </div>
      </div>

      <div className="glass-panel w-full h-[600px] relative overflow-hidden p-1 z-10">
        <div className="absolute top-4 left-4 z-20 glass-panel border border-primary/30 p-3 pointer-events-none shadow-[0_0_15px_rgba(0,191,255,0.1)]">
          <h4 className="text-[10px] font-bold text-primary uppercase tracking-widest mb-2 flex items-center">
            <Cpu className="w-4 h-4 mr-2" /> Topology Legend
          </h4>
          <div className="flex items-center text-[9px] text-white uppercase tracking-widest font-mono mb-2">
            <span className="w-4 h-0.5 bg-primary shadow-[0_0_5px_#00BFFF] mr-2" /> Model Weights Sync
          </div>
          <div className="flex items-center text-[9px] text-white uppercase tracking-widest font-mono">
            <span className="w-4 h-4 border border-[#00E676] bg-[#00E676]/10 rounded mr-2" /> Isolated Client Node
          </div>
        </div>

        {/* CSS override to make react flow connections glow and feel like data streams */}
        <style dangerouslySetInnerHTML={{__html: `
          .react-flow__edge-path { stroke-width: 2 !important; }
          .react-flow__edge.animated path { animation: dashdraw 1.5s linear infinite !important; stroke-dasharray: 10, 10 !important; }
          @keyframes dashdraw { from { stroke-dashoffset: 20; } to { stroke-dashoffset: 0; } }
        `}} />

        <div style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}>
          <ReactFlow 
            nodes={initialNodes} 
            edges={initialEdges} 
            fitView 
            className="bg-transparent"
            proOptions={{ hideAttribution: true }}
          >
            <Background color="rgba(0,191,255,0.1)" gap={30} size={1.5} />
            <Controls className="glass-panel border-primary/30 fill-primary drop-shadow-[0_0_5px_#00BFFF]" />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}
