import { motion } from 'framer-motion';
import { useStore } from '../store/useStore';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { Database, Cpu, ChevronDown } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { cn } from '../utils';

import { ArcGauge } from '../components/ArcGauge';

// --- Glass Sensor Card ---
const SensorCard = ({ name, value, normalRange }: { name: string, value: number, normalRange: [number, number] }) => {
  const [pulse, setPulse] = useState(false);
  const prevValue = useRef(value);

  useEffect(() => {
    if (value !== prevValue.current) {
      setPulse(true);
      const timer = setTimeout(() => setPulse(false), 300);
      prevValue.current = value;
      return () => clearTimeout(timer);
    }
  }, [value]);

  const isCritical = value < normalRange[0] * 0.95 || value > normalRange[1] * 1.05;
  const isWatch = !isCritical && (value < normalRange[0] || value > normalRange[1]);
  const statusColor = isCritical ? 'bg-[#FF0055] shadow-[0_0_12px_#FF0055]' : isWatch ? 'bg-[#FFB300] shadow-[0_0_12px_#FFB300]' : 'bg-[#00E676] shadow-[0_0_8px_#00E676]';
  const borderColor = isCritical ? 'border-[#FF0055]/40' : isWatch ? 'border-[#FFB300]/40' : 'border-primary/20';

  return (
    <div className={cn("glass-panel relative p-4 overflow-hidden group transition-colors", borderColor)}>
      {/* Top Status Edge */}
      <div className={cn("absolute top-0 left-0 right-0 h-[2px]", statusColor)} />
      
      {/* Update Pulse Effect */}
      <motion.div
        initial={false}
        animate={{ opacity: pulse ? 0.15 : 0 }}
        className="absolute inset-0 bg-white pointer-events-none"
        transition={{ duration: 0.3 }}
      />
      
      <div className="text-[10px] text-muted uppercase tracking-widest mb-1.5">{name}</div>
      <div className="text-2xl font-mono font-bold text-white tabular-nums drop-shadow-md">
        {value.toFixed(2)}
      </div>
    </div>
  );
};

function TurbofanEngine({ healthScore }: { healthScore: number }) {
  const color = healthScore > 70 ? '#00E676' : healthScore > 40 ? '#FFB300' : '#FF0055';
  const ref = useRef<THREE.Group>(null);
  const fanRef = useRef<THREE.Group>(null);
  
  useFrame((_, delta) => {
    if (fanRef.current) {
      fanRef.current.rotation.z -= delta * 2;
    }
  });

  return (
    <group ref={ref} position={[0, 0, 0]} rotation={[0, -Math.PI / 4, 0]}>
      {/* Nacelle/Cowling (Main Body - Horizontal Cylinder) */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[1.8, 1.8, 4.5, 32]} />
        <meshStandardMaterial color="#8BA2B8" metalness={0.6} roughness={0.3} />
      </mesh>
      
      {/* Fan Assembly at Front */}
      <group ref={fanRef} position={[0, 0, 2.25]}>
         {/* Spinner */}
         <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.4]}>
            <coneGeometry args={[0.4, 0.8, 16]} />
            <meshStandardMaterial color="#C0CDDA" metalness={0.7} roughness={0.2} />
         </mesh>

         {/* Fan Blades (12 full-diameter boxes forming 24 spokes) */}
         {Array.from({ length: 12 }).map((_, i) => (
            <mesh key={i} rotation={[0, 0, (i / 12) * Math.PI]}>
               <boxGeometry args={[3.4, 0.08, 0.15]} />
               <meshStandardMaterial color="#AAB8C6" metalness={0.8} roughness={0.4} />
            </mesh>
         ))}
      </group>
      
      {/* Core casing / exhaust at Rear */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -2.5]}>
        <cylinderGeometry args={[0.7, 0.5, 1.5, 16]} />
        <meshStandardMaterial color="#566A7D" metalness={0.8} roughness={0.6} />
      </mesh>

      {/* Sensor Nodes (Glowing markers anchored on the cylinder body surface) */}
      {[
        { pos: [1.8, 0, 1.5], label: 'T24' },
        { pos: [-1.8, 0.5, 0], label: 'P30' },
        { pos: [0, 1.8, -1.0], label: 'N1' },
        { pos: [0, -1.8, 0.5], label: 'VIB' },
      ].map((sensor, i) => (
        <group key={i} position={sensor.pos as [number, number, number]}>
          <mesh>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} />
          </mesh>
          <pointLight color={color} intensity={0.5} distance={2} />
        </group>
      ))}
    </group>
  );
}

// --- Main Page Component ---
export function AircraftMonitor() {
  const { fleet, selectedAircraftId, selectAircraft, sensors, updateSensors } = useStore();
  const aircraft = fleet.find(a => a.id === selectedAircraftId) || fleet[0];

  useEffect(() => {
    const interval = setInterval(updateSensors, 1000);
    return () => clearInterval(interval);
  }, [updateSensors]);

  const healthColor = aircraft.health > 70 ? '#00E676' : aircraft.health > 40 ? '#FFB300' : '#FF0055';
  const rulColor = aircraft.rul > 75 ? '#00BFFF' : aircraft.rul > 30 ? '#FFB300' : '#FF0055';

  const sensorRanges: Record<string, [number, number]> = {
    'Fan Speed (N1)': [5100, 5300],
    'Compressor Speed (N2)': [14600, 15000],
    'Fuel Flow': [3750, 3900],
    'Oil Pressure': [58, 62],
    'Oil Temp': [60, 65],
    'EGT': [1000, 1100],
    'Vibration': [0.2, 0.6],
    'Pressure Ratio': [14.2, 15.0],
  };

  return (
    // Radial gradient background for the sci-fi HUD feel
    <div className="space-y-6 animate-in fade-in duration-500 pb-20 relative min-h-[calc(100vh-6rem)]">
      
      {/* Header Selector */}
      <div className="flex items-center justify-between relative z-10">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-primary/10 rounded-xl border border-primary/30 shadow-[0_0_15px_rgba(0,191,255,0.15)]">
            <Cpu className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white tracking-wider font-mono drop-shadow-md">{aircraft.id} <span className="text-muted text-lg font-sans font-normal">— Digital Twin</span></h2>
            <div className="flex items-center gap-2 mt-1">
              <span className="w-2 h-2 rounded-full bg-success animate-pulse shadow-[0_0_8px_#00E676]" />
              <p className="text-xs text-muted uppercase tracking-widest">Live Telemetry Linked</p>
            </div>
          </div>
        </div>
        
        {/* Glass dropdown */}
        <div className="relative group">
          <select 
            className="appearance-none glass-panel text-white pl-4 pr-10 py-2.5 outline-none focus:border-primary focus:ring-1 focus:ring-primary/50 shadow-[0_0_10px_rgba(0,191,255,0.05)] font-mono text-sm cursor-pointer"
            value={aircraft.id}
            onChange={(e) => selectAircraft(e.target.value)}
          >
            {fleet.map(a => <option key={a.id} value={a.id} className="bg-[#050B14]">{a.id} - {a.type}</option>)}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-primary pointer-events-none" />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 relative z-10">
        
        {/* 3D Scanning Bay */}
        <div className="xl:col-span-2 relative h-[500px] overflow-hidden glass-panel">
          {/* HUD Overlay Elements */}
          <div className="absolute top-4 left-4 z-10">
            <div className="px-3 py-1.5 glass-panel text-[10px] font-bold text-primary uppercase tracking-widest flex items-center shadow-[0_0_10px_rgba(0,191,255,0.2)]">
              <Database className="w-3 h-3 mr-2" /> Turbofan Model Scanned
            </div>
          </div>
          
          <div className="absolute bottom-4 right-4 z-10 text-[10px] font-mono text-primary/70 text-right pointer-events-none">
            <p>X-AXIS: {Math.random().toFixed(4)}</p>
            <p>Y-AXIS: {Math.random().toFixed(4)}</p>
            <p>PITCH: NORMAL</p>
          </div>

          {/* Corner Targeting Brackets */}
          <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary/50 pointer-events-none" />
          <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary/50 pointer-events-none" />
          <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-primary/50 pointer-events-none" />
          <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary/50 pointer-events-none" />

          {/* Glow border ring inside the scanning bay */}
          <div className="absolute inset-0 border border-primary/10 rounded-xl pointer-events-none mix-blend-screen" />

          <Canvas camera={{ position: [5, 2, 5], fov: 45 }}>
            <ambientLight intensity={1.5} />
            <directionalLight position={[10, 10, 10]} intensity={2.5} />
            <directionalLight position={[-10, -10, -5]} intensity={2} color="#00BFFF" />
            <pointLight position={[0, 0, 5]} intensity={2.5} color="#ffffff" />
            
            {/* Holographic Floor Grid */}
            <Grid infiniteGrid fadeDistance={25} sectionColor="rgba(0,191,255,0.15)" cellColor="rgba(0,191,255,0.15)" sectionThickness={1.5} cellThickness={0.5} position={[0, -2.5, 0]} />
            
            <TurbofanEngine healthScore={aircraft.health} />
            
            <OrbitControls enablePan={false} maxPolarAngle={Math.PI / 1.5} minDistance={3} maxDistance={12} />
          </Canvas>
        </div>

        {/* Health & RUL Dials */}
        <div className="flex flex-col gap-6">
          <div className="glass-panel p-6 flex flex-col items-center justify-center flex-1 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
            <h3 className="text-[11px] font-bold text-muted uppercase tracking-[0.2em] mb-4 text-center">System Health</h3>
            <ArcGauge value={aircraft.health} max={100} label="Score" color={healthColor} format={(v) => `${v}%`} />
          </div>

          <div className="glass-panel p-6 flex flex-col items-center justify-center flex-1 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
            <h3 className="text-[11px] font-bold text-muted uppercase tracking-[0.2em] mb-4 text-center">Remaining Useful Life</h3>
            <ArcGauge value={aircraft.rul} max={150} label="Cycles" color={rulColor} />
          </div>
        </div>
      </div>

      {/* Live Sensors Grid */}
      <div className="relative z-10">
        <h3 className="text-sm font-bold text-white flex items-center mb-4 tracking-widest uppercase">
          <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" /> Live Telemetry Feed
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(sensors).map(([key, value]) => (
            <SensorCard 
              key={key} 
              name={key} 
              value={value} 
              normalRange={sensorRanges[key] || [0, 100000]} 
            />
          ))}
        </div>
      </div>

    </div>
  );
}
