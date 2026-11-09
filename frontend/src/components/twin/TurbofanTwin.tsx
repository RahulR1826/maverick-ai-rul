import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { ContactShadows, Grid, Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { Sensor } from '../../services/api';
import { MODULES, SENSOR_STATIONS, attributionColor, deviation, healthColor, toLathe } from './engineModel';
import type { ModuleId, ModuleState } from './engineModel';

export type TwinMode = 'health' | 'attribution' | 'xray';
export type CameraPreset = 'iso' | 'side' | 'front' | 'rear';

const PRESETS: Record<CameraPreset, [number, number, number]> = {
  iso: [5.4, 3.3, 7.4], side: [0, 0.6, 9.6], front: [8.6, 1.0, 2.0], rear: [-8.2, 1.8, 3.2],
};
const CUT = 0.95;                       // half-angle of the cutaway opening (rad), centred on the camera side
const PHI_START = CUT, PHI_LEN = Math.PI * 2 - 2 * CUT;

// ------------------------------------------------------------------ geometry helpers
function lathe(points: [number, number][], cut: boolean) {
  const v = points.map(([r, y]) => new THREE.Vector2(r, y));
  return cut ? new THREE.LatheGeometry(v, 96, PHI_START, PHI_LEN) : new THREE.LatheGeometry(v, 96);
}

function BladeRing({ y, hub, tip, count, chord, twist, color, emissive, intensity }: {
  y: number; hub: number; tip: number; count: number; chord: number; twist: number; color: string; emissive: string; intensity: number;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const span = tip - hub;
  useLayoutEffect(() => {
    const d = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      const th = (i / count) * Math.PI * 2;
      d.position.set(Math.sin(th) * (hub + span / 2), y, Math.cos(th) * (hub + span / 2));
      d.rotation.set(twist, th + Math.PI / 2, 0, 'YXZ');
      d.updateMatrix();
      ref.current!.setMatrixAt(i, d.matrix);
    }
    ref.current!.instanceMatrix.needsUpdate = true;
  }, [count, hub, span, twist, y]);
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, count]} castShadow>
      <boxGeometry args={[span, chord, 0.012]} />
      <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={intensity} metalness={0.75} roughness={0.28} />
    </instancedMesh>
  );
}

// ------------------------------------------------------------------ airflow
function useDotTexture() {
  return useMemo(() => {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d')!;
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.35, 'rgba(255,255,255,0.55)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }, []);
}

const COLD = new THREE.Color('#5ad1ff'), HOT = new THREE.Color('#fff1c1'), WARM = new THREE.Color('#ff7a2f'), BYPASS = new THREE.Color('#7fd8ff');

function coreColor(y: number, out: THREE.Color) {
  if (y > -0.3) return out.copy(COLD).lerp(WARM, Math.max(0, (1.1 - y) / 1.4) * 0.35);  // compression heating
  if (y > -1.0) return out.copy(WARM).lerp(HOT, Math.min(1, (-0.3 - y) / 0.4));           // combustion
  return out.copy(HOT).lerp(WARM, Math.min(1, (-1.0 - y) / 1.6));                          // expansion cooling
}

function Airflow({ speed }: { speed: number }) {
  const N = 1600, NB = 650;
  const tex = useDotTexture();
  const ref = useRef<THREE.Points>(null);
  const state = useMemo(() => {
    const y = new Float32Array(N), r = new Float32Array(N), phi = new Float32Array(N), v = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const bypass = i < NB;
      y[i] = bypass ? 0.2 + Math.random() * 2.9 : -3.2 + Math.random() * 5.4;
      r[i] = bypass ? 0.86 + Math.random() * 0.6 : 0.3 + Math.random() * 0.3;
      phi[i] = (Math.random() * 2 - 1) * (CUT + 0.15);
      v[i] = (bypass ? 1.7 : 1.15) * (0.75 + Math.random() * 0.5);
    }
    return { y, r, phi, v, pos: new Float32Array(N * 3), col: new Float32Array(N * 3) };
  }, []);
  const tmp = useMemo(() => new THREE.Color(), []);
  useFrame((_, dt) => {
    const { y, r, phi, v, pos, col } = state;
    const d = Math.min(dt, 0.05) * speed;
    for (let i = 0; i < N; i++) {
      const bypass = i < NB;
      y[i] -= v[i] * d;
      if (bypass && y[i] < 0.15) { y[i] = 3.1; phi[i] = (Math.random() * 2 - 1) * (CUT + 0.15); }
      if (!bypass && y[i] < -3.3) { y[i] = 2.2; phi[i] = (Math.random() * 2 - 1) * (CUT + 0.15); }
      // core annulus narrows through the compressor and widens through the turbine
      const rr = bypass ? r[i] : r[i] * (y[i] > -0.3 ? 0.85 + 0.15 * Math.min(1, (y[i] + 0.3) / 2.3) : 0.9 + 0.12 * Math.min(1, (-0.3 - y[i]) / 2));
      pos[i * 3] = Math.sin(phi[i]) * rr; pos[i * 3 + 1] = y[i]; pos[i * 3 + 2] = Math.cos(phi[i]) * rr;
      (bypass ? tmp.copy(BYPASS) : coreColor(y[i], tmp)).toArray(col, i * 3);
    }
    const g = ref.current!.geometry;
    (g.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (g.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  });
  return (
    <points ref={ref} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[state.pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[state.col, 3]} />
      </bufferGeometry>
      <pointsMaterial map={tex} size={0.06} vertexColors transparent opacity={0.85} depthWrite={false} blending={THREE.AdditiveBlending} sizeAttenuation />
    </points>
  );
}

// ------------------------------------------------------------------ sensor markers
function SensorMarker({ s, idx, showLabel, highlight }: { s: Sensor; idx: number; showLabel: boolean; highlight: boolean }) {
  const st = SENSOR_STATIONS[s.key];
  const [hover, setHover] = useState(false);
  const ring = useRef<THREE.Mesh>(null);
  const dev = deviation(s, idx) ?? 0;
  const color = healthColor(dev);
  useFrame(({ clock }) => {
    if (!ring.current) return;
    const t = (clock.elapsedTime * (dev > 1 ? 1.6 : 0.8)) % 1;
    ring.current.scale.setScalar(1 + t * 2.2);
    (ring.current.material as THREE.MeshBasicMaterial).opacity = (1 - t) * 0.7;
  });
  if (!st) return null;
  const v = s.values[idx];
  return (
    <group position={toLathe(st.y, st.r, st.phi)}>
      <mesh onPointerOver={(e) => { e.stopPropagation(); setHover(true); }} onPointerOut={() => setHover(false)}>
        <sphereGeometry args={[highlight ? 0.06 : 0.045, 20, 20]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      <mesh ref={ring}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.5} depthWrite={false} toneMapped={false} />
      </mesh>
      {(showLabel || hover || highlight) && (
        <Html center distanceFactor={9} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap -translate-y-6 px-2 py-1 rounded-md text-[10px] font-mono border backdrop-blur-md"
            style={{ borderColor: `${color}88`, background: 'rgba(3,8,16,0.78)', color: '#E6F1FF', boxShadow: `0 0 12px ${color}55` }}>
            <span style={{ color }} className="font-bold">{s.name}</span> {Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3)}{s.unit ? ` ${s.unit}` : ''}
          </div>
        </Html>
      )}
    </group>
  );
}

// ------------------------------------------------------------------ module halo (clickable)
function ModuleHalo({ id, color, opacity, selected, onSelect }: { id: ModuleId; color: string; opacity: number; selected: boolean; onSelect: (id: ModuleId) => void }) {
  const m = MODULES.find((x) => x.id === id)!;
  const ref = useRef<THREE.MeshBasicMaterial>(null);
  useFrame(({ clock }) => {
    if (ref.current) ref.current.opacity = opacity * (selected ? 1.6 + 0.4 * Math.sin(clock.elapsedTime * 4) : 1);
  });
  const h = m.y0 - m.y1;
  return (
    <mesh position={[0, (m.y0 + m.y1) / 2, 0]} onClick={(e) => { e.stopPropagation(); onSelect(id); }}
      onPointerOver={() => (document.body.style.cursor = 'pointer')} onPointerOut={() => (document.body.style.cursor = '')}>
      <cylinderGeometry args={[m.radius, m.radius, h, 64, 1, true, PHI_START, PHI_LEN]} />
      <meshBasicMaterial ref={ref} color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
    </mesh>
  );
}

// ------------------------------------------------------------------ engine
function Engine({ sensors, idx, mode, health, attribution, showLabels, selected, onSelect, speed }: TwinProps & { speed: number }) {
  const lowSpool = useRef<THREE.Group>(null), highSpool = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    const d = Math.min(dt, 0.05);
    if (lowSpool.current) lowSpool.current.rotation.y += d * 2.4 * speed;
    if (highSpool.current) highSpool.current.rotation.y += d * 5.5 * speed;
  });
  const xray = mode === 'xray';
  const nacelle = useMemo(() => lathe([[1.6, 3.05], [1.7, 2.9], [1.76, 2.55], [1.76, 1.3], [1.62, 0.22], [1.52, 0.22], [1.53, 1.3], [1.53, 2.6], [1.57, 2.98], [1.6, 3.05]], true), []);
  const cowl = useMemo(() => lathe([[0.6, 2.05], [0.8, 1.65], [0.82, 0.2], [0.78, -1.25], [0.7, -2.35], [0.64, -2.35], [0.72, -1.25], [0.76, 0.2], [0.75, 1.62], [0.57, 2.05]], true), []);
  const spinner = useMemo(() => lathe([[0.001, 3.08], [0.18, 2.95], [0.34, 2.7], [0.38, 2.5]], false), []);
  const plug = useMemo(() => lathe([[0.42, -2.25], [0.36, -2.6], [0.2, -2.95], [0.001, -3.15]], false), []);
  const shell = { metalness: 0.65, roughness: 0.32, side: THREE.DoubleSide };

  const modColor = (id: ModuleId) => {
    if (mode === 'attribution' && attribution) {
      const maxAbs = Math.max(1e-6, ...Object.values(attribution).map(Math.abs));
      return attributionColor(attribution[id] / maxAbs);
    }
    return healthColor(health[id].score);
  };
  const glow = (id: ModuleId) => (mode === 'attribution' && attribution
    ? 0.1 + 0.9 * Math.abs(attribution[id]) / Math.max(1e-6, ...Object.values(attribution).map(Math.abs))
    : 0.06 + Math.min(0.55, health[id].score * 0.2));
  const blade = (id: ModuleId) => ({ color: '#9fb0c2', emissive: modColor(id), intensity: glow(id) });

  return (
    <group rotation={[0, 0, -Math.PI / 2]}>
      {/* casings (cutaway on the viewer side) */}
      <mesh geometry={nacelle} castShadow receiveShadow>
        {xray ? <meshBasicMaterial color="#38bdf8" wireframe transparent opacity={0.18} /> : <meshStandardMaterial color="#c7d2de" {...shell} />}
      </mesh>
      <mesh geometry={cowl} castShadow>
        {xray ? <meshBasicMaterial color="#38bdf8" wireframe transparent opacity={0.14} /> : <meshStandardMaterial color="#8a99a9" {...shell} />}
      </mesh>
      <mesh geometry={plug}><meshStandardMaterial color="#5c6875" metalness={0.8} roughness={0.35} /></mesh>
      <mesh position={[0, 0.2, 0]}><cylinderGeometry args={[0.06, 0.06, 5.2, 16]} /><meshStandardMaterial color="#4b5563" metalness={0.9} roughness={0.2} /></mesh>

      {/* low-pressure spool: fan, LPC, LPT */}
      <group ref={lowSpool}>
        <mesh geometry={spinner}><meshStandardMaterial color="#e5ebf1" metalness={0.85} roughness={0.15} /></mesh>
        <BladeRing y={2.35} hub={0.36} tip={1.48} count={24} chord={0.22} twist={0.7} {...blade('fan')} />
        {[1.85, 1.6, 1.35].map((y) => <BladeRing key={y} y={y} hub={0.32} tip={0.66} count={30} chord={0.1} twist={0.5} {...blade('lpc')} />)}
        {[-1.4, -1.65, -1.9, -2.15].map((y, i) => <BladeRing key={y} y={y} hub={0.36} tip={0.62 + i * 0.04} count={34} chord={0.12} twist={-0.55} {...blade('lpt')} />)}
      </group>
      {/* high-pressure spool: HPC, HPT */}
      <group ref={highSpool}>
        {[0.95, 0.78, 0.61, 0.44, 0.27, 0.1, -0.07, -0.22].map((y, i) => (
          <BladeRing key={y} y={y} hub={0.3 + i * 0.014} tip={0.64 - i * 0.017} count={40} chord={0.07} twist={0.45} {...blade('hpc')} />
        ))}
        {[-0.95, -1.15].map((y) => <BladeRing key={y} y={y} hub={0.38} tip={0.6} count={36} chord={0.1} twist={-0.6} {...blade('hpt')} />)}
      </group>
      {/* combustor */}
      {/* torus arc rotated so its gap lines up with the cutaway (gap centred on the viewer side) */}
      <group position={[0, -0.55, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <mesh rotation={[0, 0, Math.PI / 2 + CUT]}>
          <torusGeometry args={[0.5, 0.1, 20, 96, PHI_LEN]} />
          <meshStandardMaterial color="#6b4a3a" emissive={modColor('combustor')} emissiveIntensity={0.6 + glow('combustor')} metalness={0.5} roughness={0.4} />
        </mesh>
      </group>
      <pointLight position={[0, -0.6, 0.3]} color="#ff8a3d" intensity={4} distance={2.4} />

      {/* module halos */}
      {MODULES.map((m) => (
        <ModuleHalo key={m.id} id={m.id} color={modColor(m.id)} opacity={xray ? 0.24 : 0.12} selected={selected === m.id} onSelect={onSelect} />
      ))}
      {/* stations */}
      {sensors.filter((s) => SENSOR_STATIONS[s.key]).map((s) => (
        <SensorMarker key={s.key} s={s} idx={idx} showLabel={showLabels}
          highlight={selected !== null && MODULES.find((m) => m.id === selected)!.sensors.includes(s.key)} />
      ))}
      <Airflow speed={speed} />
    </group>
  );
}

// ------------------------------------------------------------------ camera
function CameraRig({ preset, autoRotate }: { preset: CameraPreset; autoRotate: boolean }) {
  const { camera, size } = useThree();
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const anim = useRef(true);
  const last = useRef(`${preset}`);
  // pull the camera back on narrow viewports so the whole engine (about 6.5 units long) stays in frame
  const fit = Math.max(1, 1.45 / (size.width / Math.max(1, size.height)));
  const key = `${preset}:${fit.toFixed(2)}`;
  if (last.current !== key) { last.current = key; anim.current = true; }
  const target = useMemo(() => new THREE.Vector3(...PRESETS[preset]).multiplyScalar(fit), [preset, fit]);
  useFrame(() => {
    if (controls) controls.autoRotate = autoRotate && !anim.current;
    if (!anim.current) return;
    camera.position.lerp(target, 0.08);
    controls?.target.lerp(new THREE.Vector3(0, 0, 0), 0.1);
    controls?.update();
    if (camera.position.distanceTo(target) < 0.02) anim.current = false;
  });
  return null;
}

export interface TwinProps {
  sensors: Sensor[];
  idx: number;
  mode: TwinMode;
  health: Record<ModuleId, ModuleState>;
  attribution: Record<ModuleId, number> | null;
  showLabels: boolean;
  selected: ModuleId | null;
  onSelect: (id: ModuleId) => void;
}

export function TurbofanTwin(props: TwinProps & { preset: CameraPreset; autoRotate: boolean; running: boolean; accent: string }) {
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: PRESETS.iso, fov: 38 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
      <color attach="background" args={['#030712']} />
      <fog attach="fog" args={['#030712', 13, 28]} />
      <ambientLight intensity={0.35} />
      <hemisphereLight args={['#9cc9ff', '#0b1220', 0.6]} />
      <directionalLight position={[4, 6, 5]} intensity={2.4} castShadow shadow-mapSize={[1024, 1024]} />
      <directionalLight position={[-6, 2, -4]} intensity={1.6} color={props.accent} />
      <spotLight position={[0, 6, 0]} angle={0.5} penumbra={0.8} intensity={30} color="#7dd3fc" />
      <Engine {...props} speed={props.running ? 1 : 0.25} />
      <ContactShadows position={[0, -1.95, 0]} opacity={0.55} scale={12} blur={2.6} far={4} color="#000814" />
      <Grid infiniteGrid position={[0, -1.96, 0]} cellSize={0.4} sectionSize={2} cellColor="#0e2a3d" sectionColor={props.accent} sectionThickness={0.8} fadeDistance={18} fadeStrength={2} />
      <OrbitControls makeDefault enablePan={false} minDistance={3.5} maxDistance={20} maxPolarAngle={Math.PI * 0.62} autoRotateSpeed={0.6} />
      <CameraRig preset={props.preset} autoRotate={props.autoRotate} />
    </Canvas>
  );
}
