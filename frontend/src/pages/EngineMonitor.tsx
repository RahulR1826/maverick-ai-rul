import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Pause, Play, SkipBack, SkipForward, ChevronLeft, ChevronRight, CheckCircle2, AlertTriangle, Maximize2, Tag, RotateCw, Box, Activity, BrainCircuit, ScanLine, Loader2,
} from 'lucide-react';
import { Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useEngine, useExplain, useFleet } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Loadable, Note, Panel, RiskBadge, RISK_COLOR, fmt, tooltipStyle } from '../components/ui';
import type { EngineDetail, Sensor } from '../services/api';
import { cn } from '../utils';
import { RiskTimeline, RulGauge } from '../components/twin/Hud';
import { MODULES, attributionColor, healthColor, moduleAttribution, moduleHealth } from '../components/twin/engineModel';
import type { ModuleId } from '../components/twin/engineModel';
import type { CameraPreset, TwinMode } from '../components/twin/TurbofanTwin';

const TurbofanTwin = lazy(() => import('../components/twin/TurbofanTwin').then((m) => ({ default: m.TurbofanTwin })));
const SPEEDS = [1, 4, 10] as const;
const MODES: { key: TwinMode; label: string; icon: typeof Activity; hint: string }[] = [
  { key: 'health', label: 'Health', icon: Activity, hint: 'Modules glow by how far their sensors are from the typical healthy range' },
  { key: 'attribution', label: 'AI attribution', icon: BrainCircuit, hint: 'Modules coloured by integrated-gradients attribution of their sensors (computed while paused)' },
  { key: 'xray', label: 'X-ray', icon: ScanLine, hint: 'Transparent casings with module envelopes' },
];

const fmtSensor = (v: number | undefined) => (v === undefined || Number.isNaN(v) ? '—' : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));

function SensorCard({ s, idx }: { s: Sensor; idx: number }) {
  const v = s.values[idx];
  const constant = s.healthy_low === s.healthy_high;
  const outside = !constant && (v < s.healthy_low || v > s.healthy_high);
  const spark = s.values.slice(Math.max(0, idx - 60), idx + 1).map((y, i) => ({ i, y }));
  return (
    <div className={cn('glass-panel-sm p-3 transition-colors', outside ? 'border-[#FFB300]/50' : '')} title={s.description}>
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted uppercase tracking-widest">{s.name}{s.unit ? ` (${s.unit})` : ''}</span>
        {constant ? <span className="text-[9px] text-muted">constant</span> : outside
          ? <span className="flex items-center text-[9px] font-bold text-[#FFB300]"><AlertTriangle className="w-3 h-3 mr-1" />Atypical</span>
          : <span className="flex items-center text-[9px] text-[#00E676]"><CheckCircle2 className="w-3 h-3 mr-1" />Typical</span>}
      </div>
      <div className="text-lg font-mono font-bold text-white tabular-nums">{fmtSensor(v)}</div>
      <div className="h-8">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={spark}><YAxis hide domain={['auto', 'auto']} /><Line dataKey="y" dot={false} stroke={outside ? '#FFB300' : '#00BFFF'} strokeWidth={1.5} isAnimationActive={false} /></LineChart>
        </ResponsiveContainer>
      </div>
      <div className="text-[9px] text-muted">typical healthy: {fmtSensor(s.healthy_low)} – {fmtSensor(s.healthy_high)}</div>
    </div>
  );
}

function Seg<T extends string>({ value, onChange, items, id }: { value: T; onChange: (v: T) => void; items: { key: T; label: string; icon?: typeof Activity }[]; id: string }) {
  return (
    <div className="flex p-0.5 rounded-lg bg-black/50 border border-white/10 backdrop-blur-md" role="group" aria-label={id}>
      {items.map((it) => (
        <button key={it.key} onClick={() => onChange(it.key)} aria-pressed={value === it.key}
          className={cn('relative flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-bold rounded-md transition-colors', value === it.key ? 'text-white' : 'text-muted hover:text-white')}>
          {value === it.key && <motion.span layoutId={`seg-${id}`} className="absolute inset-0 rounded-md bg-primary/25 border border-primary/50" transition={{ type: 'spring', stiffness: 400, damping: 32 }} />}
          {it.icon && <it.icon className="w-3.5 h-3.5 relative" />}<span className="relative">{it.label}</span>
        </button>
      ))}
    </div>
  );
}

function IconBtn({ on, onClick, label, children }: { on?: boolean; onClick: () => void; label: string; children: ReactNode }) {
  return (
    <button onClick={onClick} aria-label={label} title={label} aria-pressed={on}
      className={cn('p-2 rounded-lg border backdrop-blur-md transition-colors', on ? 'border-primary/60 text-primary bg-primary/15' : 'border-white/10 text-muted bg-black/50 hover:text-white')}>
      {children}
    </button>
  );
}

function Twin({ e }: { e: EngineDetail }) {
  const { thresholds, cycleByEngine, setCycle, family } = useStore();
  const first = e.trajectory[0].cycle;
  const last = e.trajectory[e.trajectory.length - 1].cycle;
  const cycle = Math.min(Math.max(cycleByEngine[e.engine_id] ?? last, first), last);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(4);
  const [mode, setMode] = useState<TwinMode>('health');
  const [preset, setPreset] = useState<CameraPreset>('iso');
  const [labels, setLabels] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [selected, setSelected] = useState<ModuleId | null>(null);
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      const cur = useStore.getState().cycleByEngine[e.engine_id] ?? last;
      if (cur >= last) { setPlaying(false); return; }
      setCycle(e.engine_id, Math.min(last, cur + 1));
    }, 1000 / speed);
    return () => clearInterval(t);
  }, [playing, speed, e.engine_id, last, setCycle]);

  const go = (c: number) => setCycle(e.engine_id, Math.min(Math.max(c, first), last));
  const toggle = () => { if (!playing && cycle >= last) setCycle(e.engine_id, first); setPlaying((v) => !v); };

  // keyboard: space play/pause, arrows step, 1-3 view modes
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const tag = (ev.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (ev.target as HTMLElement)?.getAttribute?.('role') === 'slider') return;
      if (ev.code === 'Space') { ev.preventDefault(); toggle(); }
      else if (ev.key === 'ArrowRight') { setPlaying(false); go(cycle + 1); }
      else if (ev.key === 'ArrowLeft') { setPlaying(false); go(cycle - 1); }
      else if (ev.key === '1') setMode('health');
      else if (ev.key === '2') setMode('attribution');
      else if (ev.key === '3') setMode('xray');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const idx = cycle - 1;                                    // sensor series start at cycle 1
  const p = e.trajectory[cycle - first];
  const health = useMemo(() => moduleHealth(e.sensors, idx), [e.sensors, idx]);

  // integrated-gradients attribution for the current cycle, fetched only while paused (debounced)
  const [ixCycle, setIxCycle] = useState<number | undefined>(undefined);
  useEffect(() => {
    if (mode !== 'attribution' || playing) return;
    const t = setTimeout(() => setIxCycle(cycle), 350);
    return () => clearTimeout(t);
  }, [mode, playing, cycle]);
  const ix = useExplain(e.engine_id, mode === 'attribution' ? ixCycle : undefined);
  const attribution = mode === 'attribution' && ix.data && ix.data.cycle === ixCycle ? moduleAttribution(ix.data) : null;
  const maxAttr = attribution ? Math.max(1e-6, ...Object.values(attribution).map(Math.abs)) : 1;

  const chart = useMemo(() => e.trajectory.map((t) => ({ cycle: t.cycle, band: [t.rul_lower, t.rul_upper], mean: t.rul_mean, truth: t.true_rul })), [e]);
  const selMod = selected ? MODULES.find((m) => m.id === selected)! : null;

  return (
    <>
      <div className="grid grid-cols-1 2xl:grid-cols-[1fr_340px] gap-6 relative z-10">
        {/* ---------------- 3D viewport ---------------- */}
        <div ref={viewport} className="relative h-[640px] rounded-2xl overflow-hidden border border-cyan-400/25 shadow-[0_0_60px_rgba(0,191,255,0.12)] bg-[#030712]">
          <Suspense fallback={<div className="absolute inset-0 flex items-center justify-center text-muted"><Loader2 className="w-5 h-5 animate-spin mr-2" />Building digital twin…</div>}>
            <TurbofanTwin sensors={e.sensors} idx={idx} mode={mode} health={health} attribution={attribution} showLabels={labels}
              selected={selected} onSelect={(id) => setSelected((s) => (s === id ? null : id))} preset={preset} autoRotate={autoRotate}
              running={playing} accent={RISK_COLOR[p.risk]} />
          </Suspense>

          <div className="pointer-events-none absolute inset-0 scanlines" />
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_55%,rgba(3,7,18,0.85))]" />
          {['top-3 left-3 border-t-2 border-l-2', 'top-3 right-3 border-t-2 border-r-2', 'bottom-3 left-3 border-b-2 border-l-2', 'bottom-3 right-3 border-b-2 border-r-2'].map((c) => (
            <div key={c} className={cn('pointer-events-none absolute w-6 h-6 border-cyan-300/60', c)} />
          ))}

          {/* status */}
          <div className="absolute top-5 left-6 pointer-events-none">
            <div className="text-[10px] tracking-[0.35em] text-cyan-300/80 font-bold">DIGITAL TWIN · SCHEMATIC</div>
            <div className="text-3xl font-black text-white tracking-tight mt-1">ENGINE <span className="text-primary">#{e.engine_id}</span></div>
            <div className="flex items-center gap-2 mt-2">
              <RiskBadge risk={p.risk} />
              <span className="text-[11px] font-mono text-gray-300 bg-black/50 border border-white/10 rounded px-2 py-1">CYCLE {cycle} / {last}</span>
              <span className="text-[11px] font-mono text-gray-400 bg-black/50 border border-white/10 rounded px-2 py-1">{family}</span>
            </div>
          </div>

          {/* controls */}
          <div className="absolute top-5 right-6 flex flex-col items-end gap-2">
            <Seg id="mode" value={mode} onChange={setMode} items={MODES} />
            <div className="flex gap-1.5">
              <Seg id="camera" value={preset} onChange={setPreset} items={[{ key: 'iso', label: 'Iso' }, { key: 'side', label: 'Side' }, { key: 'front', label: 'Front' }, { key: 'rear', label: 'Rear' }]} />
              <IconBtn on={labels} onClick={() => setLabels((v) => !v)} label="Sensor labels"><Tag className="w-4 h-4" /></IconBtn>
              <IconBtn on={autoRotate} onClick={() => setAutoRotate((v) => !v)} label="Auto-rotate"><RotateCw className="w-4 h-4" /></IconBtn>
              <IconBtn onClick={() => viewport.current?.requestFullscreen?.()} label="Fullscreen"><Maximize2 className="w-4 h-4" /></IconBtn>
            </div>
            <AnimatePresence mode="wait">
              <motion.p key={mode} initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="max-w-[300px] text-right text-[10px] text-cyan-100/70 bg-black/40 rounded px-2 py-1">
                {MODES.find((m) => m.key === mode)!.hint}
                {mode === 'attribution' && (playing ? ' — pause to compute' : ix.isFetching ? ' — computing…' : '')}
              </motion.p>
            </AnimatePresence>
            {attribution && (() => {
              const top = MODULES.reduce((a, b) => (attribution[b.id] < attribution[a.id] ? b : a));
              return attribution[top.id] < 0 ? (
                <motion.div key={`${top.id}-${ixCycle}`} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  className="max-w-[300px] rounded-lg border border-[#FF2E63]/50 bg-[#FF2E63]/10 backdrop-blur-md px-3 py-2 text-right">
                  <div className="text-[9px] tracking-[0.25em] text-[#ff8fab] font-bold">LARGEST DRIVER OF LOW RUL · CYCLE {ixCycle}</div>
                  <div className="text-sm font-bold text-white">{top.name}</div>
                  <div className="text-[11px] font-mono text-[#ff8fab]">{attribution[top.id].toFixed(1)} cycles of attribution</div>
                  <div className="text-[10px] text-gray-400 mt-1">For reference: FD001's simulated fault mode is HPC degradation (NASA C-MAPSS readme).</div>
                </motion.div>
              ) : null;
            })()}
          </div>

          {/* module legend */}
          <div className="absolute left-6 top-40 flex flex-col gap-1.5">
            {MODULES.map((m) => {
              const color = mode === 'attribution' && attribution ? attributionColor(attribution[m.id] / maxAttr) : healthColor(health[m.id].score);
              return (
                <button key={m.id} onClick={() => setSelected((s) => (s === m.id ? null : m.id))}
                  className={cn('flex items-center gap-2 text-[11px] px-2 py-1 rounded-md border backdrop-blur-md text-left transition-all',
                    selected === m.id ? 'border-white/60 bg-white/10 text-white' : 'border-white/10 bg-black/45 text-gray-300 hover:text-white')}>
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: color, boxShadow: `0 0 10px ${color}` }} />
                  <span className="w-40">{m.name}</span>
                  <span className="font-mono text-[10px] text-gray-400">
                    {mode === 'attribution' ? (attribution ? `${attribution[m.id] > 0 ? '+' : ''}${attribution[m.id].toFixed(1)} cyc` : '…') : health[m.id].label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* playback deck */}
          <div className="absolute left-4 right-4 bottom-4 rounded-xl border border-white/10 bg-black/55 backdrop-blur-xl px-4 py-3">
            <div className="flex items-center gap-3">
              <IconBtn onClick={() => { setPlaying(false); go(first); }} label="First cycle"><SkipBack className="w-4 h-4" /></IconBtn>
              <button onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}
                className="w-11 h-11 shrink-0 rounded-full flex items-center justify-center text-[#030712] shadow-[0_0_25px_rgba(0,191,255,0.6)] bg-primary hover:scale-105 transition-transform">
                {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </button>
              <IconBtn onClick={() => { setPlaying(false); go(last); }} label="Last cycle"><SkipForward className="w-4 h-4" /></IconBtn>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between text-[10px] font-mono text-gray-400 mb-0.5">
                  <span>cycle {first}</span><span className="text-white">◆ {cycle}</span><span>cycle {last}</span>
                </div>
                <RiskTimeline trajectory={e.trajectory} cycle={cycle} onSeek={(c) => { setPlaying(false); go(c); }} />
              </div>
              <Seg id="speed" value={String(speed)} onChange={(v) => setSpeed(Number(v))} items={SPEEDS.map((s) => ({ key: String(s), label: `${s}×` }))} />
            </div>
            <div className="text-[10px] text-gray-500 mt-1">Space play/pause · ← → step · 1-2-3 views · drag to orbit · click a module</div>
          </div>
        </div>

        {/* ---------------- side panel ---------------- */}
        <div className="flex flex-col gap-4">
          <div className="glass-panel p-4 flex flex-col items-center">
            <p className="text-[10px] font-bold text-muted uppercase tracking-widest self-start">Remaining useful life</p>
            <RulGauge mean={p.rul_mean} lo={p.rul_lower} hi={p.rul_upper} truth={p.true_rul} risk={p.risk} />
            <div className="grid grid-cols-2 gap-2 w-full text-center">
              <div className="rounded-lg bg-black/30 p-2"><div className="text-[9px] text-muted uppercase tracking-widest">Error</div>
                <div className="font-mono text-sm" style={{ color: p.rul_mean - p.true_rul > 0 ? '#FFB300' : '#9fb4c8' }}>{p.rul_mean - p.true_rul > 0 ? '+' : ''}{fmt(p.rul_mean - p.true_rul, 1)}</div></div>
              <div className="rounded-lg bg-black/30 p-2"><div className="text-[9px] text-muted uppercase tracking-widest">Worst case</div>
                <div className="font-mono text-sm" style={{ color: RISK_COLOR[p.risk] }}>{fmt(p.rul_lower, 0)} cyc</div></div>
            </div>
          </div>

          <div className="glass-panel p-4 flex-1">
            <p className="text-[10px] font-bold text-muted uppercase tracking-widest mb-3">{selMod ? selMod.name : 'Module health'}</p>
            <AnimatePresence mode="wait">
              {selMod ? (
                <motion.div key={selMod.id} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} className="space-y-2">
                  {selMod.sensors.map((k) => {
                    const s = e.sensors.find((x) => x.key === k);
                    if (!s) return null;
                    const a = ix.data?.features.find((f) => f.key === k)?.attribution;
                    return (
                      <div key={k} className="rounded-lg bg-black/30 p-2">
                        <div className="flex justify-between gap-2 text-[11px]"><span className="text-white font-bold">{s.name}</span><span className="text-muted text-right">{s.description}</span></div>
                        <div className="flex justify-between font-mono text-xs mt-1"><span>{fmtSensor(s.values[idx])} {s.unit}</span>
                          {mode === 'attribution' && a !== undefined && <span style={{ color: a < 0 ? '#FF2E63' : '#00BFFF' }}>{a > 0 ? '+' : ''}{a.toFixed(2)} cyc</span>}</div>
                      </div>
                    );
                  })}
                  <button onClick={() => setSelected(null)} className="text-[11px] text-primary mt-1">← all modules</button>
                </motion.div>
              ) : (
                <motion.div key="all" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-2.5">
                  {MODULES.map((m) => {
                    const h = health[m.id];
                    const c = healthColor(h.score);
                    return (
                      <button key={m.id} onClick={() => setSelected(m.id)} className="w-full text-left group">
                        <div className="flex justify-between text-[11px] mb-1"><span className="text-gray-200 group-hover:text-white">{m.name}</span><span style={{ color: c }}>{h.label}</span></div>
                        <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                          <motion.div className="h-full rounded-full" style={{ background: c, boxShadow: `0 0 8px ${c}` }}
                            animate={{ width: `${Math.min(100, (h.score / 3) * 100)}%` }} transition={{ type: 'spring', stiffness: 120, damping: 20 }} />
                        </div>
                      </button>
                    );
                  })}
                  <p className="text-[10px] text-muted pt-1">Bar = largest deviation of the module's sensors from the typical healthy range (5-cycle average). A full bar is 3× the healthy half-range.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <Panel title="Remaining useful life over the engine's history">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chart} margin={{ top: 5, right: 30, left: 0, bottom: 15 }}>
              <defs><linearGradient id="band" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#00BFFF" stopOpacity={0.35} /><stop offset="1" stopColor="#00BFFF" stopOpacity={0.08} /></linearGradient></defs>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="cycle" type="number" domain={[first, last]} stroke="#6B8CAE" fontSize={11} label={{ value: 'Cycle', position: 'insideBottom', offset: -8, fill: '#6B8CAE', fontSize: 11 }} />
              <YAxis stroke="#6B8CAE" fontSize={11} domain={[0, 'auto']} label={{ value: 'RUL (cycles)', angle: -90, position: 'insideLeft', fill: '#6B8CAE', fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => (Array.isArray(v) ? [`${fmt(v[0], 0)} – ${fmt(v[1], 0)}`, 'Interval'] : [fmt(v as number, 1), String(n)])} labelFormatter={(l) => `Cycle ${l}`} />
              <Area dataKey="band" stroke="none" fill="url(#band)" name="Interval" isAnimationActive={false} />
              <Line dataKey="mean" stroke="#00BFFF" strokeWidth={2} dot={false} name="Predicted (ensemble mean)" isAnimationActive={false} />
              <Line dataKey="truth" stroke="#E0E6ED" strokeDasharray="5 4" strokeWidth={1.5} dot={false} name="True RUL" isAnimationActive={false} />
              <ReferenceLine y={thresholds.critical} stroke="#FF0055" strokeDasharray="3 3" label={{ value: 'critical', fill: '#FF0055', fontSize: 10, position: 'right' }} />
              <ReferenceLine y={thresholds.high} stroke="#FF6B4A" strokeDasharray="3 3" label={{ value: 'high', fill: '#FF6B4A', fontSize: 10, position: 'right' }} />
              <ReferenceLine x={cycle} stroke="#FFB300" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-muted mt-2">Solid: ensemble mean · band: calibrated interval · dashed: true RUL (capped at 125) · amber: replay position.</p>
      </Panel>

      <Panel title={`Sensors at cycle ${cycle} (raw values)`}>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
          {e.sensors.map((s) => <SensorCard key={s.key} s={s} idx={idx} />)}
        </div>
        <p className="text-[11px] text-muted mt-3">The model's 17 inputs. Typical healthy range = 1st–99th percentile of training cycles at the RUL cap, so about 2% of healthy readings are flagged 'atypical' by chance. A flag is a cue to look, not a fault.</p>
      </Panel>
    </>
  );
}

export function EngineMonitor() {
  const { selectedEngineId, selectEngine } = useStore();
  const fleet = useFleet();
  const engine = useEngine(selectedEngineId);
  const ids = (fleet.data ?? []).map((f) => f.engine_id);
  const step = (d: number) => { if (ids.length) selectEngine(ids[(ids.indexOf(selectedEngineId) + d + ids.length) % ids.length]); };
  return (
    <div className="space-y-6 pb-20 relative">
      <div className="flex flex-wrap items-end justify-between gap-4 relative z-10">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center"><Box className="w-8 h-8 mr-3 text-primary" />Engine digital twin</h2>
          <p className="text-muted text-sm">Replay a real FD001 test engine through the served model; the twin is driven by its recorded sensors.</p>
        </div>
        <div className="flex items-center gap-2">
          <IconBtn onClick={() => step(-1)} label="Previous engine"><ChevronLeft className="w-4 h-4" /></IconBtn>
          <select value={selectedEngineId} onChange={(ev) => selectEngine(Number(ev.target.value))} aria-label="Select engine"
            className="glass-panel text-white px-4 py-2.5 outline-none focus:border-primary font-mono text-sm cursor-pointer">
            {(fleet.data ?? Array.from({ length: 100 }, (_, i) => ({ engine_id: i + 1, risk: undefined }))).map((f) => (
              <option key={f.engine_id} value={f.engine_id} className="bg-[#050B14]">Engine #{f.engine_id}{f.risk ? ` · ${f.risk}` : ''}</option>
            ))}
          </select>
          <IconBtn onClick={() => step(1)} label="Next engine"><ChevronRight className="w-4 h-4" /></IconBtn>
        </div>
      </div>
      <Loadable q={engine} label="engine history">{(e) => <Twin key={e.engine_id} e={e} />}</Loadable>
      <Note>
        The 3D engine is a schematic (not to scale) built from the C-MAPSS station layout. Module colours, sensor markers and readouts come from this engine's recorded data and the served model.
        Spool rotation and airflow particles are illustrative. C-MAPSS test engines stop before failure, and the model never saw them or their labels.
      </Note>
    </div>
  );
}
