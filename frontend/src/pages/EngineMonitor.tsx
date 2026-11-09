import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { Gauge, Pause, Play, SkipBack, SkipForward, ChevronDown, CheckCircle2, AlertTriangle } from 'lucide-react';
import { Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useEngine, useFleet } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Kpi, Loadable, Note, PageHeader, Panel, RiskBadge, RISK_COLOR, fmt, tooltipStyle } from '../components/ui';
import type { EngineDetail, Sensor } from '../services/api';
import { cn } from '../utils';

const Turbofan3D = lazy(() => import('../components/Turbofan3D').then((m) => ({ default: m.Turbofan3D })));
const SPEEDS = [1, 4, 10] as const;

const fmtSensor = (v: number | undefined) => (v === undefined || Number.isNaN(v) ? '—' : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));

function SensorCard({ s, idx, startIdx }: { s: Sensor; idx: number; startIdx: number }) {
  const v = s.values[idx];
  const constant = s.healthy_low === s.healthy_high;
  const outside = !constant && (v < s.healthy_low || v > s.healthy_high);
  const spark = s.values.slice(Math.max(0, idx - 60), idx + 1).map((y, i) => ({ i: i + startIdx, y }));
  return (
    <div className={cn('glass-panel-sm p-3', outside ? 'border-[#FFB300]/50' : '')} title={s.description}>
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

function Replay({ e }: { e: EngineDetail }) {
  const { thresholds, cycleByEngine, setCycle } = useStore();
  const first = e.trajectory[0].cycle;
  const last = e.trajectory[e.trajectory.length - 1].cycle;
  const cycle = Math.min(Math.max(cycleByEngine[e.engine_id] ?? last, first), last);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(4);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      const cur = useStore.getState().cycleByEngine[e.engine_id] ?? last;
      if (cur >= last) { setPlaying(false); return; }
      setCycle(e.engine_id, Math.min(last, cur + 1));
    }, 1000 / speed);
    return () => clearInterval(t);
  }, [playing, speed, e.engine_id, last, setCycle]);

  const p = e.trajectory[cycle - first];
  const chart = useMemo(() => e.trajectory.map((t) => ({ cycle: t.cycle, band: [t.rul_lower, t.rul_upper], mean: t.rul_mean, truth: t.true_rul })), [e]);
  const go = (c: number) => { setPlaying(false); setCycle(e.engine_id, Math.min(Math.max(c, first), last)); };

  return (
    <>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 relative z-10">
        <Kpi label={`Predicted RUL at cycle ${cycle}`} value={`${fmt(p.rul_mean, 0)}`} color={RISK_COLOR[p.risk]} sub={`interval [${fmt(p.rul_lower, 0)}, ${fmt(p.rul_upper, 0)}] cycles`} />
        <Kpi label="True RUL (evaluation)" value={fmt(p.true_rul, 0)} color="#9fb4c8" sub={`error ${p.rul_mean - p.true_rul > 0 ? '+' : ''}${fmt(p.rul_mean - p.true_rul, 1)} cycles`} />
        <div className="glass-panel p-5"><p className="text-[10px] font-bold text-muted uppercase tracking-widest mb-2">Risk (from lower bound)</p><RiskBadge risk={p.risk} /></div>
        <Kpi label="Replay position" value={`${cycle} / ${last}`} sub={`predictions start at cycle ${first} (30-cycle window)`} />
      </div>

      <Panel title="Remaining useful life over the engine's history">
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <button onClick={() => go(first)} className="p-2 rounded border border-white/10 hover:border-primary" aria-label="First cycle"><SkipBack className="w-4 h-4" /></button>
          <button onClick={() => { if (cycle >= last) setCycle(e.engine_id, first); setPlaying(!playing); }}
            className="px-4 py-2 rounded border border-primary text-primary flex items-center gap-2 text-sm font-bold" aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}{playing ? 'Pause' : 'Replay'}
          </button>
          <button onClick={() => go(last)} className="p-2 rounded border border-white/10 hover:border-primary" aria-label="Last cycle"><SkipForward className="w-4 h-4" /></button>
          <input type="range" min={first} max={last} value={cycle} onChange={(ev) => go(Number(ev.target.value))}
            className="flex-1 min-w-[200px] accent-[#00BFFF]" aria-label="Cycle" />
          <div className="flex gap-1" role="group" aria-label="Replay speed">
            {SPEEDS.map((s) => (
              <button key={s} onClick={() => setSpeed(s)} aria-pressed={speed === s}
                className={cn('px-2 py-1 rounded text-xs border', speed === s ? 'border-primary text-primary' : 'border-white/10 text-muted')}>{s}×</button>
            ))}
          </div>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chart} margin={{ top: 5, right: 20, left: 0, bottom: 15 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="cycle" type="number" domain={[first, last]} stroke="#6B8CAE" fontSize={11} label={{ value: 'Cycle', position: 'insideBottom', offset: -8, fill: '#6B8CAE', fontSize: 11 }} />
              <YAxis stroke="#6B8CAE" fontSize={11} domain={[0, 'auto']} label={{ value: 'RUL (cycles)', angle: -90, position: 'insideLeft', fill: '#6B8CAE', fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => (Array.isArray(v) ? [`${fmt(v[0], 0)} – ${fmt(v[1], 0)}`, 'Interval'] : [fmt(v as number, 1), String(n)])} labelFormatter={(l) => `Cycle ${l}`} />
              <Area dataKey="band" stroke="none" fill="#00BFFF" fillOpacity={0.15} name="Interval" isAnimationActive={false} />
              <Line dataKey="mean" stroke="#00BFFF" strokeWidth={2} dot={false} name="Predicted (ensemble mean)" isAnimationActive={false} />
              <Line dataKey="truth" stroke="#E0E6ED" strokeDasharray="5 4" strokeWidth={1.5} dot={false} name="True RUL" isAnimationActive={false} />
              <ReferenceLine y={thresholds.critical} stroke="#FF0055" strokeDasharray="3 3" label={{ value: 'critical', fill: '#FF0055', fontSize: 10, position: 'right' }} />
              <ReferenceLine y={thresholds.high} stroke="#FF6B4A" strokeDasharray="3 3" label={{ value: 'high', fill: '#FF6B4A', fontSize: 10, position: 'right' }} />
              <ReferenceLine x={cycle} stroke="#FFB300" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-muted mt-2">Solid line: ensemble mean · band: calibrated interval · dashed: true RUL (capped at 125) · amber line: replay position.</p>
      </Panel>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <Panel title={`Sensors at cycle ${cycle} (raw values)`} className="xl:col-span-2">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {e.sensors.map((s) => <SensorCard key={s.key} s={s} idx={cycle - 1} startIdx={Math.max(1, cycle - 60)} />)}
          </div>
          <p className="text-[11px] text-muted mt-3">The model uses these 17 inputs (3 operational settings + 14 sensors). Typical healthy range = 1st–99th percentile of training cycles at the RUL cap, so about 2% of healthy readings are flagged 'atypical' by chance. A flag is a cue to look, not a fault. Sparklines show the last 60 cycles.</p>
        </Panel>
        <Panel title="Engine (illustrative 3D model)">
          <div className="h-80 rounded-lg overflow-hidden">
            <Suspense fallback={<div className="text-muted text-sm p-4">Loading 3D view…</div>}>
              <Turbofan3D color={RISK_COLOR[p.risk]} />
            </Suspense>
          </div>
          <p className="text-[11px] text-muted mt-2">Decorative. The colour follows the predicted risk at the replay position.</p>
        </Panel>
      </div>
    </>
  );
}

export function EngineMonitor() {
  const { selectedEngineId, selectEngine } = useStore();
  const fleet = useFleet();
  const engine = useEngine(selectedEngineId);
  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={Gauge} title={`Engine #${selectedEngineId}`} subtitle="Cycle-by-cycle replay of a real FD001 test engine through the served model."
        right={
          <div className="relative">
            <select value={selectedEngineId} onChange={(ev) => selectEngine(Number(ev.target.value))} aria-label="Select engine"
              className="appearance-none glass-panel text-white pl-4 pr-10 py-2.5 outline-none focus:border-primary font-mono text-sm cursor-pointer">
              {(fleet.data ?? Array.from({ length: 100 }, (_, i) => ({ engine_id: i + 1, risk: undefined }))).map((e) => (
                <option key={e.engine_id} value={e.engine_id} className="bg-[#050B14]">Engine #{e.engine_id}{e.risk ? ` · ${e.risk}` : ''}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-primary pointer-events-none" />
          </div>
        } />
      <Loadable q={engine} label="engine history">{(e) => <Replay e={e} />}</Loadable>
      <Note>C-MAPSS test engines stop before failure. The model never saw these engines or their labels during training or checkpoint selection.</Note>
    </div>
  );
}
