import { useRef } from 'react';
import type { Risk, TrajectoryPoint } from '../../services/api';
import { RISK_COLOR } from '../ui';

const SCALE = 150;   // gauge full scale in cycles (interval upper bounds can exceed the 125 cap)

function arc(cx: number, cy: number, r: number, a0: number, a1: number) {
  const p = (a: number) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  const [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M ${x0} ${y0} A ${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`;
}

/** Radial RUL gauge: mean (filled), calibrated interval (band), true RUL (tick). */
export function RulGauge({ mean, lo, hi, truth, risk, size = 190 }: { mean: number; lo: number; hi: number; truth: number; risk: Risk; size?: number }) {
  const A0 = Math.PI * 0.75, A1 = Math.PI * 2.25;            // 270° sweep, gap at the bottom
  const ang = (v: number) => A0 + (A1 - A0) * Math.max(0, Math.min(1, v / SCALE));
  const c = size / 2, r = size / 2 - 16;
  const col = RISK_COLOR[risk];
  const tick = (v: number, len: number, color: string, w = 2) => {
    const a = ang(v);
    return <line key={`${v}-${color}`} x1={c + (r - len) * Math.cos(a)} y1={c + (r - len) * Math.sin(a)} x2={c + (r + len) * Math.cos(a)} y2={c + (r + len) * Math.sin(a)} stroke={color} strokeWidth={w} strokeLinecap="round" />;
  };
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Predicted RUL ${mean.toFixed(0)} cycles, interval ${lo.toFixed(0)} to ${hi.toFixed(0)}, true RUL ${truth.toFixed(0)}`}>
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
      </defs>
      <path d={arc(c, c, r, A0, A1)} stroke="rgba(148,163,184,0.15)" strokeWidth={10} fill="none" strokeLinecap="round" />
      {[0, 25, 50, 75, 100, 125, 150].map((v) => tick(v, 3, 'rgba(148,163,184,0.35)', 1))}
      <path d={arc(c, c, r, ang(lo), Math.max(ang(lo) + 0.001, ang(hi)))} stroke={col} strokeOpacity={0.25} strokeWidth={16} fill="none" />
      <path d={arc(c, c, r, A0, Math.max(A0 + 0.001, ang(mean)))} stroke={col} strokeWidth={6} fill="none" strokeLinecap="round" filter="url(#glow)" />
      {tick(truth, 10, '#F8FAFC', 2.5)}
      <text x={c} y={c - 4} textAnchor="middle" fill="#fff" fontSize={size * 0.22} fontFamily="Space Mono, monospace" fontWeight={700}>{mean.toFixed(0)}</text>
      <text x={c} y={c + 16} textAnchor="middle" fill="#8aa4c0" fontSize={10} letterSpacing={2}>CYCLES LEFT</text>
      <text x={c} y={c + 32} textAnchor="middle" fill={col} fontSize={10} fontFamily="Space Mono, monospace">[{lo.toFixed(0)} – {hi.toFixed(0)}]</text>
      <text x={c} y={size - 6} textAnchor="middle" fill="#cbd5e1" fontSize={9}>▲ white tick = true RUL {truth.toFixed(0)}</text>
    </svg>
  );
}

/** Scrubbable timeline; the track shows the model's risk at every cycle of the engine's history. */
export function RiskTimeline({ trajectory, cycle, onSeek }: { trajectory: TrajectoryPoint[]; cycle: number; onSeek: (c: number) => void }) {
  const ref = useRef<SVGSVGElement>(null);
  const first = trajectory[0].cycle, last = trajectory[trajectory.length - 1].cycle;
  const span = Math.max(1, last - first);
  const x = (c: number) => ((c - first) / span) * 1000;
  const segs: { a: number; b: number; risk: Risk }[] = [];
  for (const p of trajectory) {
    const s = segs[segs.length - 1];
    if (s && s.risk === p.risk) s.b = p.cycle; else segs.push({ a: p.cycle, b: p.cycle, risk: p.risk });
  }
  const seek = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    onSeek(Math.round(first + Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * span));
  };
  return (
    <svg ref={ref} viewBox="0 0 1000 34" preserveAspectRatio="none" className="w-full h-9 cursor-pointer select-none touch-none"
      role="slider" aria-label="Cycle" aria-valuemin={first} aria-valuemax={last} aria-valuenow={cycle} tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'ArrowLeft') onSeek(Math.max(first, cycle - 1)); if (e.key === 'ArrowRight') onSeek(Math.min(last, cycle + 1)); }}
      onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); seek(e.clientX); }}
      onPointerMove={(e) => { if (e.buttons === 1) seek(e.clientX); }}>
      <rect x={0} y={12} width={1000} height={10} rx={5} fill="rgba(148,163,184,0.12)" />
      {segs.map((s, i) => <rect key={i} x={x(s.a)} y={12} width={Math.max(1.5, x(s.b) - x(s.a) + 1000 / span)} height={10} fill={RISK_COLOR[s.risk]} opacity={0.75} />)}
      <rect x={0} y={12} width={x(cycle)} height={10} fill="rgba(255,255,255,0.12)" />
      <line x1={x(cycle)} x2={x(cycle)} y1={1} y2={33} stroke="#fff" strokeWidth={4} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
