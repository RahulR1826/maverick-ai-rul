import { useState } from 'react';
import { motion } from 'framer-motion';
import type { FleetRow } from '../services/api';
import { RISK_COLOR, fmt } from './ui';

/** 100 engines as tiles: colour = risk, fill = predicted life remaining, glow = urgency. */
export function FleetConstellation({ fleet, onOpen }: { fleet: FleetRow[]; onOpen: (id: number) => void }) {
  const [hover, setHover] = useState<FleetRow | null>(null);
  const rows = [...fleet].sort((a, b) => a.engine_id - b.engine_id);
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-5">
      <div className="grid grid-cols-10 gap-1.5" role="list" aria-label="Fleet of 100 engines">
        {rows.map((e, i) => {
          const c = RISK_COLOR[e.risk];
          const urgent = e.risk === 'Critical';
          return (
            <motion.button key={e.engine_id} role="listitem" onClick={() => onOpen(e.engine_id)}
              onMouseEnter={() => setHover(e)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(e)}
              initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.006, type: 'spring', stiffness: 260, damping: 22 }}
              whileHover={{ scale: 1.18, zIndex: 10 }}
              aria-label={`Engine ${e.engine_id}, ${e.risk}, predicted ${e.rul_mean.toFixed(0)} cycles`}
              className="relative aspect-square rounded-md overflow-hidden border text-[9px] font-mono font-bold"
              style={{ borderColor: `${c}66`, background: `${c}14`, boxShadow: urgent ? `0 0 14px ${c}88` : 'none' }}>
              <span className="absolute inset-x-0 bottom-0" style={{ height: `${e.life_remaining_pct}%`, background: `linear-gradient(180deg, ${c}aa, ${c}33)` }} />
              {urgent && <motion.span className="absolute inset-0 rounded-md border" style={{ borderColor: c }}
                animate={{ opacity: [0.9, 0, 0.9] }} transition={{ duration: 1.6, repeat: Infinity, delay: (i % 7) * 0.2 }} />}
              <span className="relative text-white/90 drop-shadow">{e.engine_id}</span>
            </motion.button>
          );
        })}
      </div>
      <div className="glass-panel-sm p-4 text-xs self-start min-h-[180px]">
        {hover ? (
          <motion.div key={hover.engine_id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-[10px] text-muted uppercase tracking-widest">Engine</div>
            <div className="text-2xl font-black text-white">#{hover.engine_id}</div>
            <div className="mt-2 space-y-1 font-mono">
              <div className="flex justify-between"><span className="text-muted">Predicted</span><span style={{ color: RISK_COLOR[hover.risk] }}>{fmt(hover.rul_mean, 0)} cyc</span></div>
              <div className="flex justify-between"><span className="text-muted">Interval</span><span>{fmt(hover.rul_lower, 0)}–{fmt(hover.rul_upper, 0)}</span></div>
              <div className="flex justify-between"><span className="text-muted">True RUL</span><span>{fmt(hover.true_rul, 0)}</span></div>
              <div className="flex justify-between"><span className="text-muted">Cycles flown</span><span>{hover.last_cycle}</span></div>
              <div className="flex justify-between"><span className="text-muted">Risk</span><span style={{ color: RISK_COLOR[hover.risk] }}>{hover.risk}</span></div>
            </div>
            <div className="text-[10px] text-primary mt-3">Click to open the digital twin →</div>
          </motion.div>
        ) : (
          <div className="text-muted leading-relaxed">
            Hover an engine for details; click to open its digital twin.<br /><br />
            Tile colour = risk (from the lower RUL bound) · fill height = predicted life remaining · pulsing border = critical.
          </div>
        )}
      </div>
    </div>
  );
}

/** Predicted vs true RUL for the whole fleet, with each engine's interval. */
export function CalibrationPlot({ fleet }: { fleet: FleetRow[] }) {
  const [hover, setHover] = useState<FleetRow | null>(null);
  const W = 520, H = 360, L = 46, R = 14, T = 14, B = 40, MAX = 150;
  const x = (v: number) => L + (Math.min(v, MAX) / MAX) * (W - L - R);
  const y = (v: number) => H - B - (Math.min(v, MAX) / MAX) * (H - T - B);
  const covered = fleet.filter((e) => e.true_rul >= e.rul_lower && e.true_rul <= e.rul_upper).length;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Predicted versus true remaining useful life for 100 test engines">
        {[0, 25, 50, 75, 100, 125, 150].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={T} y2={H - B} stroke="rgba(255,255,255,0.05)" />
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="rgba(255,255,255,0.05)" />
            <text x={x(t)} y={H - B + 16} fill="#6B8CAE" fontSize={10} textAnchor="middle">{t}</text>
            <text x={L - 8} y={y(t) + 3} fill="#6B8CAE" fontSize={10} textAnchor="end">{t}</text>
          </g>
        ))}
        <polygon points={`${x(0)},${y(10)} ${x(MAX - 10)},${y(MAX)} ${x(MAX)},${y(MAX)} ${x(MAX)},${y(MAX - 10)} ${x(10)},${y(0)} ${x(0)},${y(0)}`} fill="rgba(0,191,255,0.07)" />
        <line x1={x(0)} y1={y(0)} x2={x(MAX)} y2={y(MAX)} stroke="#E0E6ED" strokeDasharray="5 4" strokeOpacity={0.6} />
        {fleet.map((e) => (
          <g key={e.engine_id} onMouseEnter={() => setHover(e)} onMouseLeave={() => setHover(null)} className="cursor-default">
            <line x1={x(e.true_rul)} x2={x(e.true_rul)} y1={y(e.rul_lower)} y2={y(e.rul_upper)} stroke={RISK_COLOR[e.risk]} strokeOpacity={hover?.engine_id === e.engine_id ? 0.9 : 0.25} strokeWidth={2} />
            <circle cx={x(e.true_rul)} cy={y(e.rul_mean)} r={hover?.engine_id === e.engine_id ? 6 : 3.6} fill={RISK_COLOR[e.risk]} stroke="#030712" strokeWidth={1} />
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} fill="#9fb4c8" fontSize={11} textAnchor="middle">True RUL (cycles)</text>
        <text x={12} y={(T + H - B) / 2} fill="#9fb4c8" fontSize={11} textAnchor="middle" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>Predicted RUL (cycles)</text>
        {hover && (
          <g transform={`translate(${Math.min(x(hover.true_rul) + 10, W - 150)},${Math.max(y(hover.rul_mean) - 50, T)})`}>
            <rect width={140} height={46} rx={6} fill="rgba(3,8,16,0.92)" stroke="rgba(0,191,255,0.4)" />
            <text x={8} y={16} fill="#fff" fontSize={11} fontWeight={700}>Engine #{hover.engine_id}</text>
            <text x={8} y={31} fill="#9fb4c8" fontSize={10}>pred {hover.rul_mean.toFixed(0)} [{hover.rul_lower.toFixed(0)}–{hover.rul_upper.toFixed(0)}]</text>
            <text x={8} y={42} fill="#9fb4c8" fontSize={10}>true {hover.true_rul.toFixed(0)}</text>
          </g>
        )}
      </svg>
      <p className="text-[11px] text-muted">Each dot is one test engine at its last cycle; the vertical whisker is its calibrated interval. Dashed line: perfect prediction · shaded band: within ±10 cycles. {covered}/{fleet.length} intervals contain the true RUL.</p>
    </div>
  );
}
