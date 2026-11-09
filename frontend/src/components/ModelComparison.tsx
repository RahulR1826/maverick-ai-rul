import { useMemo } from 'react';
import { CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useCompare } from '../hooks/useData';
import type { EngineRef, Family } from '../services/api';
import { Loadable, RiskBadge, fmt, tooltipStyle } from './ui';

const STYLE: Record<Family, { label: string; color: string }> = {
  fedavg_iid: { label: 'FedAvg · IID window', color: '#00BFFF' },
  fedavg_engine: { label: 'FedAvg · whole-engine', color: '#1baf7a' },
  centralized: { label: 'Centralized (pooled data)', color: '#eb6834' },
};
const ORDER: Family[] = ['fedavg_iid', 'fedavg_engine', 'centralized'];

/** The same engine through the federated and centralized ensembles, against the true RUL. */
export function ModelComparison({ id, cycle }: { id: EngineRef; cycle: number }) {
  const q = useCompare(id);
  const data = useMemo(() => {
    if (!q.data) return [];
    const d = q.data;
    return d.cycles.map((c, i) => {
      const row: Record<string, number | null> = { cycle: c, truth: d.true_rul ? d.true_rul[i] : null };
      for (const f of ORDER) row[f] = d.families[f].rul_mean[i];
      return row;
    });
  }, [q.data]);
  return (
    <Loadable q={q} label="model comparison">
      {(d) => (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-6">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 15 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="cycle" type="number" domain={['dataMin', 'dataMax']} stroke="#6B8CAE" fontSize={11} label={{ value: 'Cycle', position: 'insideBottom', offset: -8, fill: '#6B8CAE', fontSize: 11 }} />
                <YAxis stroke="#6B8CAE" fontSize={11} domain={[0, 'auto']} label={{ value: 'Predicted RUL (cycles)', angle: -90, position: 'insideLeft', fill: '#6B8CAE', fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => [fmt(v as number, 1), String(n)]} labelFormatter={(l) => `Cycle ${l}`} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                {ORDER.map((f) => <Line key={f} dataKey={f} name={STYLE[f].label} stroke={STYLE[f].color} strokeWidth={2} dot={false} isAnimationActive={false} />)}
                {d.true_rul && <Line dataKey="truth" name="True RUL" stroke="#E0E6ED" strokeDasharray="5 4" strokeWidth={1.5} dot={false} isAnimationActive={false} />}
                <ReferenceLine x={cycle} stroke="#FFB300" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {ORDER.map((f) => {
              const r = d.families[f];
              return (
                <div key={f} className="rounded-lg bg-black/30 border border-white/5 p-3">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-2 text-sm font-bold text-white"><span className="w-2.5 h-2.5 rounded-full" style={{ background: STYLE[f].color }} />{STYLE[f].label}</span>
                    <RiskBadge risk={r.risk_last} />
                  </div>
                  <div className="grid grid-cols-3 gap-2 mt-2 text-[11px] font-mono">
                    <div><div className="text-muted">This engine RMSE</div><div className="text-white">{fmt(r.rmse_this_engine_all_cycles, 1)}</div></div>
                    <div><div className="text-muted">Last-cycle error</div><div className="text-white">{r.error_last_cycle === undefined ? '—' : `${r.error_last_cycle > 0 ? '+' : ''}${fmt(r.error_last_cycle, 1)}`}</div></div>
                    <div><div className="text-muted">Fleet RMSE</div><div className="text-white">{fmt(r.fleet_test_rmse_last_cycle, 2)}</div></div>
                  </div>
                </div>
              );
            })}
            <p className="text-[10px] text-muted">"This engine RMSE" covers all of this engine's cycles; one engine says little about a method. "Fleet RMSE" is the ensemble's error on all 100 test engines at their last cycle.</p>
          </div>
        </div>
      )}
    </Loadable>
  );
}
