import { useMemo, useState } from 'react';
import { Network } from 'lucide-react';
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useConvergence, usePartitions, useRuntime, useSummary } from '../hooks/useData';
import { Kpi, Loadable, Note, PageHeader, Panel, fmt, pm, tooltipStyle } from '../components/ui';
import type { ConvergenceRow } from '../services/api';
import { cn } from '../utils';

const SERIES = [
  { exp: 'fedavg_iid', grid: 'default', label: 'FedAvg · IID window', color: '#00BFFF' },
  { exp: 'fedavg_engine', grid: 'default', label: 'FedAvg · whole-engine', color: '#1baf7a' },
  { exp: 'centralized', grid: 'default', label: 'Centralized (per 5 epochs)', color: '#eb6834' },
];
const HET = [
  { exp: 'fedavg_iid', grid: 'default', label: 'IID', color: '#00BFFF' },
  { exp: 'fedavg_dirichlet', grid: 'dirichlet_alpha=1.0', label: 'α = 1.0', color: '#eb6834' },
  { exp: 'fedavg_dirichlet', grid: 'dirichlet_alpha=0.5', label: 'α = 0.5', color: '#1baf7a' },
  { exp: 'fedavg_dirichlet', grid: 'dirichlet_alpha=0.1', label: 'α = 0.1', color: '#eda100' },
];
const PARTS = [
  { exp: 'fedavg_iid', grid: 'default', label: 'IID window' },
  { exp: 'fedavg_dirichlet', grid: 'dirichlet_alpha=1.0', label: 'Dirichlet α=1.0' },
  { exp: 'fedavg_dirichlet', grid: 'dirichlet_alpha=0.5', label: 'Dirichlet α=0.5' },
  { exp: 'fedavg_dirichlet', grid: 'dirichlet_alpha=0.1', label: 'Dirichlet α=0.1' },
  { exp: 'fedavg_engine', grid: 'default', label: 'Whole-engine' },
];

function seriesByRound(rows: ConvergenceRow[], defs: typeof SERIES) {
  const byRound: Record<number, Record<string, number | number[]>> = {};
  for (const d of defs) {
    for (const r of rows.filter((x) => x.experiment === d.exp && x.grid_label === d.grid)) {
      const round = r.step_type === 'epoch' ? (r.step % 5 === 0 ? r.step / 5 : null) : r.step;
      if (round === null) continue;
      byRound[round] ??= { round };
      byRound[round][d.label] = r.mean;
      byRound[round][`${d.label}__band`] = [r.mean - (r.std ?? 0), r.mean + (r.std ?? 0)];
    }
  }
  return Object.values(byRound).sort((a, b) => (a.round as number) - (b.round as number));
}

function ConvChart({ rows, defs }: { rows: ConvergenceRow[]; defs: typeof SERIES }) {
  const data = useMemo(() => seriesByRound(rows, defs), [rows, defs]);
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 15 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="round" stroke="#6B8CAE" fontSize={11} label={{ value: 'Communication round', position: 'insideBottom', offset: -8, fill: '#6B8CAE', fontSize: 11 }} />
          <YAxis stroke="#6B8CAE" fontSize={11} domain={[10, 30]} allowDataOverflow label={{ value: 'Val. RMSE (cycles)', angle: -90, position: 'insideLeft', fill: '#6B8CAE', fontSize: 11 }} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => (Array.isArray(v) ? [null, null] : [fmt(v as number, 2), String(n)])} />
          <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
          {defs.map((d) => <Area key={`${d.label}b`} dataKey={`${d.label}__band`} stroke="none" fill={d.color} fillOpacity={0.12} legendType="none" isAnimationActive={false} />)}
          {defs.map((d) => <Line key={d.label} dataKey={d.label} stroke={d.color} dot={false} strokeWidth={2} isAnimationActive={false} />)}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function FederatedLearning() {
  const conv = useConvergence();
  const summary = useSummary('selected');
  const runtime = useRuntime();
  const [part, setPart] = useState(0);
  const parts = usePartitions(42);
  const row = (exp: string) => summary.data?.find((r) => r.experiment === exp && r.grid_label === 'default');
  const payload = runtime.data?.find((r) => r.experiment === 'fedavg_iid')?.payload_bytes;

  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={Network} title="Federated learning"
        subtitle="How the served model was trained: FedAvg with K=5 clients, R=20 rounds and E=5 local epochs. Results are the mean over 5 seeds from results/v2." />
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 relative z-10">
        <Kpi label="FedAvg · IID window" value={pm(row('fedavg_iid')?.rmse_mean, row('fedavg_iid')?.rmse_std)} sub="test RMSE, cycles" color="#00BFFF" />
        <Kpi label="Centralized (pooled data)" value={pm(row('centralized')?.rmse_mean, row('centralized')?.rmse_std)} sub="same budget, same model" color="#eb6834" />
        <Kpi label="Local-only · own 18 engines" value={pm(row('local_engine')?.rmse_mean, row('local_engine')?.rmse_std)} sub="what an airline gets alone" color="#FF0055" />
        <Kpi label="Model traffic per training run" value={`${fmt(row('fedavg_iid')?.comm_total_MB_mean, 1)} MB`} color="#FFB300"
          sub={payload ? `${payload.toLocaleString()} B per transfer × 2 × 5 clients × 20 rounds (payload estimate)` : 'payload estimate'} />
      </div>

      <Loadable q={conv} label="convergence">
        {(rows) => (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Panel title="Convergence at equal training budget" note="Centralized validation error is lowest around epoch 45 and then rises (overfitting); FedAvg keeps improving. Band: ± 1 SD over 5 seeds. Y-axis clipped at 30.">
              <ConvChart rows={rows} defs={SERIES} />
            </Panel>
            <Panel title="Label skew across clients (Dirichlet α)" note="Smaller α = more skewed clients. α = 0.1 also makes client sizes very unequal (see the client chart below), so its effect is not label skew alone.">
              <ConvChart rows={rows} defs={HET} />
            </Panel>
          </div>
        )}
      </Loadable>

      <Panel title="Client data in each partition (seed 42)">
        <div className="flex flex-wrap gap-2 mb-4" role="group" aria-label="Partition">
          {PARTS.map((p, i) => (
            <button key={p.label} onClick={() => setPart(i)} aria-pressed={part === i}
              className={cn('px-3 py-1.5 rounded border text-xs font-bold', part === i ? 'border-primary text-primary bg-primary/10' : 'border-white/10 text-muted hover:text-white')}>{p.label}</button>
          ))}
        </div>
        <Loadable q={parts} label="partitions">
          {(rows) => {
            const d = rows.filter((r) => r.experiment === PARTS[part].exp && r.grid_label === PARTS[part].grid)
              .map((r) => ({ client: `Client ${r.client_id}`, windows: r.n_windows, rul: r.rul_mean, engines: r.n_engines, exclusive: r.n_engines_exclusive }));
            return (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={d}><CartesianGrid stroke="rgba(255,255,255,0.06)" /><XAxis dataKey="client" stroke="#6B8CAE" fontSize={11} /><YAxis stroke="#6B8CAE" fontSize={11} />
                      <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => [String(v), String(n)]} />
                      <Bar dataKey="windows" name="Training windows" fill="#00BFFF" isAnimationActive={false} /></BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={d}><CartesianGrid stroke="rgba(255,255,255,0.06)" /><XAxis dataKey="client" stroke="#6B8CAE" fontSize={11} /><YAxis stroke="#6B8CAE" fontSize={11} domain={[0, 125]} />
                      <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown) => [fmt(v as number, 1), 'Mean RUL label']} />
                      <Bar dataKey="rul" name="Mean RUL label" fill="#1baf7a" isAnimationActive={false} /></BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-[11px] text-muted xl:col-span-2">
                  Engines per client: {d.map((x) => `${x.engines}${x.exclusive === x.engines ? ' (all owned)' : ''}`).join(' · ')}.
                  Window-level partitions spread every engine across all clients; whole-engine gives each client complete trajectories.
                </p>
              </div>
            );
          }}
        </Loadable>
      </Panel>
      <Note>Simulated federation: clients ran in one process and no network was used. Traffic is the measured serialized payload size, excluding protocol overhead. Keeping raw data on clients is not a formal privacy guarantee; no differential privacy or secure aggregation is implemented.</Note>
    </div>
  );
}
