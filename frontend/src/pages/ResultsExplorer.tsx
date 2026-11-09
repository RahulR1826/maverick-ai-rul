import { useMemo, useState } from 'react';
import { BarChart3 } from 'lucide-react';
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { usePaired, usePerSeed, useSummary } from '../hooks/useData';
import { EXPERIMENT_LABEL, Loadable, Note, PageHeader, Panel, fmt, pm, tooltipStyle } from '../components/ui';
import type { SummaryRow } from '../services/api';
import { cn } from '../utils';

type Metric = 'rmse' | 'mae' | 'nasa_score';
const METRICS: { key: Metric; label: string }[] = [
  { key: 'rmse', label: 'RMSE (cycles)' }, { key: 'mae', label: 'MAE (cycles)' }, { key: 'nasa_score', label: 'NASA score' },
];
const MAIN = ['centralized', 'centralized_legacy_es', 'ablation_fedavg_k1', 'fedavg_iid', 'local_iid', 'fedavg_engine', 'local_engine'];
const SWEEPS = [
  { exp: 'sweep_rounds', key: 'num_rounds', label: 'Communication rounds R', fixed: 'K=5, E=5, IID window' },
  { exp: 'sweep_local_epochs', key: 'local_epochs', label: 'Local epochs E', fixed: 'K=5, R=20, IID window' },
  { exp: 'sweep_clients', key: 'num_clients', label: 'Clients K (IID window)', fixed: 'R=20, E=5, total data fixed' },
  { exp: 'sweep_clients_engine', key: 'num_clients', label: 'Clients K (whole-engine)', fixed: 'R=20, E=5, total data fixed' },
  { exp: 'fedavg_dirichlet', key: 'alpha', label: 'Dirichlet α', fixed: 'K=5, R=20, E=5' },
] as const;

function Toggle<T extends string>({ value, options, onChange, label }: { value: T; options: { key: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="flex flex-wrap gap-1" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.key} onClick={() => onChange(o.key)} aria-pressed={value === o.key}
          className={cn('px-3 py-1.5 rounded border text-xs font-bold', value === o.key ? 'border-primary text-primary bg-primary/10' : 'border-white/10 text-muted hover:text-white')}>{o.label}</button>
      ))}
    </div>
  );
}

export function ResultsExplorer() {
  const [ck, setCk] = useState<'selected' | 'final'>('selected');
  const [metric, setMetric] = useState<Metric>('rmse');
  const [sweep, setSweep] = useState(0);
  const summary = useSummary(ck);
  const seeds = usePerSeed(ck);
  const paired = usePaired();
  const mLabel = METRICS.find((m) => m.key === metric)!.label;

  const mainData = useMemo(() => {
    if (!summary.data || !seeds.data) return [];
    return MAIN.map((exp, i) => {
      const r = summary.data!.find((x) => x.experiment === exp && x.grid_label === 'default');
      const pts = seeds.data!.filter((s) => s.experiment === exp && s.grid_label === 'default');
      return r ? { i, name: EXPERIMENT_LABEL[exp] ?? exp, mean: r[`${metric}_mean`], sd: r[`${metric}_std`] ?? 0, seeds: pts.map((p) => ({ i: i + (p.seed % 7 - 3) * 0.04, v: p[metric], seed: p.seed })) } : null;
    }).filter(Boolean) as { i: number; name: string; mean: number; sd: number; seeds: { i: number; v: number; seed: number }[] }[];
  }, [summary.data, seeds.data, metric]);

  const sw = SWEEPS[sweep];
  const sweepData = useMemo(() => (summary.data ?? [])
    .filter((r) => r.experiment === sw.exp)
    .map((r) => ({ x: Number(r[sw.key as keyof SummaryRow]), mean: r[`${metric}_mean`], sd: r[`${metric}_std`] ?? 0, n: r.n_seeds, mb: r.comm_total_MB_mean }))
    .sort((a, b) => a.x - b.x), [summary.data, sw, metric]);

  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={BarChart3} title="Results explorer"
        subtitle="The paper's experiments (results/v2): 5 matched seeds per configuration; mean ± SD with every seed shown."
        right={<div className="flex flex-col gap-2 items-end">
          <Toggle label="Checkpoint" value={ck} onChange={setCk} options={[{ key: 'selected', label: 'Validation-selected' }, { key: 'final', label: 'Final round' }]} />
          <Toggle label="Metric" value={metric} onChange={setMetric} options={METRICS} />
        </div>} />

      <Loadable q={{ data: mainData.length ? mainData : undefined, isLoading: summary.isLoading || seeds.isLoading, error: summary.error ?? seeds.error }} label="results">
        {(d) => (
          <Panel title={`Method comparison — ${mLabel}`} note="Bars: mean ± 1 SD over 5 seeds; dots: individual seeds. All methods: same LSTM, data, validation engines and 100 passes over their data.">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart margin={{ top: 10, right: 20, left: 60, bottom: 70 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="i" type="number" domain={[-0.5, d.length - 0.5]} ticks={d.map((x) => x.i)} tickFormatter={(i) => d.find((x) => x.i === i)?.name ?? ''}
                    stroke="#6B8CAE" fontSize={10} angle={-25} textAnchor="end" interval={0} />
                  <YAxis stroke="#6B8CAE" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown) => fmt(v as number, 2)} labelFormatter={() => ''} />
                  {d.map((x) => (
                    <Scatter key={`w${x.i}`} data={[{ i: x.i, mean: x.mean - x.sd }, { i: x.i, mean: x.mean + x.sd }]} dataKey="mean"
                      line={{ stroke: '#00BFFF', strokeWidth: 2 }} shape={() => <g />} legendType="none" tooltipType="none" isAnimationActive={false} />
                  ))}
                  <Scatter data={d} dataKey="mean" fill="#00BFFF" shape="square" name="Mean" isAnimationActive={false} />
                  <Scatter data={d.flatMap((x) => x.seeds)} dataKey="v" fill="#9fb4c8" name="Seed" isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="overflow-x-auto mt-4">
              <table className="w-full text-sm text-left [&_th]:px-2 [&_td]:px-2">
                <thead><tr className="text-[10px] uppercase tracking-widest text-muted border-b border-primary/20">
                  <th className="pb-2">Method</th><th>n</th><th>RMSE</th><th>MAE</th><th>NASA</th><th>Val. RMSE</th><th>Traffic (MB)</th></tr></thead>
                <tbody>
                  {MAIN.map((exp) => summary.data!.find((x) => x.experiment === exp && x.grid_label === 'default')).filter(Boolean).map((r) => (
                    <tr key={r!.experiment} className="border-b border-primary/10">
                      <td className="py-2 text-gray-200">{EXPERIMENT_LABEL[r!.experiment]}</td><td className="font-mono">{r!.n_seeds}</td>
                      <td className="font-mono">{pm(r!.rmse_mean, r!.rmse_std)}</td><td className="font-mono">{pm(r!.mae_mean, r!.mae_std)}</td>
                      <td className="font-mono">{pm(r!.nasa_score_mean, r!.nasa_score_std, 1)}</td><td className="font-mono">{fmt(r!.best_val_rmse_mean, 2)}</td>
                      <td className="font-mono">{fmt(r!.comm_total_MB_mean, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        )}
      </Loadable>

      <Panel title={`Parameter sweeps — ${mLabel}`}>
        <Toggle label="Sweep" value={String(sweep)} onChange={(v) => setSweep(Number(v))} options={SWEEPS.map((s, i) => ({ key: String(i), label: s.label }))} />
        <p className="text-[11px] text-muted mt-2">Fixed: {sw.fixed}. {sw.exp === 'sweep_rounds' || sw.exp === 'sweep_local_epochs' ? 'Total training (R×E) changes with the swept value.' : ''}</p>
        <div className="h-72 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={sweepData} margin={{ top: 10, right: 20, left: 0, bottom: 15 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="x" type="category" allowDuplicatedCategory={false} stroke="#6B8CAE" fontSize={11} label={{ value: sw.label, position: 'insideBottom', offset: -8, fill: '#6B8CAE', fontSize: 11 }} />
              <YAxis stroke="#6B8CAE" fontSize={11} domain={['auto', 'auto']} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => [fmt(v as number, 2), String(n)]} />
              {sweepData.map((p) => (
                <Scatter key={`w${p.x}`} data={[{ x: p.x, mean: p.mean - p.sd }, { x: p.x, mean: p.mean + p.sd }]} dataKey="mean"
                  line={{ stroke: '#00BFFF', strokeWidth: 2 }} shape={() => <g />} legendType="none" tooltipType="none" isAnimationActive={false} />
              ))}
              <Line dataKey="mean" name={mLabel} stroke="#00BFFF" strokeWidth={2} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-muted">{sweepData.map((p) => `${p.x}: ${fmt(p.mean, 2)} ± ${fmt(p.sd, 2)} (${fmt(p.mb, 1)} MB)`).join(' · ')}</p>
      </Panel>

      <Loadable q={paired} label="paired differences">
        {(rows) => (
          <Panel title="Paired differences vs FedAvg · IID window (matched seeds)"
            note="diff = other − FedAvg-IID for the same seed; negative means the other method had lower error. With 5 seeds no difference is claimed as statistically significant (the smallest attainable two-sided p is 0.0625).">
            <table className="w-full text-sm text-left [&_th]:px-2 [&_td]:px-2">
              <thead><tr className="text-[10px] uppercase tracking-widest text-muted border-b border-primary/20"><th className="pb-2">Comparison</th><th>Mean diff ± SD</th><th>Other lower in</th><th>Per-seed</th></tr></thead>
              <tbody>
                {rows.filter((r) => r.checkpoint === ck && r.metric === metric).map((r) => (
                  <tr key={r.comparison} className="border-b border-primary/10">
                    <td className="py-2 text-gray-200">{EXPERIMENT_LABEL[r.comparison.split(' - ')[0]] ?? r.comparison}</td>
                    <td className="font-mono">{r.mean_diff > 0 ? '+' : ''}{pm(r.mean_diff, r.std_diff)}</td>
                    <td className="font-mono">{r.n_other_lower}/{r.n_matched_seeds} seeds</td>
                    <td className="font-mono text-xs text-muted">{r.per_seed_diff}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </Loadable>
      <Note>Source: results/v2/tables. Regenerate with python -m research.aggregate after new runs.</Note>
    </div>
  );
}
