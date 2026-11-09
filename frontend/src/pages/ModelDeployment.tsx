import { Server } from 'lucide-react';
import { useMeta, useRuntime } from '../hooks/useData';
import { EXPERIMENT_LABEL, Kpi, Loadable, Note, PageHeader, Panel, fmt } from '../components/ui';

export function ModelDeployment() {
  const meta = useMeta();
  const runtime = useRuntime();
  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={Server} title="Model & deployment" subtitle="What is being served, how its uncertainty is calibrated, and measured costs." />
      <Loadable q={meta} label="model information">
        {(m) => {
          const t = m.model.test;
          return (
            <>
              <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 relative z-10">
                <Kpi label="Ensemble inference (1 window)" value={`${fmt(m.model.latency_ms.ensemble_window_ms_median, 1)} ms`}
                  sub={`median of 30 runs · p95 ${fmt(m.model.latency_ms.ensemble_window_ms_p95, 1)} ms · ${m.model.latency_ms.device}, ${m.model.latency_ms.torch_threads} threads`} />
                <Kpi label="Interval method" value="Conformal" color="#FFB300"
                  sub={`Mondrian, k = ${m.interval.calibration.bins.map((b) => fmt(b.k, 1)).join(' / ')} by RUL bin; ${fmt(m.model.val_coverage * 100, 0)}% on validation`} />
                <Kpi label="Coverage on test engines" value={`${fmt(t.interval_coverage_last_cycle * 100, 0)}%`} color="#00E676"
                  sub={`last cycle · ${fmt(t.interval_coverage_all_cycles * 100, 1)}% over all ${t.n_cycle_predictions.toLocaleString()} cycle predictions`} />
                <Kpi label="Mean interval width" value={`${fmt(t.mean_interval_width_last_cycle, 1)}`} color="#9fb4c8" sub="cycles (last cycle)" />
              </div>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <Panel title="Served ensemble">
                  <dl className="grid grid-cols-[auto,1fr] gap-x-6 gap-y-2 text-sm">
                    <dt className="text-muted">Model family</dt><dd className="text-white">{m.model.description}</dd>
                    <dt className="text-muted">Architecture</dt><dd className="text-white">{m.model.architecture}, 17 inputs, 30-cycle window</dd>
                    <dt className="text-muted">Members</dt><dd className="text-white">{m.model.seeds.length} seeds ({m.model.seeds.join(', ')}), {m.model.checkpoint}</dd>
                    <dt className="text-muted">Checkpoints</dt><dd className="font-mono text-xs text-gray-300">{m.model.checkpoints.map((c) => <div key={c.run_id}>{c.run_id}/selected.pt <span className="text-muted">sha256 {c.sha256}</span></div>)}</dd>
                    <dt className="text-muted">Version</dt><dd className="font-mono text-xs text-gray-300">API {m.version.api} · commit {m.version.git_commit ?? 'unknown'} · cache {m.version.cache} ({m.version.cache_key})</dd>
                    <dt className="text-muted">Explanations</dt><dd className="text-white">{m.model.attributions_precomputed ? 'precomputed for every cycle of all 100 test engines' : 'computed on request (run python -m serving.precompute for live replay colouring)'}</dd>
                    <dt className="text-muted">Interval</dt><dd className="text-white">{m.interval.method}</dd>
                    <dt className="text-muted">Risk rule</dt><dd className="text-white">{m.risk.rule}</dd>
                  </dl>
                </Panel>
                <Panel title="Test-set accuracy of this ensemble (100 engines, last cycle)">
                  <dl className="grid grid-cols-[auto,1fr] gap-x-6 gap-y-2 text-sm">
                    <dt className="text-muted">RMSE / MAE</dt><dd className="font-mono text-white">{fmt(t.ensemble_last_cycle.rmse, 2)} / {fmt(t.ensemble_last_cycle.mae, 2)} cycles</dd>
                    <dt className="text-muted">NASA score</dt><dd className="font-mono text-white">{fmt(t.ensemble_last_cycle.nasa_score, 1)}</dd>
                    <dt className="text-muted">Mean error (bias)</dt><dd className="font-mono text-white">{fmt(t.ensemble_last_cycle.bias_mean_error, 2)} cycles · {fmt(t.ensemble_last_cycle.frac_late * 100, 0)}% late</dd>
                    <dt className="text-muted">Single members</dt><dd className="font-mono text-white">RMSE {fmt(t.members_last_cycle_rmse_mean, 2)} ± {fmt(t.members_last_cycle_rmse_std, 2)} (matches the paper)</dd>
                    <dt className="text-muted">Validation RMSE</dt><dd className="font-mono text-white">{fmt(m.model.val_rmse, 2)} (all validation windows)</dd>
                  </dl>
                </Panel>
              </div>
              <Panel title="Uncertainty calibration: Mondrian conformal vs a single global scale"
                note={`${m.interval.note} Bins are defined on the predicted RUL, which is known at prediction time. Rows show coverage and mean interval width over all ${t.n_cycle_predictions.toLocaleString()} test-cycle predictions.`}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left [&_th]:px-2 [&_td]:px-2">
                    <thead><tr className="text-[10px] uppercase tracking-widest text-muted border-b border-primary/20">
                      <th className="pb-2">Predicted RUL</th><th>k (served)</th><th>Calibration windows</th>
                      <th>Coverage: global → Mondrian</th><th>Width: global → Mondrian (cycles)</th></tr></thead>
                    <tbody>
                      {m.interval.calibration.bins.map((b, i) => {
                        const g = m.interval.evaluation.global.all_cycles.per_bin[i];
                        const md = m.interval.evaluation.mondrian.all_cycles.per_bin[i];
                        const cov = (v: number | null) => (v === null ? '—' : `${fmt(v * 100, 0)}%`);
                        return (
                          <tr key={b.predicted_rul} className="border-b border-primary/10">
                            <td className="py-2 font-mono text-white">{b.predicted_rul}</td>
                            <td className="font-mono">{fmt(b.k, 2)}{b.fallback_to_global ? ' (global)' : ''}</td>
                            <td className="font-mono">{b.n_calibration.toLocaleString()}</td>
                            <td className="font-mono">{cov(g.coverage)} → <span style={{ color: (md.coverage ?? 0) >= 0.9 ? '#00E676' : '#FFB300' }}>{cov(md.coverage)}</span></td>
                            <td className="font-mono">{fmt(g.mean_width, 1)} → {fmt(md.mean_width, 1)}</td>
                          </tr>
                        );
                      })}
                      <tr className="font-bold">
                        <td className="py-2 text-white">All cycles</td><td /><td />
                        <td className="font-mono">{fmt(m.interval.evaluation.global.all_cycles.coverage * 100, 1)}% → {fmt(m.interval.evaluation.mondrian.all_cycles.coverage * 100, 1)}%</td>
                        <td className="font-mono">{fmt(m.interval.evaluation.global.all_cycles.mean_width, 1)} → {fmt(m.interval.evaluation.mondrian.all_cycles.mean_width, 1)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          );
        }}
      </Loadable>
      <Loadable q={runtime} label="runtime benchmark">
        {(rows) => (
          <Panel title="Training cost (dedicated benchmark, seed 42, RTX 4060 Laptop GPU)"
            note="Clients were simulated one after another on one GPU, so FedAvg wall time is the sum of all client work. The parallel estimate (slowest client + aggregation per round) is derived, not measured.">
            <table className="w-full text-sm text-left">
              <thead><tr className="text-[10px] uppercase tracking-widest text-muted border-b border-primary/20">
                <th className="pb-2">Method</th><th>Wall time (s)</th><th>Passes over data</th><th>Client time / round (s)</th><th>Parallel estimate (s)</th><th>Model traffic (MB)</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.experiment} className="border-b border-primary/10">
                    <td className="py-2 text-gray-200">{EXPERIMENT_LABEL[r.experiment] ?? r.experiment}</td>
                    <td className="font-mono">{fmt(r.wall_s, 1)}</td><td className="font-mono">{r.sample_passes}</td>
                    <td className="font-mono">{fmt(r.mean_client_local_train_s, 2)}</td><td className="font-mono">{fmt(r.parallel_estimate_total_s, 1)}</td>
                    <td className="font-mono">{fmt(r.comm_total_MB, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </Loadable>
      <Note>Not deployed to aircraft or edge hardware: this is a research prototype served by one local API process.</Note>
    </div>
  );
}
