import { motion } from 'framer-motion';
import { Shield, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useFleet, useMeta } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Kpi, Loadable, Note, Panel, RiskBadge, RISK_COLOR, fmt } from '../components/ui';
import { CalibrationPlot, FleetConstellation } from '../components/FleetViz';
import { FederationOrbit } from '../components/FederationOrbit';
import type { FleetRow, Meta } from '../services/api';

function Kpis({ fleet, meta }: { fleet: FleetRow[]; meta: Meta }) {
  const t = meta.model.test;
  const atRisk = fleet.filter((e) => e.risk === 'Critical' || e.risk === 'High').length;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 relative z-10">
      <Kpi label="Engines monitored" value={fleet.length} sub={`${meta.dataset.name} ${meta.dataset.subset} official test fleet`} />
      <Kpi label="Critical / high risk" value={atRisk} color="#FF0055"
        sub="Lower bound of predicted remaining life below the 'high' threshold" />
      <Kpi label="Test RMSE (ensemble)" value={fmt(t.ensemble_last_cycle.rmse, 2)} color="#00E676"
        sub={`cycles · 5 members: ${fmt(t.members_last_cycle_rmse_mean, 2)} ± ${fmt(t.members_last_cycle_rmse_std, 2)}`} />
      <Kpi label="Interval coverage (test)" value={`${fmt(t.interval_coverage_last_cycle * 100, 0)}%`} color="#FFB300"
        sub={`target ${fmt(meta.interval.coverage_target * 100, 0)}% · calibrated on validation engines`} />
    </div>
  );
}

export function Dashboard() {
  const fleet = useFleet();
  const meta = useMeta();
  const selectEngine = useStore((s) => s.selectEngine);
  const navigate = useNavigate();
  const open = (id: number) => { selectEngine(id); navigate('/engine'); };

  return (
    <div className="space-y-8 pb-20 relative">
      <div className="glass-panel p-10 relative overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,191,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(0,191,255,0.05)_1px,transparent_1px)] bg-[size:30px_30px] pointer-events-none opacity-50" />
        <div className="absolute right-0 top-0 w-96 h-96 bg-primary/20 blur-[100px] rounded-full pointer-events-none -translate-y-1/2 translate-x-1/3" />
        <div className="absolute right-6 top-1/2 -translate-y-1/2 hidden xl:block opacity-90"><FederationOrbit /></div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="relative z-10">
          <div className="inline-flex items-center px-3 py-1.5 rounded bg-primary/10 text-primary text-[10px] font-bold tracking-widest uppercase border border-primary/30 mb-4">
            <Shield className="w-3.5 h-3.5 mr-2" /> Federated learning for engine prognostics
          </div>
          <h1 className="text-6xl font-black tracking-tight text-white mb-4 drop-shadow-[0_0_25px_rgba(0,191,255,0.35)]">
            MAVERICK <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">AI</span>
          </h1>
          <p className="text-lg text-gray-400 max-w-3xl leading-relaxed">
            Remaining-useful-life prediction from an LSTM trained with Federated Averaging, served as a 5-seed ensemble with
            calibrated uncertainty. Every number on this dashboard is computed from the trained models and the NASA C-MAPSS data.
          </p>
        </motion.div>
      </div>

      <Loadable q={{ data: fleet.data && meta.data ? { f: fleet.data, m: meta.data } : undefined, isLoading: fleet.isLoading || meta.isLoading, error: fleet.error ?? meta.error }} label="fleet">
        {({ f, m }) => (
          <>
            <Kpis fleet={f} meta={m} />
            <div className="grid grid-cols-1 2xl:grid-cols-[1.45fr_1fr] gap-6 relative z-10">
              <Panel title="Fleet constellation · 100 test engines"><FleetConstellation fleet={f} onOpen={open} /></Panel>
              <Panel title="Does the model know? Predicted vs true RUL"><CalibrationPlot fleet={f} /></Panel>
            </div>
            <Panel title="Highest-risk engines (ranked by lower remaining-life bound)"
              note={`Remaining life in cycles: ensemble mean with a ${fmt(m.interval.coverage_target * 100, 0)}% interval. True RUL comes from RUL_FD001.txt and is shown for evaluation only.`}>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-primary/20 text-[10px] font-bold uppercase tracking-widest text-muted">
                      <th className="pb-3">Engine</th><th className="pb-3">Cycles flown</th><th className="pb-3">Predicted RUL</th>
                      <th className="pb-3">Interval</th><th className="pb-3">Risk</th><th className="pb-3">True RUL</th><th />
                    </tr>
                  </thead>
                  <tbody>
                    {[...f].sort((a, b) => a.rul_lower - b.rul_lower || a.rul_mean - b.rul_mean).slice(0, 10).map((e) => (
                      <tr key={e.engine_id} onClick={() => open(e.engine_id)} className="border-b border-primary/10 hover:bg-primary/5 cursor-pointer group">
                        <td className="py-3 font-mono text-primary font-bold">#{e.engine_id}</td>
                        <td className="py-3 font-mono text-gray-300">{e.last_cycle}</td>
                        <td className="py-3 font-mono font-bold" style={{ color: RISK_COLOR[e.risk] }}>{fmt(e.rul_mean, 0)} cyc</td>
                        <td className="py-3 font-mono text-gray-400">[{fmt(e.rul_lower, 0)}, {fmt(e.rul_upper, 0)}]</td>
                        <td className="py-3"><RiskBadge risk={e.risk} /></td>
                        <td className="py-3 font-mono text-gray-400">{fmt(e.true_rul, 0)}</td>
                        <td className="py-3"><ChevronRight className="w-4 h-4 text-muted group-hover:text-primary" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        )}
      </Loadable>
      <Note>Engines are the 100 test engines of C-MAPSS FD001 (simulated turbofan run-to-failure data from NASA). Each engine is shown at its last observed cycle; use the Engine Monitor to replay its history.</Note>
    </div>
  );
}
