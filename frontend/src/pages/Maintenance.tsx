import { PenTool } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useFleet } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Loadable, Note, PageHeader, Panel, RiskBadge, RISK_COLOR, fmt } from '../components/ui';
import type { FleetRow, Risk } from '../services/api';

const BUCKETS: { risk: Risk; title: string; action: string }[] = [
  { risk: 'Critical', title: 'Act now', action: 'Lower RUL bound below the critical threshold: remove from service or inspect before next flight.' },
  { risk: 'High', title: 'Schedule soon', action: 'Plan a shop visit before the lower bound reaches the critical threshold.' },
  { risk: 'Medium', title: 'Plan', action: 'Include in the next maintenance planning cycle.' },
  { risk: 'Low', title: 'Monitor', action: 'No action; keep monitoring.' },
];

export function Maintenance() {
  const fleet = useFleet();
  const th = useStore((s) => s.thresholds);
  const selectEngine = useStore((s) => s.selectEngine);
  const navigate = useNavigate();
  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={PenTool} title="Maintenance planner"
        subtitle={`Engines are grouped by the lower end of their predicted RUL interval: critical < ${th.critical}, high < ${th.high}, medium < ${th.medium} cycles. Change thresholds in Settings.`} />
      <Loadable q={fleet} label="fleet">
        {(f) => (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {BUCKETS.map((b) => {
              const rows: FleetRow[] = f.filter((e) => e.risk === b.risk).sort((a, c) => a.rul_lower - c.rul_lower || a.rul_mean - c.rul_mean);
              return (
                <Panel key={b.risk} title={`${b.title} · ${rows.length} engines`}>
                  <div className="flex items-center gap-2 mb-3"><RiskBadge risk={b.risk} /><span className="text-xs text-muted">{b.action}</span></div>
                  {rows.length === 0 ? <p className="text-sm text-muted">None.</p> : (
                    <ul className="divide-y divide-primary/10 max-h-72 overflow-y-auto pr-1">
                      {rows.map((e) => (
                        <li key={e.engine_id}>
                          <button onClick={() => { selectEngine(e.engine_id); navigate('/engine'); }}
                            className="w-full flex items-center justify-between py-2 text-sm hover:bg-primary/5 px-2 rounded text-left">
                            <span className="font-mono text-primary font-bold">#{e.engine_id}</span>
                            <span className="font-mono" style={{ color: RISK_COLOR[e.risk] }}>{fmt(e.rul_mean, 0)} cyc</span>
                            <span className="font-mono text-muted text-xs">worst case {fmt(e.rul_lower, 0)} cyc</span>
                            {b.risk === 'High' && <span className="text-xs text-gray-300">≈{fmt(Math.max(0, e.rul_lower - th.critical), 0)} cycles until critical (worst case)</span>}
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </Panel>
              );
            })}
          </div>
        )}
      </Loadable>
      <Note>Planning on the lower bound is deliberately conservative: the model over-predicts RUL on average (late predictions are the costly direction). This is decision support from a research model, not a certified maintenance programme.</Note>
    </div>
  );
}
