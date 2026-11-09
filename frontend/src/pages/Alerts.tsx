import { AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useFleet } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Loadable, Note, PageHeader, Panel, RiskBadge, fmt } from '../components/ui';

export function Alerts() {
  const fleet = useFleet();
  const th = useStore((s) => s.thresholds);
  const selectEngine = useStore((s) => s.selectEngine);
  const setCycle = useStore((s) => s.setCycle);
  const navigate = useNavigate();
  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={AlertTriangle} title="Alerts"
        subtitle={`Raised when an engine's lower RUL bound falls below the high (${th.high}) or critical (${th.critical}) threshold and stays below it.`} />
      <Loadable q={fleet} label="alerts">
        {(f) => {
          const alerts = f.filter((e) => e.risk === 'Critical' || e.risk === 'High')
            .sort((a, b) => (a.risk === b.risk ? a.rul_lower - b.rul_lower || a.rul_mean - b.rul_mean : a.risk === 'Critical' ? -1 : 1));
          return (
            <Panel title={`${alerts.length} active alerts`}>
              {alerts.length === 0 ? <p className="text-sm text-muted">No engine is above the high-risk threshold.</p> : (
                <ul className="space-y-2">
                  {alerts.map((e) => {
                    const since = e.risk === 'Critical' ? e.critical_since_cycle : e.high_since_cycle;
                    return (
                      <li key={e.engine_id} className="glass-panel-sm p-3 flex flex-wrap items-center gap-4">
                        <RiskBadge risk={e.risk} />
                        <span className="font-mono font-bold text-primary">Engine #{e.engine_id}</span>
                        <span className="text-sm text-gray-300">
                          Predicted RUL {fmt(e.rul_mean, 0)} cycles (interval {fmt(e.rul_lower, 0)}–{fmt(e.rul_upper, 0)})
                          {since !== null && <> · below {e.risk === 'Critical' ? 'critical' : 'high'} threshold since cycle {since} ({e.last_cycle - since} cycles ago)</>}
                        </span>
                        <button className="ml-auto px-3 py-1.5 rounded border border-primary text-primary text-xs font-bold"
                          onClick={() => { selectEngine(e.engine_id); if (since !== null) setCycle(e.engine_id, since); navigate('/engine'); }}>
                          {since !== null ? 'Replay from alert' : 'Open'}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </Panel>
          );
        }}
      </Loadable>
      <Note>Alerts are derived from the served predictions; no notification is sent anywhere.</Note>
    </div>
  );
}
