import { useMemo, useState } from 'react';
import { Plane, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useFleet } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Kpi, Loadable, PageHeader, Panel, RiskBadge, RISK_COLOR, fmt } from '../components/ui';
import type { FleetRow, Risk } from '../services/api';
import { cn } from '../utils';

const RISKS: Risk[] = ['Critical', 'High', 'Medium', 'Low'];
type SortKey = 'engine_id' | 'rul_lower' | 'rul_mean' | 'last_cycle' | 'abs_err';

export function Fleet() {
  const fleet = useFleet();
  const selectEngine = useStore((s) => s.selectEngine);
  const navigate = useNavigate();
  const [filter, setFilter] = useState<Risk | 'All'>('All');
  const [sort, setSort] = useState<SortKey>('rul_lower');

  const rows = useMemo(() => {
    const d = (fleet.data ?? []).filter((e) => filter === 'All' || e.risk === filter);
    const key = (e: FleetRow) => (sort === 'abs_err' ? -Math.abs(e.rul_mean - e.true_rul) : e[sort]);
    return [...d].sort((a, b) => key(a) - key(b));
  }, [fleet.data, filter, sort]);

  const th = (k: SortKey, label: string) => (
    <th className="pb-3">
      <button onClick={() => setSort(k)} className={cn('uppercase tracking-widest', sort === k ? 'text-primary' : 'hover:text-white')} aria-pressed={sort === k}>
        {label}{sort === k ? ' ▲' : ''}
      </button>
    </th>
  );

  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={Plane} title="Fleet" subtitle="All 100 FD001 test engines at their last observed cycle. Click an engine to replay its history." />
      <Loadable q={fleet} label="fleet">
        {(f) => (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
              {RISKS.map((r) => <Kpi key={r} label={r} value={f.filter((e) => e.risk === r).length} color={RISK_COLOR[r]} />)}
            </div>
            <Panel>
              <div className="flex flex-wrap gap-2 mb-4" role="group" aria-label="Filter by risk">
                {(['All', ...RISKS] as const).map((r) => (
                  <button key={r} onClick={() => setFilter(r)} aria-pressed={filter === r}
                    className={cn('px-3 py-1.5 rounded border text-xs font-bold uppercase tracking-widest',
                      filter === r ? 'border-primary text-primary bg-primary/10' : 'border-white/10 text-muted hover:text-white')}>
                    {r}
                  </button>
                ))}
                <span className="ml-auto text-xs text-muted self-center">{rows.length} engines</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-primary/20 text-[10px] font-bold text-muted">
                      {th('engine_id', 'Engine')}{th('last_cycle', 'Cycles flown')}{th('rul_mean', 'Predicted RUL')}{th('rul_lower', 'Interval')}
                      <th className="pb-3 uppercase tracking-widest">Life remaining</th><th className="pb-3 uppercase tracking-widest">Risk</th>
                      <th className="pb-3 uppercase tracking-widest">True RUL</th>{th('abs_err', 'Error')}<th />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((e) => {
                      const err = e.rul_mean - e.true_rul;
                      return (
                        <tr key={e.engine_id} onClick={() => { selectEngine(e.engine_id); navigate('/engine'); }}
                          className="border-b border-primary/10 hover:bg-primary/5 cursor-pointer group">
                          <td className="py-3 font-mono font-bold text-primary">#{e.engine_id}</td>
                          <td className="py-3 font-mono text-gray-300">{e.last_cycle}</td>
                          <td className="py-3 font-mono font-bold" style={{ color: RISK_COLOR[e.risk] }}>{fmt(e.rul_mean, 0)}</td>
                          <td className="py-3 font-mono text-gray-400">[{fmt(e.rul_lower, 0)}, {fmt(e.rul_upper, 0)}]</td>
                          <td className="py-3">
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-gray-900 rounded-full h-[3px] overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${e.life_remaining_pct}%`, background: RISK_COLOR[e.risk] }} />
                              </div>
                              <span className="font-mono text-xs text-gray-300">{fmt(e.life_remaining_pct, 0)}%</span>
                            </div>
                          </td>
                          <td className="py-3"><RiskBadge risk={e.risk} /></td>
                          <td className="py-3 font-mono text-gray-400">{fmt(e.true_rul, 0)}</td>
                          <td className="py-3 font-mono" style={{ color: err > 0 ? '#FFB300' : '#9fb4c8' }} title={err > 0 ? 'Late (over-predicted)' : 'Early (under-predicted)'}>
                            {err > 0 ? '+' : ''}{fmt(err, 0)}
                          </td>
                          <td className="py-3"><ChevronRight className="w-4 h-4 text-muted group-hover:text-primary" /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-muted mt-4">Life remaining = predicted RUL ÷ 125-cycle cap. Error = predicted − true. Positive means late, which the NASA score penalises more.</p>
            </Panel>
          </>
        )}
      </Loadable>
    </div>
  );
}
