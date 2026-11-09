import type { ComponentType, ReactNode } from 'react';
import { AlertOctagon, AlertTriangle, CheckCircle2, Info, Loader2, ShieldAlert } from 'lucide-react';
import { cn } from '../utils';
import { errorMessage } from '../services/api';
import type { Risk } from '../services/api';

export const RISK_COLOR: Record<Risk, string> = {
  Critical: '#FF0055', High: '#FF6B4A', Medium: '#FFB300', Low: '#00E676',
};

const RISK_ICON = { Critical: AlertOctagon, High: ShieldAlert, Medium: AlertTriangle, Low: CheckCircle2 } as const;

/** Risk is conveyed by icon + text, never by colour alone. */
export function RiskBadge({ risk }: { risk: Risk }) {
  const Icon = RISK_ICON[risk];
  const c = RISK_COLOR[risk];
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded border text-[10px] font-bold uppercase tracking-widest"
      style={{ color: c, borderColor: `${c}55`, background: `${c}14` }}>
      <Icon className="w-3.5 h-3.5" aria-hidden /> {risk}
    </span>
  );
}

export function PageHeader({ icon: Icon, title, subtitle, right }: { icon: ComponentType<{ className?: string }>; title: string; subtitle: string; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 relative z-10">
      <div>
        <h2 className="text-3xl font-bold text-white flex items-center mb-1 drop-shadow-md">
          <Icon className="w-8 h-8 mr-3 text-primary" /> {title}
        </h2>
        <p className="text-muted text-sm max-w-3xl">{subtitle}</p>
      </div>
      {right}
    </div>
  );
}

export function Panel({ title, children, className, note }: { title?: string; children: ReactNode; className?: string; note?: ReactNode }) {
  return (
    <section className={cn('glass-panel p-6 relative z-10', className)}>
      {title && (
        <h3 className="text-xs font-bold text-white uppercase tracking-widest mb-4 flex items-center">
          <span className="w-1.5 h-1.5 rounded-full bg-primary mr-2 shadow-[0_0_5px_#00BFFF]" />{title}
        </h3>
      )}
      {children}
      {note && <p className="text-[11px] text-muted mt-4 leading-relaxed">{note}</p>}
    </section>
  );
}

export function Kpi({ label, value, sub, color = '#00BFFF' }: { label: string; value: ReactNode; sub?: ReactNode; color?: string }) {
  return (
    <div className="glass-panel p-5 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full" style={{ background: color, boxShadow: `0 0 12px ${color}` }} />
      <p className="text-[10px] font-bold text-muted uppercase tracking-widest">{label}</p>
      <div className="text-3xl font-mono font-bold mt-1 tabular-nums" style={{ color }}>{value}</div>
      {sub && <p className="text-[11px] text-muted mt-1">{sub}</p>}
    </div>
  );
}

/** Wraps a react-query result: shows loading / error states instead of placeholder numbers. */
export function Loadable<T>({ q, children, label = 'data' }: { q: { data?: T; isLoading: boolean; error: unknown }; children: (d: T) => ReactNode; label?: string }) {
  if (q.isLoading) {
    return <div className="flex items-center text-muted text-sm py-8 justify-center"><Loader2 className="w-4 h-4 mr-2 animate-spin" />Loading {label}… (the first request loads the model ensemble, ~10–20 s)</div>;
  }
  if (q.error || q.data === undefined) {
    return (
      <div role="alert" className="flex items-start gap-2 text-sm text-[#FF6B4A] border border-[#FF6B4A]/30 bg-[#FF6B4A]/10 rounded-lg p-4">
        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" /> {q.error ? errorMessage(q.error) : `No ${label}.`}
      </div>
    );
  }
  return <>{children(q.data)}</>;
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 text-[11px] text-muted border border-primary/15 bg-primary/5 rounded-lg px-3 py-2 relative z-10">
      <Info className="w-3.5 h-3.5 mt-0.5 shrink-0 text-primary" /> <div>{children}</div>
    </div>
  );
}

export const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(d));
export const pm = (m: number | null | undefined, s: number | null | undefined, d = 2) => (s === null || s === undefined ? fmt(m, d) : `${fmt(m, d)} ± ${fmt(s, d)}`);

export const EXPERIMENT_LABEL: Record<string, string> = {
  centralized: 'Centralized (budget-matched)', centralized_legacy_es: 'Centralized (original early stopping)',
  fedavg_iid: 'FedAvg · IID window', local_iid: 'Local-only · IID window', fedavg_engine: 'FedAvg · whole-engine',
  local_engine: 'Local-only · whole-engine', ablation_fedavg_k1: 'Ablation: FedAvg K=1', fedavg_dirichlet: 'FedAvg · Dirichlet',
};

export const tooltipStyle = { background: 'rgba(5,11,20,0.95)', border: '1px solid rgba(0,191,255,0.3)', borderRadius: 8, fontSize: 12 };
