import { useState } from 'react';
import { Settings as SettingsIcon } from 'lucide-react';
import { useStore, DEFAULT_THRESHOLDS } from '../store/useStore';
import { API_URL } from '../services/api';
import type { Family, Thresholds } from '../services/api';
import { Note, PageHeader, Panel } from '../components/ui';
import { cn } from '../utils';

const FAMILIES: { key: Family; label: string; note: string }[] = [
  { key: 'fedavg_iid', label: 'FedAvg · IID window', note: 'Primary federated model (default).' },
  { key: 'fedavg_engine', label: 'FedAvg · whole-engine', note: 'Federated model where each client owns whole engines (most realistic).' },
  { key: 'centralized', label: 'Centralized', note: 'Trained on pooled data with the same budget, for comparison.' },
];

export function Settings() {
  const { family, thresholds, setFamily, setThresholds, resetSettings } = useStore();
  const [draft, setDraft] = useState<Thresholds>(thresholds);
  const valid = draft.critical >= 0 && draft.critical <= draft.high && draft.high <= draft.medium && draft.medium <= 200;
  const field = (k: keyof Thresholds, label: string) => (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-muted text-xs uppercase tracking-widest">{label} (cycles)</span>
      <input type="number" min={0} max={200} value={draft[k]} onChange={(e) => setDraft({ ...draft, [k]: Number(e.target.value) })}
        className="bg-black/40 border border-primary/20 rounded px-3 py-2 font-mono text-white focus:border-primary outline-none w-32" />
    </label>
  );
  return (
    <div className="space-y-6 pb-20 relative max-w-4xl">
      <PageHeader icon={SettingsIcon} title="Settings" subtitle="Saved in this browser only." />
      <Panel title="Served model">
        <div className="grid gap-2">
          {FAMILIES.map((f) => (
            <button key={f.key} onClick={() => setFamily(f.key)} aria-pressed={family === f.key}
              className={cn('text-left p-3 rounded border', family === f.key ? 'border-primary bg-primary/10' : 'border-white/10 hover:border-primary/40')}>
              <div className={cn('font-bold text-sm', family === f.key ? 'text-primary' : 'text-white')}>{f.label}</div>
              <div className="text-xs text-muted">{f.note}</div>
            </button>
          ))}
        </div>
        <p className="text-[11px] text-muted mt-3">Each option is a 5-seed ensemble of validation-selected checkpoints. Switching loads it on the server (about 10–20 s the first time).</p>
      </Panel>
      <Panel title="Risk thresholds" note="Risk is assigned from the lower end of the predicted RUL interval: below 'critical' → Critical, below 'high' → High, below 'medium' → Medium, otherwise Low.">
        <div className="flex flex-wrap gap-6 items-end">
          {field('critical', 'Critical')}{field('high', 'High')}{field('medium', 'Medium')}
          <button disabled={!valid} onClick={() => setThresholds(draft)}
            className="px-4 py-2 rounded border border-primary text-primary text-sm font-bold disabled:opacity-40">Apply</button>
          <button onClick={() => { resetSettings(); setDraft(DEFAULT_THRESHOLDS); }}
            className="px-4 py-2 rounded border border-white/10 text-muted text-sm hover:text-white">Reset to defaults</button>
        </div>
        {!valid && <p role="alert" className="text-xs text-[#FF6B4A] mt-2">Thresholds must satisfy 0 ≤ critical ≤ high ≤ medium ≤ 200.</p>}
      </Panel>
      <Panel title="Connection">
        <p className="text-sm text-gray-300">API: <span className="font-mono">{API_URL}</span></p>
        <p className="text-[11px] text-muted mt-1">Set VITE_API_URL in frontend/.env to point at another backend.</p>
      </Panel>
      <Note>Privacy features such as differential privacy or secure aggregation are not implemented in this project, so they are not offered here.</Note>
    </div>
  );
}
