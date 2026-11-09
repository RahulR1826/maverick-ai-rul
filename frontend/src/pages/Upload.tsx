import { useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FileUp, Loader2, ShieldAlert, CheckCircle2, ArrowRight, UploadCloud } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { api, errorMessage } from '../services/api';
import type { UploadResponse } from '../services/api';
import { useUploads } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Note, PageHeader, Panel, fmt } from '../components/ui';
import { cn } from '../utils';

const MAX_BYTES = 2 * 1024 * 1024;

export function Upload() {
  const [drag, setDrag] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [rul, setRul] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const uploads = useUploads();
  const selectEngine = useStore((s) => s.selectEngine);
  const navigate = useNavigate();

  const pick = (f: File | undefined | null) => {
    setResult(null); setError('');
    if (!f) return;
    if (f.size > MAX_BYTES) { setError(`File is ${fmt(f.size / 1024 / 1024, 1)} MB; the limit is 2 MB.`); return; }
    setFile(f);
  };
  const submit = async () => {
    if (!file) return;
    setBusy(true); setError(''); setResult(null);
    try {
      const r = await api.upload(file, rul.trim() === '' ? undefined : Number(rul));
      setResult(r);
      await qc.invalidateQueries({ queryKey: ['uploads'] });
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const open = (id: string) => { selectEngine(id); navigate('/engine'); };

  return (
    <div className="space-y-6 pb-20 relative max-w-5xl">
      <PageHeader icon={UploadCloud} title="Analyse your own engine"
        subtitle="Upload run data in NASA's C-MAPSS text format and get predictions, uncertainty, explanations and a digital twin, from the same models." />
      <Panel>
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
          onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') input.current?.click(); }}
          aria-label="Choose or drop an engine data file"
          className={cn('relative rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all overflow-hidden',
            drag ? 'border-primary bg-primary/10 shadow-[0_0_40px_rgba(0,191,255,0.25)]' : 'border-cyan-400/25 hover:border-primary/60 hover:bg-primary/5')}>
          <motion.div animate={{ y: drag ? -6 : 0, scale: drag ? 1.08 : 1 }} className="inline-flex p-4 rounded-2xl bg-primary/10 border border-primary/30 mb-4">
            <FileUp className="w-10 h-10 text-primary" />
          </motion.div>
          <div className="text-lg font-bold text-white">{file ? file.name : 'Drop a file here or click to browse'}</div>
          <div className="text-sm text-muted mt-1">{file ? `${fmt(file.size / 1024, 1)} KB` : '.txt or .csv · 26 columns per row: unit, cycle, 3 settings, 21 sensors · up to 10 engines · 2 MB'}</div>
          <input ref={input} type="file" accept=".txt,.csv,text/plain,text/csv" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
        </div>
        <div className="flex flex-wrap items-end gap-4 mt-5">
          <label className="text-xs"><span className="text-muted uppercase tracking-widest">Known RUL at the last cycle (optional, single engine)</span>
            <input type="number" min={0} value={rul} onChange={(e) => setRul(e.target.value)} placeholder="e.g. 112"
              className="mt-1 block w-56 bg-black/40 border border-primary/20 rounded px-3 py-2 font-mono text-white" /></label>
          <button onClick={submit} disabled={!file || busy}
            className="px-6 py-2.5 rounded-lg bg-primary text-[#030712] font-black disabled:opacity-40 shadow-[0_0_20px_rgba(0,191,255,0.4)] flex items-center gap-2">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />} Analyse
          </button>
        </div>
        <AnimatePresence>
          {error && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert"
            className="mt-4 text-sm text-[#FF6B4A] border border-[#FF6B4A]/30 bg-[#FF6B4A]/10 rounded-lg p-3">{error}</motion.div>}
        </AnimatePresence>
      </Panel>

      {result && (
        <Panel title={`${result.uploads.length} engine${result.uploads.length > 1 ? 's' : ''} analysed`}>
          {result.warning && (
            <div className="flex items-start gap-2 text-sm text-[#FFB300] border border-[#FFB300]/40 bg-[#FFB300]/10 rounded-lg p-3 mb-3">
              <ShieldAlert className="w-4 h-4 mt-0.5 shrink-0" />{result.warning}
            </div>
          )}
          <ul className="space-y-2">
            {result.uploads.map((u) => (
              <motion.li key={u.upload_id} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                className="glass-panel-sm p-3 flex flex-wrap items-center gap-4">
                {u.max_ood_fraction > 0 ? <ShieldAlert className="w-5 h-5 text-[#FFB300]" /> : <CheckCircle2 className="w-5 h-5 text-[#00E676]" />}
                <div className="flex-1 min-w-[200px]">
                  <div className="font-mono font-bold text-primary">{u.upload_id}</div>
                  <div className="text-xs text-muted">{u.source} · {u.cycles} cycles · {u.predicted_cycles} predictions</div>
                </div>
                <div className="text-xs text-gray-300">{u.max_ood_fraction > 0 ? `up to ${fmt(u.max_ood_fraction * 100, 0)}% of inputs outside the training range` : 'all inputs within the training range'}</div>
                <button onClick={() => open(u.upload_id)} className="px-3 py-1.5 rounded border border-primary text-primary text-xs font-bold">Open digital twin →</button>
              </motion.li>
            ))}
          </ul>
          <p className="text-[11px] text-muted mt-3">{result.note}</p>
        </Panel>
      )}

      <Panel title="Uploaded in this session">
        {(uploads.data ?? []).length === 0 ? <p className="text-sm text-muted">None yet.</p> : (
          <ul className="divide-y divide-primary/10">
            {uploads.data!.map((u) => (
              <li key={u.upload_id}>
                <button onClick={() => open(u.upload_id)} className="w-full flex items-center justify-between py-2 px-2 rounded hover:bg-primary/5 text-sm text-left">
                  <span className="font-mono text-primary">{u.upload_id}</span><span className="text-muted text-xs">{u.source}</span>
                  <span className="text-xs text-gray-300">{u.predicted_cycles} predictions{u.has_ground_truth ? ' · with RUL label' : ''}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Panel>
      <Note>
        Any engine from NASA's train_FD001.txt or test_FD001.txt works (copy the rows for one unit). Data from other C-MAPSS subsets (FD002–FD004: six operating conditions)
        lies outside the training range and is flagged; treat those predictions as extrapolations. Files are processed in server memory and never stored.
      </Note>
    </div>
  );
}
