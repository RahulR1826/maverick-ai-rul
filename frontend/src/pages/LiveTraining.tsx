import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Play, Square, Zap } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import type { TrainEvent, TrainParams } from '../services/api';
import { useConvergence } from '../hooks/useData';
import { Kpi, Note, PageHeader, Panel, fmt, tooltipStyle } from '../components/ui';
import { FederationOrbit } from '../components/FederationOrbit';
import { cn } from '../utils';

type Status = 'idle' | 'running' | 'done' | 'error' | 'cancelled';
type RoundEv = Extract<TrainEvent, { type: 'round' }>;
type DoneEv = Extract<TrainEvent, { type: 'done' }>;
type PartEv = Extract<TrainEvent, { type: 'partition' }>;

function Slider({ label, value, min, max, onChange, disabled }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void; disabled?: boolean }) {
  return (
    <label className="block">
      <div className="flex justify-between text-xs mb-1"><span className="text-muted uppercase tracking-widest">{label}</span><span className="font-mono text-white">{value}</span></div>
      <input type="range" min={min} max={max} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-[#00BFFF]" />
    </label>
  );
}

export function LiveTraining() {
  const [p, setP] = useState<TrainParams>({ num_clients: 5, num_rounds: 10, local_epochs: 5, partition: 'iid_window', alpha: 0.5, seed: 42 });
  const [status, setStatus] = useState<Status>('idle');
  const [rounds, setRounds] = useState<RoundEv[]>([]);
  const [done, setDone] = useState<DoneEv | null>(null);
  const [part, setPart] = useState<PartEv | null>(null);
  const [device, setDevice] = useState<string>('');
  const [error, setError] = useState<string>('');
  const es = useRef<EventSource | null>(null);
  const trainStatus = useQuery({ queryKey: ['trainStatus'], queryFn: api.trainStatus, refetchInterval: status === 'running' ? false : 5000 });
  const conv = useConvergence();

  const budget = p.num_rounds * p.local_epochs;
  const valid = budget <= 100;
  useEffect(() => () => es.current?.close(), []);

  const start = () => {
    setRounds([]); setDone(null); setPart(null); setError(''); setStatus('running');
    const src = new EventSource(api.trainStreamUrl(p));
    es.current = src;
    const on = (type: string, fn: (d: TrainEvent) => void) => src.addEventListener(type, (m) => fn(JSON.parse((m as MessageEvent).data)));
    on('start', (d) => { if (d.type === 'start') setDevice(d.gpu ?? d.device); });
    on('partition', (d) => { if (d.type === 'partition') setPart(d); });
    on('round', (d) => { if (d.type === 'round') setRounds((r) => [...r, d]); });
    on('done', (d) => { if (d.type === 'done') setDone(d); setStatus('done'); src.close(); });
    on('cancelled', () => { setStatus('cancelled'); src.close(); });
    src.addEventListener('error', (m) => {
      const data = (m as MessageEvent).data;
      if (data) { try { setError(JSON.parse(data).message); } catch { setError(String(data)); } }
      else if (src.readyState === EventSource.CLOSED || status === 'running') setError((e) => e || 'Connection to the training stream failed (is another job running, or the API down?).');
      setStatus((s) => (s === 'done' ? s : 'error'));
      src.close();
    });
  };
  const stop = () => { es.current?.close(); setStatus('cancelled'); };

  // the paper's 5-seed FedAvg-IID validation curve is a fair reference only for the same set-up (K=5, E=5, IID)
  const comparable = p.num_clients === 5 && p.local_epochs === 5 && p.partition === 'iid_window';
  const chart = useMemo(() => {
    const ref = comparable ? (conv.data ?? []).filter((r) => r.experiment === 'fedavg_iid' && r.grid_label === 'default') : [];
    const n = Math.max(p.num_rounds, rounds.length);
    return Array.from({ length: n }, (_, i) => ({
      round: i + 1, live: rounds[i]?.val_rmse ?? null, paper: ref.find((r) => r.step === i + 1)?.mean ?? null,
    }));
  }, [rounds, conv.data, comparable, p.num_rounds]);
  const last = rounds[rounds.length - 1];
  const progress = last ? last.round / last.of : 0;
  const running = status === 'running';

  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={Zap} title="Live federated training" subtitle="Run a short, real FedAvg job on the server (GPU if available) and watch every communication round arrive. Same data, model and trainer as the paper." />
      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-6">
        <Panel title="Federation set-up">
          <div className="space-y-4">
            <Slider label="Clients (K)" value={p.num_clients} min={1} max={10} disabled={running} onChange={(v) => setP({ ...p, num_clients: v })} />
            <Slider label="Rounds (R)" value={p.num_rounds} min={1} max={30} disabled={running} onChange={(v) => setP({ ...p, num_rounds: v })} />
            <Slider label="Local epochs (E)" value={p.local_epochs} min={1} max={10} disabled={running} onChange={(v) => setP({ ...p, local_epochs: v })} />
            <div>
              <div className="text-xs text-muted uppercase tracking-widest mb-1">Partition</div>
              <div className="grid grid-cols-3 gap-1">
                {([['iid_window', 'IID window'], ['dirichlet_window', 'Dirichlet'], ['engine', 'Whole-engine']] as const).map(([k, l]) => (
                  <button key={k} disabled={running} onClick={() => setP({ ...p, partition: k })} aria-pressed={p.partition === k}
                    className={cn('px-2 py-1.5 rounded border text-[11px] font-bold', p.partition === k ? 'border-primary text-primary bg-primary/10' : 'border-white/10 text-muted hover:text-white')}>{l}</button>
                ))}
              </div>
            </div>
            {p.partition === 'dirichlet_window' && (
              <div>
                <div className="text-xs text-muted uppercase tracking-widest mb-1">Dirichlet α (smaller = more skewed)</div>
                <div className="grid grid-cols-3 gap-1">
                  {[1.0, 0.5, 0.1].map((a) => (
                    <button key={a} disabled={running} onClick={() => setP({ ...p, alpha: a })} aria-pressed={p.alpha === a}
                      className={cn('px-2 py-1.5 rounded border text-[11px] font-bold', p.alpha === a ? 'border-primary text-primary bg-primary/10' : 'border-white/10 text-muted')}>{a}</button>
                  ))}
                </div>
              </div>
            )}
            <label className="block text-xs"><span className="text-muted uppercase tracking-widest">Seed</span>
              <input type="number" value={p.seed} disabled={running} onChange={(e) => setP({ ...p, seed: Number(e.target.value) })}
                className="mt-1 w-full bg-black/40 border border-primary/20 rounded px-3 py-2 font-mono text-white" /></label>
            <div className={cn('text-[11px] rounded px-2 py-1.5', valid ? 'text-muted bg-black/30' : 'text-[#FF6B4A] bg-[#FF6B4A]/10')}>
              Budget R × E = {budget} passes over the data {valid ? '(≤ 100, the paper\'s budget)' : '— must be ≤ 100'}
            </div>
            {running
              ? <button onClick={stop} className="w-full flex items-center justify-center gap-2 py-3 rounded-lg border border-[#FF6B4A] text-[#FF6B4A] font-bold"><Square className="w-4 h-4" />Stop</button>
              : <button onClick={start} disabled={!valid || trainStatus.data?.busy}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-lg bg-primary text-[#030712] font-black shadow-[0_0_25px_rgba(0,191,255,0.5)] disabled:opacity-40 disabled:shadow-none">
                  <Play className="w-4 h-4" />{trainStatus.data?.busy ? 'Another job is running' : 'Start training'}</button>}
            <div className="text-[11px] text-muted flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5" />
              {device ? `Running on ${device}` : trainStatus.data ? (trainStatus.data.cuda_available ? 'GPU available on the server' : 'Server has no GPU: training runs on CPU (slower)') : 'Checking server…'}</div>
          </div>
        </Panel>

        <div className="space-y-6">
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
            <Kpi label="Round" value={last ? `${last.round} / ${last.of}` : '—'} sub={status === 'idle' ? 'press Start' : status} />
            <Kpi label="Validation RMSE" value={last ? fmt(last.val_rmse, 2) : '—'} color="#00E676" sub="global model, 10 held-out engines" />
            <Kpi label="Model traffic" value={last ? `${fmt(last.cum_MB, 1)} MB` : '—'} color="#FFB300" sub="serialized payload, up + down" />
            <Kpi label="Elapsed" value={last ? `${fmt(last.elapsed_s, 1)} s` : '—'} color="#9fb4c8" sub={last ? `last round ${fmt(last.round_s, 2)} s` : ''} />
          </div>
          <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
            <motion.div className="h-full bg-primary shadow-[0_0_12px_#00BFFF]" animate={{ width: `${progress * 100}%` }} transition={{ type: 'spring', stiffness: 80, damping: 20 }} />
          </div>
          <div className="grid grid-cols-1 2xl:grid-cols-[1fr_300px] gap-6">
            <Panel title="Validation RMSE after each communication round">
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chart} margin={{ top: 5, right: 20, left: 0, bottom: 15 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="round" stroke="#6B8CAE" fontSize={11} label={{ value: 'Round', position: 'insideBottom', offset: -8, fill: '#6B8CAE', fontSize: 11 }} />
                    <YAxis stroke="#6B8CAE" fontSize={11} domain={[0, 'auto']} label={{ value: 'Val. RMSE (cycles)', angle: -90, position: 'insideLeft', fill: '#6B8CAE', fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown, n: unknown) => [fmt(v as number, 2), String(n)]} />
                    <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
                    {comparable && <Line dataKey="paper" name="Paper: FedAvg-IID, 5-seed mean" stroke="#6B8CAE" strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />}
                    <Line dataKey="live" name="This run" stroke="#00BFFF" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} connectNulls />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[11px] text-muted mt-2">{comparable ? 'Dashed: the paper\'s 5-seed mean for the same set-up (K=5, E=5, IID window).' : 'The paper reference curve is shown only for the matching set-up (K=5, E=5, IID window).'}</p>
            </Panel>
            <Panel title="Federation">
              <div className="flex justify-center"><FederationOrbit size={250} k={p.num_clients} paused={!running} caption={`K = ${p.num_clients} · R = ${p.num_rounds} · E = ${p.local_epochs}`} /></div>
              <p className="text-[10px] text-muted text-center">Animation runs while training; packets are illustrative.</p>
            </Panel>
          </div>

          {part && (
            <Panel title="Client data in this run">
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={part.clients.map((c) => ({ client: `C${c.client_id + 1}`, windows: c.n_windows, engines: c.n_engines }))}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" /><XAxis dataKey="client" stroke="#6B8CAE" fontSize={11} /><YAxis stroke="#6B8CAE" fontSize={11} />
                    <Tooltip contentStyle={tooltipStyle} /><Bar dataKey="windows" name="Training windows" fill="#00BFFF" isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[11px] text-muted">Engines per client: {part.clients.map((c) => c.n_engines).join(' · ')}. Mean RUL label per client: {part.clients.map((c) => fmt(c.rul_mean, 0)).join(' · ')}.</p>
            </Panel>
          )}

          {done && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              <Panel title="Result on the 100 FD001 test engines">
                <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
                  <Kpi label="Test RMSE (selected)" value={fmt(done.test_rmse_selected, 2)} color="#00E676" sub={`best validation at round ${done.best_round}`} />
                  <Kpi label="Test RMSE (final)" value={fmt(done.test_rmse_final, 2)} color="#00BFFF" />
                  <Kpi label="NASA score" value={fmt(done.test_nasa_selected, 0)} color="#FFB300" />
                  <Kpi label="Total time" value={`${fmt(done.wall_s, 1)} s`} color="#9fb4c8" sub={`${fmt(done.total_MB, 1)} MB of model traffic`} />
                </div>
                <p className="text-[11px] text-muted mt-3">{done.note} For comparison, the paper's FedAvg-IID (K=5, R=20, E=5) reaches 13.24 ± 0.32 over 5 seeds.</p>
              </Panel>
            </motion.div>
          )}
          {error && <div role="alert" className="text-sm text-[#FF6B4A] border border-[#FF6B4A]/30 bg-[#FF6B4A]/10 rounded-lg p-4">{error}</div>}
        </div>
      </div>
      <Note>Each run trains a new LSTM from scratch with FedAvg, simulating the clients one after another on the server. Nothing is saved, and the served models are not changed. One job runs at a time.</Note>
    </div>
  );
}
