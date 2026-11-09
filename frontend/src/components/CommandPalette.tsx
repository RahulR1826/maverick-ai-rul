import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { CornerDownLeft, Gauge, Search } from 'lucide-react';
import { useFleet } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { RISK_COLOR } from './ui';
import { cn } from '../utils';

export interface NavItem { name: string; path: string; icon: React.ComponentType<{ className?: string }> }

interface Cmd { id: string; label: string; hint: string; color?: string; icon: React.ComponentType<{ className?: string }>; run: () => void }

/** Ctrl/⌘+K: jump to any page, or open an engine's digital twin by number. */
export function CommandPalette({ open, onClose, nav }: { open: boolean; onClose: () => void; nav: NavItem[] }) {
  const [q, setQ] = useState('');
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const fleet = useFleet();
  const selectEngine = useStore((s) => s.selectEngine);

  const cmds = useMemo<Cmd[]>(() => {
    const pages: Cmd[] = nav.map((n) => ({ id: n.path, label: n.name, hint: 'Page', icon: n.icon, run: () => navigate(n.path) }));
    const engines: Cmd[] = (fleet.data ?? []).map((e) => ({
      id: `e${e.engine_id}`, label: `Engine #${e.engine_id}`, hint: `${e.risk} · ${e.rul_mean.toFixed(0)} cycles left`, color: RISK_COLOR[e.risk], icon: Gauge,
      run: () => { selectEngine(e.engine_id); navigate('/engine'); },
    }));
    const t = q.trim().toLowerCase().replace(/^#|^engine\s*/, '');
    if (!t) return [...pages, ...engines.filter((e) => e.hint.startsWith('Critical')).slice(0, 5)];
    if (/^\d+$/.test(t)) return engines.filter((e) => e.label.endsWith(`#${t}`) || e.label.includes(`#${t}`)).slice(0, 8);
    return [...pages, ...engines].filter((c) => `${c.label} ${c.hint}`.toLowerCase().includes(t)).slice(0, 10);
  }, [q, nav, fleet.data, navigate, selectEngine]);

  useEffect(() => { if (open) { setQ(''); setSel(0); setTimeout(() => input.current?.focus(), 30); } }, [open]);
  useEffect(() => setSel(0), [q]);

  const run = (c?: Cmd) => { if (c) { c.run(); onClose(); } };
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-50 flex items-start justify-center pt-[14vh] bg-black/60 backdrop-blur-sm" onClick={onClose}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          <motion.div role="dialog" aria-label="Command palette" onClick={(e) => e.stopPropagation()}
            initial={{ y: -20, scale: 0.97, opacity: 0 }} animate={{ y: 0, scale: 1, opacity: 1 }} exit={{ y: -10, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 380, damping: 30 }}
            className="w-full max-w-xl rounded-2xl border border-cyan-400/30 bg-[#050b16]/95 shadow-[0_0_80px_rgba(0,191,255,0.25)] overflow-hidden">
            <div className="flex items-center gap-3 px-4 border-b border-white/10">
              <Search className="w-4 h-4 text-primary" />
              <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Jump to a page or type an engine number…"
                className="flex-1 bg-transparent py-4 text-white outline-none placeholder:text-muted"
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(cmds.length - 1, s + 1)); }
                  else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
                  else if (e.key === 'Enter') run(cmds[sel]);
                  else if (e.key === 'Escape') onClose();
                }} />
              <kbd className="text-[10px] text-muted border border-white/15 rounded px-1.5 py-0.5">ESC</kbd>
            </div>
            <ul className="max-h-80 overflow-y-auto py-2" role="listbox">
              {cmds.length === 0 && <li className="px-4 py-3 text-sm text-muted">No match.</li>}
              {cmds.map((c, i) => (
                <li key={c.id} role="option" aria-selected={i === sel}>
                  <button onMouseEnter={() => setSel(i)} onClick={() => run(c)}
                    className={cn('w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left', i === sel ? 'bg-primary/15 text-white' : 'text-gray-300')}>
                    <c.icon className="w-4 h-4 text-primary" />
                    <span className="flex-1">{c.label}</span>
                    <span className="text-[11px]" style={{ color: c.color ?? '#6B8CAE' }}>{c.hint}</span>
                    {i === sel && <CornerDownLeft className="w-3.5 h-3.5 text-muted" />}
                  </button>
                </li>
              ))}
            </ul>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
