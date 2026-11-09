import { useEffect, useState } from 'react';
import { Link, useLocation, useOutlet } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { cn } from '../utils';
import { LayoutDashboard, Plane, Box, Server, Network, PenTool, BarChart3, AlertTriangle, Settings, BrainCircuit, Search, Zap, UploadCloud } from 'lucide-react';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { useHealth } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { API_URL } from '../services/api';
import { CommandPalette } from '../components/CommandPalette';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Fleet', path: '/fleet', icon: Plane },
  { name: 'Digital Twin', path: '/engine', icon: Box },
  { name: 'Explainable AI', path: '/xai', icon: BrainCircuit },
  { name: 'Maintenance', path: '/maintenance', icon: PenTool },
  { name: 'Alerts', path: '/alerts', icon: AlertTriangle },
  { name: 'Federated Learning', path: '/fl', icon: Network },
  { name: 'Live Training', path: '/train', icon: Zap },
  { name: 'Upload Engine', path: '/upload', icon: UploadCloud },
  { name: 'Results Explorer', path: '/results', icon: BarChart3 },
  { name: 'Model & Deployment', path: '/model', icon: Server },
  { name: 'Settings', path: '/settings', icon: Settings },
];

const FAMILY_SHORT = { fedavg_iid: 'FedAvg · IID', fedavg_engine: 'FedAvg · whole-engine', centralized: 'Centralized' } as const;

export function DashboardLayout() {
  const location = useLocation();
  const outlet = useOutlet();
  const health = useHealth();
  const family = useStore((s) => s.family);
  const [palette, setPalette] = useState(false);
  const status = health.isError ? 'offline' : health.data?.status ?? 'connecting';
  const statusColor = status === 'ready' ? '#00E676' : status === 'loading' || status === 'connecting' ? '#FFB300' : '#FF0055';
  const statusText = { ready: 'Model ready', loading: 'Loading model…', connecting: 'Connecting…', offline: 'API offline', error: 'Model error' }[status];
  const current = NAV_ITEMS.find((n) => n.path === location.pathname) ?? NAV_ITEMS[0];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette((v) => !v); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-[#02060f]">
      {/* backdrop: slow aurora + grid */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="aurora aurora-a" /><div className="aurora aurora-b" /><div className="aurora aurora-c" />
        <div className="absolute inset-0 opacity-[0.035]" style={{ backgroundImage: 'linear-gradient(white 1px, transparent 1px), linear-gradient(90deg, white 1px, transparent 1px)', backgroundSize: '44px 44px' }} />
      </div>

      <aside className="relative z-10 w-64 shrink-0 border-r border-cyan-500/20 flex flex-col"
        style={{ background: 'rgba(4, 10, 20, 0.72)', backdropFilter: 'blur(18px)', WebkitBackdropFilter: 'blur(18px)' }}>
        <div className="h-16 flex items-center px-6 border-b border-cyan-500/20">
          <motion.div animate={{ rotate: [0, 8, 0, -8, 0] }} transition={{ duration: 8, repeat: Infinity }}>
            <Plane className="text-primary w-6 h-6 mr-3 drop-shadow-[0_0_10px_rgba(0,191,255,0.9)]" />
          </motion.div>
          <h1 className="font-black text-lg tracking-[0.2em] text-white">MAVERICK<span className="text-primary">.AI</span></h1>
        </div>

        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1" aria-label="Main">
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
            return (
              <Link key={item.name} to={item.path} aria-current={isActive ? 'page' : undefined}
                className={cn('relative flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-colors group', isActive ? 'text-white' : 'text-muted hover:text-white')}>
                {isActive && (
                  <motion.span layoutId="nav-active" className="absolute inset-0 rounded-lg border border-cyan-400/40"
                    style={{ background: 'linear-gradient(90deg, rgba(0,191,255,0.18), rgba(0,191,255,0.03))', boxShadow: '0 0 18px rgba(0,191,255,0.18)' }}
                    transition={{ type: 'spring', stiffness: 420, damping: 34 }} />
                )}
                {isActive && <motion.span layoutId="nav-bar" className="absolute left-0 top-2 bottom-2 w-[3px] rounded-full bg-primary shadow-[0_0_10px_#00BFFF]" />}
                <item.icon className={cn('relative w-5 h-5 mr-3 transition-colors', isActive ? 'text-primary' : 'text-muted group-hover:text-white')} />
                <span className="relative">{item.name}</span>
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-cyan-500/20">
          <div className="glass-panel-sm p-3 text-xs flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-muted">Backend</span>
              <span className="flex items-center" style={{ color: statusColor }}>
                <span className="relative w-2 h-2 mr-1.5">
                  <span className="absolute inset-0 rounded-full animate-ping" style={{ background: statusColor, opacity: 0.5 }} />
                  <span className="absolute inset-0 rounded-full" style={{ background: statusColor }} />
                </span>{statusText}
              </span>
            </div>
            <div className="flex items-center justify-between"><span className="text-muted">Model</span><span className="text-white">{FAMILY_SHORT[family]}</span></div>
            <div className="mt-1 text-[10px] text-muted text-center border-t border-cyan-500/10 pt-2 break-all" title="API base URL">{API_URL}</div>
          </div>
        </div>
      </aside>

      <main className="relative z-10 flex-1 overflow-y-auto">
        <header className="sticky top-0 z-30 h-14 flex items-center justify-between px-8 border-b border-cyan-500/10"
          style={{ background: 'rgba(2,6,15,0.6)', backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)' }}>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted">Maverick AI</span><span className="text-muted">/</span>
            <span className="text-white font-semibold flex items-center gap-2"><current.icon className="w-4 h-4 text-primary" />{current.name}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden md:inline text-[11px] text-muted font-mono">NASA C-MAPSS FD001 · 5-seed ensemble</span>
            <button onClick={() => setPalette(true)}
              className="flex items-center gap-2 text-xs text-muted border border-white/10 rounded-lg px-3 py-1.5 bg-black/40 hover:text-white hover:border-primary/40 transition-colors">
              <Search className="w-3.5 h-3.5" /> Search pages & engines
              <kbd className="ml-2 text-[10px] border border-white/15 rounded px-1.5">Ctrl K</kbd>
            </button>
          </div>
        </header>
        <div className="p-8 pb-20 min-h-full">
          <AnimatePresence mode="wait">
            <motion.div key={location.pathname} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.28, ease: 'easeOut' }}>
              <ErrorBoundary key={location.pathname} label="page">{outlet}</ErrorBoundary>
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      <CommandPalette open={palette} onClose={() => setPalette(false)} nav={NAV_ITEMS} />
    </div>
  );
}
