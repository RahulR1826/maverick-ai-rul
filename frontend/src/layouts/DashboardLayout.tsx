import { Link, Outlet, useLocation } from 'react-router-dom';
import { cn } from '../utils';
import { LayoutDashboard, Plane, Gauge, Server, Network, PenTool, BarChart3, AlertTriangle, Settings, BrainCircuit } from 'lucide-react';
import { useHealth } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { API_URL } from '../services/api';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Fleet', path: '/fleet', icon: Plane },
  { name: 'Engine Monitor', path: '/engine', icon: Gauge },
  { name: 'Explainable AI', path: '/xai', icon: BrainCircuit },
  { name: 'Maintenance', path: '/maintenance', icon: PenTool },
  { name: 'Alerts', path: '/alerts', icon: AlertTriangle },
  { name: 'Federated Learning', path: '/fl', icon: Network },
  { name: 'Results Explorer', path: '/results', icon: BarChart3 },
  { name: 'Model & Deployment', path: '/model', icon: Server },
  { name: 'Settings', path: '/settings', icon: Settings },
];

const FAMILY_SHORT = { fedavg_iid: 'FedAvg · IID', fedavg_engine: 'FedAvg · whole-engine', centralized: 'Centralized' } as const;

export function DashboardLayout() {
  const location = useLocation();
  const health = useHealth();
  const family = useStore((s) => s.family);
  const status = health.isError ? 'offline' : health.data?.status ?? 'connecting';
  const statusColor = status === 'ready' ? '#00E676' : status === 'loading' || status === 'connecting' ? '#FFB300' : '#FF0055';
  const statusText = { ready: 'Model ready', loading: 'Loading model…', connecting: 'Connecting…', offline: 'API offline', error: 'Model error' }[status];

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      <div className="fixed inset-0 pointer-events-none z-0 bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.12),_transparent_70%)]" />
      <div className="fixed inset-0 pointer-events-none z-0 opacity-[0.03]"
        style={{ backgroundImage: 'linear-gradient(white 1px, transparent 1px), linear-gradient(90deg, white 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

      <aside className="relative z-10 w-64 shrink-0 border-r border-cyan-500/20 flex flex-col"
        style={{ background: 'rgba(5, 11, 20, 0.70)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)' }}>
        <div className="h-16 flex items-center px-6 border-b border-cyan-500/20">
          <Plane className="text-primary w-6 h-6 mr-3 drop-shadow-[0_0_8px_rgba(0,191,255,0.8)]" />
          <h1 className="font-bold text-lg tracking-wider text-white drop-shadow-[0_0_10px_rgba(0,191,255,0.4)]">MAVERICK AI</h1>
        </div>

        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1" aria-label="Main">
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
            return (
              <Link key={item.name} to={item.path} aria-current={isActive ? 'page' : undefined}
                className={cn('flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group',
                  isActive ? 'text-primary border border-cyan-500/30' : 'text-muted hover:text-white hover:border hover:border-cyan-500/10')}
                style={isActive ? { background: 'rgba(0,191,255,0.08)', boxShadow: '0 0 15px rgba(0,191,255,0.12), inset 0 0 10px rgba(0,191,255,0.04)' } : {}}>
                <item.icon className={cn('w-5 h-5 mr-3 transition-colors', isActive ? 'text-primary' : 'text-muted group-hover:text-white')} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-cyan-500/20">
          <div className="glass-panel-sm p-3 text-xs flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-muted">Backend</span>
              <span className="flex items-center" style={{ color: statusColor }}>
                <span className="w-2 h-2 rounded-full mr-1.5" style={{ background: statusColor }} />{statusText}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted">Model</span><span className="text-white">{FAMILY_SHORT[family]}</span>
            </div>
            <div className="mt-1 text-[10px] text-muted text-center border-t border-cyan-500/10 pt-2 break-all" title="API base URL">{API_URL}</div>
          </div>
        </div>
      </aside>

      <main className="relative z-10 flex-1 overflow-y-auto">
        <div className="p-8 pb-20 min-h-full"><Outlet /></div>
      </main>
    </div>
  );
}
