import { create } from 'zustand';
import type { Family, Thresholds } from '../services/api';

// Per-viewer UI settings. Everything shown on screen comes from the API.
export const DEFAULT_THRESHOLDS: Thresholds = { critical: 20, high: 40, medium: 70 };

interface Settings { family: Family; thresholds: Thresholds }

const KEY = 'maverick-settings-v2';

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const s = JSON.parse(raw) as Partial<Settings>;
      return { family: s.family ?? 'fedavg_iid', thresholds: { ...DEFAULT_THRESHOLDS, ...s.thresholds } };
    }
  } catch { /* storage unavailable: use defaults */ }
  return { family: 'fedavg_iid', thresholds: DEFAULT_THRESHOLDS };
}

function save(s: Settings) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* ignore */ }
}

interface AppState extends Settings {
  selectedEngineId: number;
  cycleByEngine: Record<number, number>;
  selectEngine: (id: number) => void;
  setCycle: (id: number, cycle: number) => void;
  setFamily: (f: Family) => void;
  setThresholds: (t: Thresholds) => void;
  resetSettings: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  ...loadSettings(),
  selectedEngineId: 34,
  cycleByEngine: {},
  selectEngine: (id) => set({ selectedEngineId: id }),
  setCycle: (id, cycle) => set((s) => ({ cycleByEngine: { ...s.cycleByEngine, [id]: cycle } })),
  setFamily: (family) => { set({ family }); save({ family, thresholds: get().thresholds }); },
  setThresholds: (thresholds) => { set({ thresholds }); save({ family: get().family, thresholds }); },
  resetSettings: () => { set({ family: 'fedavg_iid', thresholds: DEFAULT_THRESHOLDS }); save({ family: 'fedavg_iid', thresholds: DEFAULT_THRESHOLDS }); },
}));
