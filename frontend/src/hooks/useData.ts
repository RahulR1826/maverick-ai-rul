import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import type { EngineRef } from '../services/api';
import { useStore } from '../store/useStore';

// All server data is static per (model family, thresholds): cache it for the session.
const STATIC = { staleTime: Infinity, retry: 1 } as const;

export const useHealth = () => useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: (q) => (q.state.data?.status === 'ready' ? false : 3000) });

export function useMeta() {
  const family = useStore((s) => s.family);
  return useQuery({ queryKey: ['meta', family], queryFn: () => api.meta(family), ...STATIC });
}

export function useFleet() {
  const family = useStore((s) => s.family);
  const th = useStore((s) => s.thresholds);
  return useQuery({ queryKey: ['fleet', family, th], queryFn: () => api.fleet(family, th), ...STATIC });
}

export function useEngine(id: EngineRef) {
  const family = useStore((s) => s.family);
  const th = useStore((s) => s.thresholds);
  return useQuery({ queryKey: ['engine', id, family, th], queryFn: () => api.engine(id, family, th), ...STATIC });
}

export function useExplain(id: EngineRef, cycle: number | undefined) {
  const family = useStore((s) => s.family);
  return useQuery({ queryKey: ['explain', id, cycle, family], queryFn: () => api.explain(id, cycle!, family), enabled: cycle !== undefined, ...STATIC });
}

export const useSummary = (checkpoint: 'selected' | 'final') => useQuery({ queryKey: ['summary', checkpoint], queryFn: () => api.summary(checkpoint), ...STATIC });
export const usePerSeed = (checkpoint: 'selected' | 'final') => useQuery({ queryKey: ['perSeed', checkpoint], queryFn: () => api.perSeed(checkpoint), ...STATIC });
export const useConvergence = () => useQuery({ queryKey: ['convergence'], queryFn: api.convergence, ...STATIC });
export const usePartitions = (seed: number) => useQuery({ queryKey: ['partitions', seed], queryFn: () => api.partitions(seed), ...STATIC });
export const usePaired = () => useQuery({ queryKey: ['paired'], queryFn: api.paired, ...STATIC });
export const useRuntime = () => useQuery({ queryKey: ['runtime'], queryFn: api.runtime, ...STATIC });

/** Per-cycle attributions (precomputed on the server) for FD001 test engines; null for uploads. */
export function useAttributions(id: EngineRef, enabled: boolean) {
  const family = useStore((s) => s.family);
  return useQuery({ queryKey: ['attributions', id, family], queryFn: () => api.attributions(id as number, family),
    enabled: enabled && typeof id === 'number', ...STATIC, retry: false });
}

export function useCompare(id: EngineRef) {
  const th = useStore((s) => s.thresholds);
  return useQuery({ queryKey: ['compare', id, th], queryFn: () => api.compare(id, th), ...STATIC });
}

export const useUploads = () => useQuery({ queryKey: ['uploads'], queryFn: api.uploads, staleTime: 0 });
