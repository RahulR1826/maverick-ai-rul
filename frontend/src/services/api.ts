import axios from 'axios';

// Configure with VITE_API_URL (see frontend/.env.example); defaults to the local backend.
export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';

const http = axios.create({ baseURL: API_URL, timeout: 60000 });

export type Risk = 'Low' | 'Medium' | 'High' | 'Critical';
export type Family = 'fedavg_iid' | 'fedavg_engine' | 'centralized';

export interface Thresholds { critical: number; high: number; medium: number }

export interface FleetRow {
  engine_id: number;
  last_cycle: number;
  rul_mean: number;
  rul_std: number;
  rul_lower: number;
  rul_upper: number;
  life_remaining_pct: number;
  risk: Risk;
  true_rul: number;
  true_rul_uncapped: number;
  critical_since_cycle: number | null;
  high_since_cycle: number | null;
}

export interface TrajectoryPoint { cycle: number; rul_mean: number; rul_lower: number; rul_upper: number; true_rul: number; risk: Risk }
export interface Sensor { key: string; name: string; description: string; unit: string; healthy_low: number; healthy_high: number; values: number[] }
export interface EngineDetail { engine_id: number; family: Family; observed_cycles: number[]; window: number; trajectory: TrajectoryPoint[]; sensors: Sensor[] }

export interface Explanation {
  engine_id: number; cycle: number; family: Family; method: string; baseline: string; steps: number;
  prediction: number; baseline_prediction: number; sum_of_attributions: number; completeness_error: number;
  features: { key: string; name: string; description: string; attribution: number }[];
  heatmap: { cycles: number[]; features: string[]; values: number[][] };
}

export interface ErrorSummary { rmse: number; mae: number; nasa_score: number; bias_mean_error: number; frac_late: number; n: number }
export interface Meta {
  dataset: { name: string; subset: string; test_engines: number; window: number; rul_cap: number; features: { key: string; name: string; description: string; unit: string }[] };
  model: {
    name: Family; description: string; run_ids: string[]; seeds: number[]; checkpoint: string; architecture: string;
    k: number; val_coverage: number; val_rmse: number;
    test: { ensemble_last_cycle: ErrorSummary; members_last_cycle_rmse_mean: number; members_last_cycle_rmse_std: number;
      interval_coverage_last_cycle: number; interval_coverage_all_cycles: number; mean_interval_width_last_cycle: number;
      rmse_all_cycles: number; n_engines: number; n_cycle_predictions: number };
    latency_ms: { ensemble_window_ms_median: number; ensemble_window_ms_p95: number; device: string; torch_threads: number };
  };
  families: Record<Family, string>;
  default_family: Family;
  interval: { coverage_target: number; method: string };
  risk: { rule: string; default_thresholds: Thresholds };
  ground_truth_note: string;
}

export interface SummaryRow {
  experiment: string; grid_label: string; method: string; partition: string | null; num_clients: number | null;
  num_rounds: number | null; local_epochs: number | null; alpha: number | null; checkpoint: string; n_seeds: number;
  rmse_mean: number; rmse_std: number | null; mae_mean: number; mae_std: number | null; nasa_score_mean: number;
  nasa_score_std: number | null; best_val_rmse_mean: number; comm_total_MB_mean: number; selected_at_mean: number | null;
}
export interface SeedRow { experiment: string; grid_label: string; seed: number; rmse: number; mae: number; nasa_score: number }
export interface ConvergenceRow { experiment: string; grid_label: string; step_type: 'round' | 'epoch'; step: number; mean: number; std: number | null; count: number; cum_total_MB?: number | null }
export interface PartitionRow { experiment: string; grid_label: string; client_id: number; n_windows: number; n_engines: number; n_engines_exclusive: number; rul_mean: number; rul_std: number }
export interface PairedRow { checkpoint: string; comparison: string; metric: string; n_matched_seeds: number; mean_diff: number; std_diff: number; n_other_lower: number; per_seed_diff: string }
export interface RuntimeRow { experiment: string; wall_s: number; mean_round_wall_s: number | null; mean_client_local_train_s: number | null; parallel_estimate_total_s: number | null; sample_passes: number; payload_bytes: number; comm_total_MB: number }

const get = async <T,>(url: string, params?: Record<string, unknown>): Promise<T> => (await http.get<T>(url, { params })).data;

export const api = {
  health: () => get<{ status: 'ready' | 'loading' | 'error'; error: string | null }>('/health'),
  meta: (family: Family) => get<Meta>('/meta', { family }),
  fleet: (family: Family, th: Thresholds) => get<FleetRow[]>('/fleet', { family, ...th }),
  engine: (id: number, family: Family, th: Thresholds) => get<EngineDetail>(`/engines/${id}`, { family, ...th }),
  explain: (id: number, cycle: number, family: Family) => get<Explanation>(`/engines/${id}/explain`, { cycle, family }),
  summary: (checkpoint: 'selected' | 'final') => get<SummaryRow[]>('/results/summary', { checkpoint }),
  perSeed: (checkpoint: 'selected' | 'final') => get<SeedRow[]>('/results/per_seed', { checkpoint }),
  convergence: () => get<ConvergenceRow[]>('/results/convergence'),
  partitions: (seed: number) => get<PartitionRow[]>('/results/partitions', { seed }),
  paired: () => get<PairedRow[]>('/results/paired'),
  runtime: () => get<RuntimeRow[]>('/results/runtime'),
};

export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    if (!err.response) return `Cannot reach the API at ${API_URL}. Start it with "python api_server.py".`;
    const detail = (err.response.data as { detail?: unknown })?.detail;
    return `API error ${err.response.status}: ${typeof detail === 'string' ? detail : err.message}`;
  }
  return String(err);
}
