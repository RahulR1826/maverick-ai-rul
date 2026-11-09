import axios from 'axios';

// Configure with VITE_API_URL (see frontend/.env.example); defaults to the local backend.
export const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api';

const http = axios.create({ baseURL: API_URL, timeout: 60000 });

export type Risk = 'Low' | 'Medium' | 'High' | 'Critical';
export type Family = 'fedavg_iid' | 'fedavg_engine' | 'centralized';
/** FD001 test engines are 1-100; uploaded engines have ids like "u-1a2b3c4d". */
export type EngineRef = number | string;

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

export interface TrajectoryPoint {
  cycle: number; rul_mean: number; rul_lower: number; rul_upper: number; true_rul: number | null; risk: Risk;
  /** share of inputs outside the training range at this cycle (0 = fully in range) */
  ood_fraction?: number;
}
export interface Sensor { key: string; name: string; description: string; unit: string; healthy_low: number; healthy_high: number; values: number[] }
export interface EngineDetail {
  engine_id: EngineRef; family: Family; source: string; has_ground_truth: boolean; attributions_available: boolean;
  observed_cycles: number[]; window: number; trajectory: TrajectoryPoint[]; sensors: Sensor[];
}

export interface Explanation {
  engine_id: EngineRef; cycle: number; family: Family; method: string; baseline: string; steps: number; source: string;
  prediction: number; baseline_prediction: number; sum_of_attributions: number; completeness_error: number;
  features: { key: string; name: string; description: string; attribution: number }[];
  heatmap: { cycles: number[]; features: string[]; values: number[][] };
}

/** Per-cycle attribution summed over the 30-cycle window: values[cycleIndex][featureIndex]. */
export interface AttributionSeries { engine_id: number; family: Family; steps: number; cycles: number[]; features: string[]; values: number[][] }

export interface CompareFamily {
  description: string; rul_mean: number[]; rul_lower: number[]; rul_upper: number[]; risk_last: Risk;
  fleet_test_rmse_last_cycle: number; rmse_this_engine_all_cycles?: number; error_last_cycle?: number;
}
export interface Comparison { engine_id: EngineRef; cycles: number[]; true_rul: number[] | null; families: Record<Family, CompareFamily> }

export interface UploadCreated { upload_id: string; unit: number; cycles: number; predicted_cycles: number; max_ood_fraction: number; source: string; has_ground_truth: boolean }
export interface UploadResponse { uploads: UploadCreated[]; note: string; warning: string | null }
export interface UploadListItem { upload_id: string; source: string; predicted_cycles: number; has_ground_truth: boolean }

export interface ErrorSummary { rmse: number; mae: number; nasa_score: number; bias_mean_error: number; frac_late: number; n: number }
export interface IntervalEval { coverage: number; mean_width: number; n: number; per_bin: { predicted_rul: string; n: number; coverage: number | null; mean_width: number | null }[] }
export interface Meta {
  dataset: { name: string; subset: string; test_engines: number; window: number; rul_cap: number; features: { key: string; name: string; description: string; unit: string }[] };
  model: {
    name: Family; description: string; run_ids: string[]; seeds: number[]; checkpoint: string; architecture: string;
    checkpoints: { run_id: string; sha256: string }[]; cache: string; cache_key: string; attributions_precomputed: boolean; load_s: number;
    k: number; val_coverage: number; val_rmse: number;
    test: { ensemble_last_cycle: ErrorSummary; members_last_cycle_rmse_mean: number; members_last_cycle_rmse_std: number;
      interval_coverage_last_cycle: number; interval_coverage_all_cycles: number; mean_interval_width_last_cycle: number;
      rmse_all_cycles: number; n_engines: number; n_cycle_predictions: number };
    latency_ms: { ensemble_window_ms_median: number; ensemble_window_ms_p95: number; device: string; torch_threads: number };
  };
  families: Record<Family, string>;
  default_family: Family;
  interval: {
    coverage_target: number; method: string; note: string;
    calibration: { method: string; alpha: number; k_global: number; bins: { predicted_rul: string; k: number; n_calibration: number; fallback_to_global: boolean }[] };
    baseline_global: { method: string; k: number; n_calibration: number };
    evaluation: Record<'mondrian' | 'global', { last_cycle: IntervalEval; all_cycles: IntervalEval; validation: IntervalEval }>;
  };
  risk: { rule: string; default_thresholds: Thresholds };
  ground_truth_note: string;
  version: { api: string; git_commit: string | null; cache_key: string; cache: string };
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

export interface Health { status: 'ready' | 'loading' | 'error'; error: string | null; families_loaded: string[]; api_version: string; git_commit: string | null; cuda_available: boolean; training_busy: boolean }

export interface TrainParams { num_clients: number; num_rounds: number; local_epochs: number; partition: 'iid_window' | 'dirichlet_window' | 'engine'; alpha?: number; seed: number }
export type TrainEvent =
  | { type: 'start'; params: TrainParams; device: string; gpu: string | null }
  | { type: 'partition'; clients: { client_id: number; n_windows: number; n_engines: number; rul_mean: number }[] }
  | { type: 'round'; round: number; of: number; val_rmse: number; round_s: number; cum_MB: number; elapsed_s: number }
  | { type: 'done'; best_round: number; best_val_rmse: number; test_rmse_selected: number; test_rmse_final: number; test_nasa_selected: number; total_MB: number; wall_s: number; note: string }
  | { type: 'error'; message: string }
  | { type: 'cancelled' };

const get = async <T,>(url: string, params?: Record<string, unknown>): Promise<T> => (await http.get<T>(url, { params })).data;
const ref = (id: EngineRef) => encodeURIComponent(String(id));

export const api = {
  health: () => get<Health>('/health'),
  meta: (family: Family) => get<Meta>('/meta', { family }),
  fleet: (family: Family, th: Thresholds) => get<FleetRow[]>('/fleet', { family, ...th }),
  engine: (id: EngineRef, family: Family, th: Thresholds) => get<EngineDetail>(`/engines/${ref(id)}`, { family, ...th }),
  explain: (id: EngineRef, cycle: number, family: Family) => get<Explanation>(`/engines/${ref(id)}/explain`, { cycle, family }),
  attributions: (id: number, family: Family) => get<AttributionSeries>(`/engines/${ref(id)}/attributions`, { family }),
  compare: (id: EngineRef, th: Thresholds) => get<Comparison>(`/engines/${ref(id)}/compare`, { ...th }),
  uploads: () => get<UploadListItem[]>('/uploads'),
  upload: async (file: File, rulAtEnd?: number) => {
    const form = new FormData();
    form.append('file', file);
    if (rulAtEnd !== undefined) form.append('rul_at_end', String(rulAtEnd));
    return (await http.post<UploadResponse>('/uploads', form)).data;
  },
  trainStatus: () => get<{ busy: boolean; limits: Record<string, unknown>; cuda_available: boolean }>('/train/status'),
  trainStreamUrl: (p: TrainParams) => {
    const q = new URLSearchParams({ num_clients: String(p.num_clients), num_rounds: String(p.num_rounds), local_epochs: String(p.local_epochs), partition: p.partition, seed: String(p.seed) });
    if (p.partition === 'dirichlet_window' && p.alpha !== undefined) q.set('alpha', String(p.alpha));
    return `${API_URL}/train/stream?${q.toString()}`;
  },
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

export const isUpload = (id: EngineRef): id is string => typeof id === 'string';
export const engineLabel = (id: EngineRef) => (isUpload(id) ? `Upload ${id}` : `Engine #${id}`);
