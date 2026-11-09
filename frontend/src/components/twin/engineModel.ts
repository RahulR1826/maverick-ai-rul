import type { Explanation, Sensor } from '../../services/api';

/**
 * Schematic turbofan used by the digital twin.
 *
 * Geometry is built in "lathe space": the engine axis is +Y (intake at +Y,
 * exhaust at -Y), radius in the XZ plane. The scene rotates it so the axis
 * runs left-right on screen. Dimensions are illustrative, not to scale.
 *
 * Sensor placement follows the C-MAPSS station numbering (Saxena et al. 2008,
 * Fig. 1): 2 = fan inlet, 24 = LPC outlet, 30 = HPC outlet, 50 = LPT outlet.
 */

export type ModuleId = 'fan' | 'lpc' | 'hpc' | 'combustor' | 'hpt' | 'lpt';

export interface ModuleDef {
  id: ModuleId;
  name: string;
  /** C-MAPSS feature keys measured at / attributed to this module */
  sensors: string[];
  /** axial extent in lathe space (y0 > y1) */
  y0: number;
  y1: number;
  radius: number;
}

export const MODULES: ModuleDef[] = [
  { id: 'fan', name: 'Fan', sensors: ['s_8', 's_13', 's_15'], y0: 2.55, y1: 2.15, radius: 1.5 },
  { id: 'lpc', name: 'Low-pressure compressor', sensors: ['s_2'], y0: 2.0, y1: 1.15, radius: 0.72 },
  { id: 'hpc', name: 'High-pressure compressor', sensors: ['s_3', 's_7', 's_11', 's_9', 's_14'], y0: 1.05, y1: -0.25, radius: 0.66 },
  { id: 'combustor', name: 'Combustor', sensors: ['s_12', 's_17'], y0: -0.3, y1: -0.8, radius: 0.66 },
  { id: 'hpt', name: 'High-pressure turbine', sensors: ['s_20'], y0: -0.85, y1: -1.25, radius: 0.68 },
  { id: 'lpt', name: 'Low-pressure turbine', sensors: ['s_4', 's_21'], y0: -1.3, y1: -2.25, radius: 0.8 },
];

/** Where each sensor's marker sits: axial position y, radius r, azimuth phi (0 = facing the camera). */
export const SENSOR_STATIONS: Record<string, { y: number; r: number; phi: number }> = {
  s_8: { y: 2.3, r: 0.42, phi: 0.25 },     // Nf  - fan spool speed
  s_13: { y: 2.3, r: 0.42, phi: -0.25 },   // NRf - corrected fan speed
  s_15: { y: 1.0, r: 1.22, phi: 0.1 },     // BPR - bypass duct
  s_2: { y: 1.15, r: 0.6, phi: 0.35 },     // T24 - LPC outlet
  s_3: { y: -0.25, r: 0.56, phi: 0.45 },   // T30 - HPC outlet
  s_7: { y: -0.25, r: 0.56, phi: 0.05 },   // P30
  s_11: { y: -0.25, r: 0.56, phi: -0.35 }, // Ps30
  s_9: { y: 0.45, r: 0.22, phi: 0.5 },     // Nc  - core spool speed
  s_14: { y: 0.45, r: 0.22, phi: -0.5 },   // NRc
  s_12: { y: -0.55, r: 0.66, phi: 0.3 },   // phi - fuel flow / Ps30
  s_17: { y: -0.1, r: 0.7, phi: -0.6 },    // htBleed
  s_20: { y: -1.05, r: 0.64, phi: 0.25 },  // W31 - HPT coolant bleed
  s_21: { y: -1.75, r: 0.74, phi: -0.25 }, // W32 - LPT coolant bleed
  s_4: { y: -2.25, r: 0.7, phi: 0.3 },     // T50 - LPT outlet
};

export const toLathe = (y: number, r: number, phi: number): [number, number, number] => [r * Math.sin(phi), y, r * Math.cos(phi)];

/**
 * How far a reading is from the typical healthy range, in half-widths of that range:
 * 0 at the centre, 1 at the edge of the 1st–99th percentile band.
 */
export function deviation(s: Sensor, idx: number, smooth = 5): number | null {
  if (s.healthy_high === s.healthy_low) return null;      // constant feature
  const lo = Math.max(0, idx - smooth + 1);
  const win = s.values.slice(lo, idx + 1);
  const v = win.reduce((a, b) => a + b, 0) / win.length;  // short moving average removes sensor noise
  const mid = (s.healthy_high + s.healthy_low) / 2;
  const half = (s.healthy_high - s.healthy_low) / 2;
  return Math.abs(v - mid) / half;
}

export interface ModuleState { id: ModuleId; score: number; label: string; detail: string }

/** Module health from its sensors' deviation (max over sensors, smoothed). */
export function moduleHealth(sensors: Sensor[], idx: number): Record<ModuleId, ModuleState> {
  const byKey = Object.fromEntries(sensors.map((s) => [s.key, s]));
  const out = {} as Record<ModuleId, ModuleState>;
  for (const m of MODULES) {
    const devs = m.sensors.map((k) => (byKey[k] ? deviation(byKey[k], idx) : null)).filter((d): d is number => d !== null);
    const score = devs.length ? Math.max(...devs) : 0;
    out[m.id] = {
      id: m.id, score,
      label: score <= 1 ? 'Typical' : score <= 1.75 ? 'Drifting' : 'Atypical',
      detail: `max deviation ${score.toFixed(2)}× healthy half-range`,
    };
  }
  return out;
}

/** Module attribution: sum of integrated-gradients attributions of its sensors (cycles of RUL). */
export function moduleAttribution(x: Explanation): Record<ModuleId, number> {
  const byKey = Object.fromEntries(x.features.map((f) => [f.key, f.attribution]));
  const out = {} as Record<ModuleId, number>;
  for (const m of MODULES) out[m.id] = m.sensors.reduce((a, k) => a + (byKey[k] ?? 0), 0);
  return out;
}

/** 0 → cyan/green, 1 → amber, ≥2 → red (deviation scale). */
export function healthColor(score: number): string {
  if (score <= 1) return lerpColor('#00E5A0', '#7CFFB2', score);
  if (score <= 2) return lerpColor('#FFB300', '#FF6B4A', score - 1);
  return '#FF0055';
}

/** Negative attribution (lowers RUL) → hot; positive → cool. t in [-1, 1]. */
export function attributionColor(t: number): string {
  return t < 0 ? lerpColor('#2a3550', '#FF2E63', Math.min(1, -t)) : lerpColor('#2a3550', '#00BFFF', Math.min(1, t));
}

export function lerpColor(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (s: number) => Math.round(((pa >> s) & 255) + (((pb >> s) & 255) - ((pa >> s) & 255)) * Math.max(0, Math.min(1, t)));
  return `#${((ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).padStart(6, '0')}`;
}
