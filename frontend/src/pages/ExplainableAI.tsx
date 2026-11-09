import { BrainCircuit } from 'lucide-react';
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useNavigate } from 'react-router-dom';
import { useEngine, useExplain } from '../hooks/useData';
import { useStore } from '../store/useStore';
import { Loadable, Note, PageHeader, Panel, fmt, tooltipStyle } from '../components/ui';
import type { Explanation } from '../services/api';
import { engineLabel } from '../services/api';

function Heatmap({ h }: { h: Explanation['heatmap'] }) {
  const max = Math.max(1e-9, ...h.values.flat().map(Math.abs));
  const cw = 18, rh = 16, left = 64, top = 4;
  const color = (v: number) => {
    const a = Math.min(1, Math.abs(v) / max);
    return v >= 0 ? `rgba(0,191,255,${a})` : `rgba(255,107,74,${a})`;
  };
  return (
    <div className="overflow-x-auto">
      <svg width={left + h.cycles.length * cw + 4} height={top + h.features.length * rh + 22} role="img" aria-label="Attribution heatmap by sensor and cycle">
        {h.features.map((f, i) => (
          <g key={f}>
            <text x={left - 6} y={top + i * rh + 12} fill="#9fb4c8" fontSize={10} textAnchor="end">{f}</text>
            {h.values[i].map((v, j) => (
              <rect key={j} x={left + j * cw} y={top + i * rh} width={cw - 1} height={rh - 1} fill={color(v)} stroke="rgba(255,255,255,0.04)">
                <title>{`${f}, cycle ${h.cycles[j]}: ${v.toFixed(3)} cycles`}</title>
              </rect>
            ))}
          </g>
        ))}
        {h.cycles.map((c, j) => (j % 5 === 0 || j === h.cycles.length - 1) && (
          <text key={c} x={left + j * cw + cw / 2} y={top + h.features.length * rh + 14} fill="#6B8CAE" fontSize={10} textAnchor="middle">{c}</text>
        ))}
      </svg>
    </div>
  );
}

function View({ x }: { x: Explanation }) {
  const data = x.features.map((f) => ({ name: f.name, value: f.attribution, description: f.description }));
  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 relative z-10">
        <div className="glass-panel p-5"><p className="text-[10px] text-muted uppercase tracking-widest">Prediction</p><p className="text-3xl font-mono font-bold text-primary">{fmt(x.prediction, 1)} <span className="text-sm">cycles</span></p></div>
        <div className="glass-panel p-5"><p className="text-[10px] text-muted uppercase tracking-widest">Baseline (mean training window)</p><p className="text-3xl font-mono font-bold text-gray-300">{fmt(x.baseline_prediction, 1)}</p></div>
        <div className="glass-panel p-5"><p className="text-[10px] text-muted uppercase tracking-widest">Sum of attributions</p><p className="text-3xl font-mono font-bold text-white">{fmt(x.sum_of_attributions, 1)}</p>
          <p className="text-[11px] text-muted">completeness error {fmt(x.completeness_error, 3)} cycles</p></div>
      </div>
      <Panel title="Contribution of each input to the predicted RUL (cycles)"
        note="Positive (blue): the sensor's recent values raise the predicted remaining life above the baseline. Negative (orange): they lower it, i.e. signal degradation. Values are summed over the 30-cycle window and averaged over the 5 ensemble members.">
        <div className="h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis type="number" stroke="#6B8CAE" fontSize={11} />
              <YAxis type="category" dataKey="name" stroke="#9fb4c8" fontSize={11} width={70} />
              <ReferenceLine x={0} stroke="#6B8CAE" />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown) => [`${fmt(v as number, 2)} cycles`, 'Attribution']}
                labelFormatter={(l, p) => `${l} — ${(p?.[0]?.payload as { description?: string })?.description ?? ''}`} />
              <Bar dataKey="value" isAnimationActive={false}>
                {data.map((d) => <Cell key={d.name} fill={d.value >= 0 ? '#00BFFF' : '#FF6B4A'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>
      <Panel title="When in the window each input mattered" note="Each cell is one sensor at one cycle of the input window. Hover a cell for its value.">
        <Heatmap h={x.heatmap} />
      </Panel>
    </>
  );
}

export function ExplainableAI() {
  const { selectedEngineId, cycleByEngine } = useStore();
  const engine = useEngine(selectedEngineId);
  const last = engine.data?.trajectory.at(-1)?.cycle;
  const cycle = cycleByEngine[selectedEngineId] ?? last;
  const x = useExplain(selectedEngineId, cycle);
  const navigate = useNavigate();
  return (
    <div className="space-y-6 pb-20 relative">
      <PageHeader icon={BrainCircuit} title="Explainable AI"
        subtitle={`Integrated-gradients attribution for ${engineLabel(selectedEngineId)}${cycle ? ` at cycle ${cycle}` : ''}. Change engine or cycle in the Engine Monitor.`}
        right={<button onClick={() => navigate('/engine')} className="px-4 py-2 rounded border border-primary text-primary text-sm font-bold">Open Engine Monitor</button>} />
      <Loadable q={x} label="explanation">{(d) => <View x={d} />}</Loadable>
      <Note>Integrated gradients (Sundararajan et al., 2017) attributes the difference between the prediction and the baseline prediction to each input. Attributions show what the model relies on. They are not proof of a physical cause.</Note>
    </div>
  );
}
