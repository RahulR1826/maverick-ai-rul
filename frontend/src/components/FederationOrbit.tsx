import { motion } from 'framer-motion';

/**
 * Illustration of the training set-up used for the served model:
 * K = 5 clients exchanging model updates with an aggregation server (FedAvg).
 * Decorative: packets do not represent live traffic.
 */
export function FederationOrbit({ size = 300, k = 5, caption = 'K = 5 CLIENTS · R = 20 ROUNDS · E = 5', paused = false }: { size?: number; k?: number; caption?: string; paused?: boolean }) {
  const c = size / 2, R = size / 2 - 34;
  const nodes = Array.from({ length: k }, (_, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / k;
    return { x: c + R * Math.cos(a), y: c + R * Math.sin(a), i };
  });
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
      <defs>
        <radialGradient id="core" cx="50%" cy="50%" r="50%"><stop offset="0" stopColor="#00FFFF" stopOpacity="0.9" /><stop offset="1" stopColor="#00BFFF" stopOpacity="0" /></radialGradient>
      </defs>
      <motion.circle cx={c} cy={c} r={R} fill="none" stroke="rgba(0,191,255,0.18)" strokeDasharray="3 6"
        animate={{ rotate: 360 }} style={{ originX: '50%', originY: '50%' }} transition={{ duration: 60, repeat: Infinity, ease: 'linear' }} />
      <circle cx={c} cy={c} r={R * 0.55} fill="none" stroke="rgba(0,191,255,0.08)" />
      {nodes.map((n) => (
        <g key={n.i}>
          <line x1={c} y1={c} x2={n.x} y2={n.y} stroke="rgba(0,191,255,0.25)" />
          {/* update travelling to the server, then the global model back */}
          {!paused && <motion.circle r={3} cx={n.x} cy={n.y} fill="#00E676" initial={{ cx: n.x, cy: n.y, opacity: 0 }} animate={{ cx: [n.x, c], cy: [n.y, c], opacity: [0, 1, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, delay: (n.i * 1.8) / k, ease: 'easeIn' }} />}
          {!paused && <motion.circle r={2.5} cx={c} cy={c} fill="#00BFFF" initial={{ cx: c, cy: c, opacity: 0 }} animate={{ cx: [c, n.x], cy: [c, n.y], opacity: [0, 1, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, delay: 0.9 + (n.i * 1.8) / k, ease: 'easeOut' }} />}
          <circle cx={n.x} cy={n.y} r={15} fill="rgba(3,8,16,0.9)" stroke="#00BFFF" strokeOpacity={0.6} />
          <text x={n.x} y={n.y + 4} textAnchor="middle" fill="#cfefff" fontSize={10} fontFamily="Space Mono, monospace">C{n.i + 1}</text>
        </g>
      ))}
      <circle cx={c} cy={c} r={36} fill="url(#core)" />
      <motion.circle cx={c} cy={c} r={22} fill="rgba(3,8,16,0.95)" stroke="#00FFFF" strokeWidth={1.5}
        initial={{ r: 22 }} animate={{ r: [22, 25, 22] }} transition={{ duration: 2.4, repeat: Infinity }} />
      <text x={c} y={c + 4} textAnchor="middle" fill="#fff" fontSize={10} fontWeight={700} letterSpacing={1}>FedAvg</text>
      <text x={c} y={size - 6} textAnchor="middle" fill="#6B8CAE" fontSize={9} letterSpacing={1}>{caption}</text>
    </svg>
  );
}
