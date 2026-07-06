import { cn } from '../utils';

export const ArcGauge = ({ value, max, label, color, format }: { value: number, max: number, label: string, color: string, format?: (v: number) => string }) => {
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const offset = circumference - (value / max) * arcLength;

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg className="w-40 h-40 rotate-[135deg]" viewBox="0 0 140 140">
        <circle
          cx="70" cy="70" r={radius}
          stroke="rgba(255,255,255,0.05)" strokeWidth="6" fill="none"
          strokeDasharray={`${arcLength} ${circumference - arcLength}`}
        />
        <circle
          cx="70" cy="70" r={radius - 10}
          stroke="rgba(255,255,255,0.15)" strokeWidth="2" fill="none"
          strokeDasharray="2 8" strokeDashoffset={circumference * 0.25}
        />
        <circle
          cx="70" cy="70" r={radius}
          stroke={color} strokeWidth="6" fill="none" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
        <span className="text-4xl font-mono font-bold text-white tabular-nums tracking-tighter" style={{ textShadow: `0 0 10px ${color}` }}>
          {format ? format(value) : value}
        </span>
        <span className="text-[9px] text-muted tracking-widest uppercase mt-1 text-center leading-tight whitespace-pre-wrap">{label}</span>
      </div>
    </div>
  );
};
