import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

/** Keeps one failing page (or 3D view) from taking down the whole app. */
export class ErrorBoundary extends Component<{ children: ReactNode; label?: string }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[${this.props.label ?? 'page'}]`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="glass-panel p-8 flex flex-col items-center text-center gap-3 relative z-10">
        <AlertTriangle className="w-8 h-8 text-[#FFB300]" />
        <div className="text-white font-bold">This {this.props.label ?? 'page'} hit an error</div>
        <div className="text-xs text-muted font-mono max-w-xl break-words">{this.state.error.message}</div>
        <button onClick={() => this.setState({ error: null })}
          className="mt-2 px-4 py-2 rounded border border-primary text-primary text-sm font-bold flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> Try again
        </button>
      </div>
    );
  }
}

/** True if the browser can create a WebGL context (needed for the 3D twin). */
export function hasWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}
