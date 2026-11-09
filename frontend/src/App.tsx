import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { DashboardLayout } from './layouts/DashboardLayout';
import { Dashboard } from './pages/Dashboard';
import { Fleet } from './pages/Fleet';
import { EngineMonitor } from './pages/EngineMonitor';
import { ModelDeployment } from './pages/ModelDeployment';
import { FederatedLearning } from './pages/FederatedLearning';
import { ExplainableAI } from './pages/ExplainableAI';
import { Maintenance } from './pages/Maintenance';
import { ResultsExplorer } from './pages/ResultsExplorer';
import { Alerts } from './pages/Alerts';
import { Settings } from './pages/Settings';
import { LiveTraining } from './pages/LiveTraining';
import { Upload } from './pages/Upload';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="fleet" element={<Fleet />} />
          <Route path="engine" element={<EngineMonitor />} />
          <Route path="xai" element={<ExplainableAI />} />
          <Route path="maintenance" element={<Maintenance />} />
          <Route path="alerts" element={<Alerts />} />
          <Route path="fl" element={<FederatedLearning />} />
          <Route path="results" element={<ResultsExplorer />} />
          <Route path="model" element={<ModelDeployment />} />
          <Route path="settings" element={<Settings />} />
          <Route path="train" element={<LiveTraining />} />
          <Route path="upload" element={<Upload />} />
          {/* old routes */}
          <Route path="aircraft" element={<Navigate to="/engine" replace />} />
          <Route path="analytics" element={<Navigate to="/results" replace />} />
          <Route path="edge" element={<Navigate to="/model" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
