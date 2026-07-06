import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './layouts/DashboardLayout';
import { Dashboard } from './pages/Dashboard';
import { Fleet } from './pages/Fleet';
import { AircraftMonitor } from './pages/AircraftMonitor';
import { EdgeAI } from './pages/EdgeAI';
import { FederatedLearning } from './pages/FederatedLearning';
import { ExplainableAI } from './pages/ExplainableAI';
import { Maintenance } from './pages/Maintenance';
import { Analytics } from './pages/Analytics';
import { Alerts } from './pages/Alerts';
import { Settings } from './pages/Settings';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="fleet" element={<Fleet />} />
          <Route path="aircraft" element={<AircraftMonitor />} />
          <Route path="edge" element={<EdgeAI />} />
          <Route path="fl" element={<FederatedLearning />} />
          <Route path="xai" element={<ExplainableAI />} />
          <Route path="maintenance" element={<Maintenance />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="alerts" element={<Alerts />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
