import { create } from 'zustand';

export interface Aircraft {
  id: string;
  type: string;
  health: number;
  rul: number;
  risk: 'Low' | 'Medium' | 'High' | 'Critical';
  phase: string;
  edgeStatus: 'Active' | 'Offline' | 'Syncing';
  modelVersion: string;
  maintenanceStatus: 'OK' | 'Scheduled' | 'Overdue';
}

interface AppState {
  fleet: Aircraft[];
  sensors: Record<string, number>;
  selectedAircraftId: string | null;
  selectAircraft: (id: string) => void;
  updateSensors: () => void;
}

const generateFleet = (): Aircraft[] => [
  { id: 'AX-101', type: 'Boeing 737', health: 85, rul: 84, risk: 'Low', phase: 'Cruise', edgeStatus: 'Active', modelVersion: 'v2.7.3', maintenanceStatus: 'OK' },
  { id: 'AX-102', type: 'Boeing 737', health: 58, rul: 32, risk: 'High', phase: 'Takeoff', edgeStatus: 'Active', modelVersion: 'v2.7.3', maintenanceStatus: 'Scheduled' },
  { id: 'BX-214', type: 'Airbus A320', health: 22, rul: 12, risk: 'Critical', phase: 'Landing', edgeStatus: 'Active', modelVersion: 'v2.7.3', maintenanceStatus: 'Scheduled' },
  { id: 'CX-319', type: 'Boeing 777', health: 92, rul: 110, risk: 'Low', phase: 'Cruise', edgeStatus: 'Syncing', modelVersion: 'v2.7.2', maintenanceStatus: 'OK' },
  { id: 'EX-511', type: 'Boeing 787', health: 45, rul: 28, risk: 'High', phase: 'Descent', edgeStatus: 'Active', modelVersion: 'v2.7.3', maintenanceStatus: 'Overdue' },
];

export const useStore = create<AppState>((set) => ({
  fleet: generateFleet(),
  sensors: {
    'Fan Speed (N1)': 5200,
    'Compressor Speed (N2)': 14800,
    'Fuel Flow': 3820,
    'Oil Pressure': 60,
    'Oil Temp': 62,
    'EGT': 1050,
    'Vibration': 0.42,
    'Pressure Ratio': 14.6,
  },
  selectedAircraftId: 'AX-102',
  selectAircraft: (id) => set({ selectedAircraftId: id }),
  updateSensors: () => set((state) => {
    const noise = (val: number, variance: number) => val + (Math.random() - 0.5) * variance;
    return {
      sensors: {
        'Fan Speed (N1)': noise(5200, 50),
        'Compressor Speed (N2)': noise(14800, 100),
        'Fuel Flow': noise(3820, 40),
        'Oil Pressure': noise(60, 2),
        'Oil Temp': noise(62, 1),
        'EGT': noise(1050, 10),
        'Vibration': noise(0.42, 0.05),
        'Pressure Ratio': noise(14.6, 0.2),
      }
    };
  })
}));
