import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const fetchFleet = async () => {
  const response = await axios.get(`${API_URL}/fleet`);
  return response.data;
};

export const fetchAircraft = async (id: string) => {
  const response = await axios.get(`${API_URL}/aircraft/${id}`);
  return response.data;
};

export const fetchFlStatus = async () => {
  const response = await axios.get(`${API_URL}/fl-status`);
  return response.data;
};

export const fetchEdgeStatus = async () => {
  const response = await axios.get(`${API_URL}/edge-status`);
  return response.data;
};

export const fetchAnalytics = async () => {
  const response = await axios.get(`${API_URL}/analytics`);
  return response.data;
};
