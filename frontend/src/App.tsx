import { Navigate, Route, Routes } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import MapPage from "./pages/MapPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/map" element={<MapPage />} />
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}