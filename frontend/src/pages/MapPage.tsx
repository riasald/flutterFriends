import "../styles/flutterfriends.css";
import butterflyPng from "../assets/butterfly.png";
import titlePng from "../assets/title.png";
import { useState } from "react";
import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import type { LeafletMouseEvent } from "leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as L.Icon.Default & { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

type Position = {
  lat: number;
  lng: number;
};

type LocationMarkerProps = {
  position: Position | null;
  setPosition: React.Dispatch<React.SetStateAction<Position | null>>;
  setLatInput: React.Dispatch<React.SetStateAction<string>>;
  setLngInput: React.Dispatch<React.SetStateAction<string>>;
};

type ButterflyResponse = {
  job_id: string;
  status: string;
  image_url: string;
  report?: {
    top_species?: Array<{
      rank: number;
      species: string;
      probability: number;
    }>;
    selected?: Array<{
      rank: number;
      quality_score: number;
      foreground_luminance: number;
    }>;
  };
};

function isWithinUS(lat: number, lng: number) {
  return lat >= 24.5 && lat <= 49.5 && lng >= -125 && lng <= -66.5;
}

function LocationMarker({
  position,
  setPosition,
  setLatInput,
  setLngInput,
}: LocationMarkerProps) {
  useMapEvents({
    click(e: LeafletMouseEvent) {
      const lat = e.latlng.lat;
      const lng = e.latlng.lng;

      if (!isWithinUS(lat, lng)) {
        alert("Please select a location within the United States 🇺🇸");
        return;
      }

      const newPos = { lat, lng };

      setPosition(newPos);
      setLatInput(newPos.lat.toFixed(6));
      setLngInput(newPos.lng.toFixed(6));
},
  });

  if (!position) return null;

  return <Marker position={[position.lat, position.lng]} />;
}

function RecenterMap({ position }: { position: Position | null }) {
  const map = useMapEvents({});

  if (position) {
    map.flyTo([position.lat, position.lng], map.getZoom());
  }

  return null;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export default function MapPage() {
  const defaultCenter: Position = {
    lat: 29.652,
    lng: -82.325,
  };

  const [position, setPosition] = useState<Position | null>(defaultCenter);
  const [latInput, setLatInput] = useState(defaultCenter.lat.toFixed(6));
  const [lngInput, setLngInput] = useState(defaultCenter.lng.toFixed(6));
  const [showButterflyModal, setShowButterflyModal] = useState(false);
  const [generatedButterflyUrl, setGeneratedButterflyUrl] = useState<string | null>(null);
  const [report, setReport] = useState<ButterflyResponse["report"] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const displayedButterfly = generatedButterflyUrl ?? butterflyPng;

  const handleSetCoordinates = () => {
    const lat = Number(latInput);
    const lng = Number(lngInput);

    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      setErrorMessage("Please enter valid numbers for latitude and longitude.");
      return;
    }

    if (lat < -90 || lat > 90) {
      setErrorMessage("Latitude must be between -90 and 90.");
      return;
    }

    if (lng < -180 || lng > 180) {
      setErrorMessage("Longitude must be between -180 and 180.");
      return;
    }

    if (!isWithinUS(lat, lng)) {
      setErrorMessage("Please choose a location within the United States 🇺🇸");
      return;
    }

    setErrorMessage(null);
    setPosition({ lat, lng });
  };

  const handleCustomizeButterfly = async () => {
    if (!position || !isWithinUS(position.lat, position.lng)) {
      setErrorMessage("Please select a valid US location before generating a butterfly.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const payload = {
        lat: position.lat,
        lon: position.lng,
        num_candidates: 4,
        num_outputs: 4,
        samples_per_batch: 1,
      };

      const response = await fetch(`${API_BASE_URL}/generate-butterfly`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate butterfly");
      }

      const result = data as ButterflyResponse;
      setGeneratedButterflyUrl(result.image_url);
      setReport(result.report ?? null);
    } catch (error) {
      console.error("Error generating butterfly:", error);
      setErrorMessage(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="ff-page ff-babyYellow">
      <header className="ff-headerSoft ff-headerRow">
        <img className="ff-titleImg" src={titlePng} alt="FlutterFriends" />

        <button
          className="ff-pill ff-pillBig"
          onClick={() => (window.location.href = "/login")}
        >
          Log Out
        </button>
      </header>

      <main className="ff-mainSoft">
        <div className="ff-twoColumnLayout">
          <section className="ff-bubble ff-butterflyCard">
            <button
              className="ff-ctaSoft"
              onClick={handleCustomizeButterfly}
              disabled={isLoading || !position}
            >
              {isLoading ? "Generating Butterfly..." : "Customize Your Butterfly"}
            </button>

            <div className="ff-location">
              📍 Location:
              <br />
              {position
                ? `${position.lat.toFixed(4)}°, ${position.lng.toFixed(4)}°`
                : "Click on the map or type coordinates"}
            </div>

            <div className="ff-coordInputs">
              <label className="ff-coordLabel">
                Latitude
                <input
                  className="ff-coordInput"
                  type="number"
                  step="any"
                  value={latInput}
                  onChange={(e) => setLatInput(e.target.value)}
                  placeholder="e.g. 29.6516"
                />
              </label>

              <label className="ff-coordLabel">
                Longitude
                <input
                  className="ff-coordInput"
                  type="number"
                  step="any"
                  value={lngInput}
                  onChange={(e) => setLngInput(e.target.value)}
                  placeholder="e.g. -82.3248"
                />
              </label>

              <button className="ff-orangeBtn" onClick={handleSetCoordinates}>
                Set Coordinates
              </button>
            </div>

            {errorMessage && (
              <div style={{ color: "#b00020", marginTop: "12px" }}>
                Error: {errorMessage}
              </div>
            )}

            <div className="ff-butterflyWrapClear">
              <img
                src={displayedButterfly}
                alt="Butterfly"
                className="ff-butterflyBig"
              />
            </div>

            <button
              className="ff-smallBtnSoft"
              onClick={() => setShowButterflyModal(true)}
            >
              ⤢
            </button>

            {report?.top_species && report.top_species.length > 0 && (
              <div style={{ marginTop: "16px", textAlign: "left" }}>
                <strong>Top species:</strong>
                <ul>
                  {report.top_species.slice(0, 5).map((item) => (
                    <li key={`${item.rank}-${item.species}`}>
                      {item.rank}. {item.species} ({item.probability.toFixed(4)})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          <section className="ff-bubble ff-mapBubble">
            <MapContainer
              center={[defaultCenter.lat, defaultCenter.lng]}
              zoom={5}
              maxBounds={[
                [24.5, -125],
                [49.5, -66.5],
              ]}
              maxBoundsViscosity={1.0}
              className="ff-mapFrameBig"
            >
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              <RecenterMap position={position} />

              <LocationMarker
                position={position}
                setPosition={setPosition}
                setLatInput={setLatInput}
                setLngInput={setLngInput}
              />
            </MapContainer>
          </section>
        </div>
      </main>

      {showButterflyModal && (
        <div
          className="ff-modalOverlay"
          onClick={() => setShowButterflyModal(false)}
        >
          <div
            className="ff-modalContent"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="ff-modalClose"
              onClick={() => setShowButterflyModal(false)}
            >
              ✕
            </button>

            <img
              src={displayedButterfly}
              alt="Enlarged Butterfly"
              className="ff-butterflyModalImg"
            />
          </div>
        </div>
      )}
    </div>
  );
}