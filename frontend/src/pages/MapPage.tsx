import "../styles/mapPageRedesign.css";
import logoPng from "../assets/FlutterFriendsLogo.png";
import {
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { useNavigate } from "react-router-dom";
import { MapContainer, TileLayer, Marker, useMap, useMapEvents } from "react-leaflet";
import type { LeafletMouseEvent } from "leaflet";
import L from "leaflet";

const PIN_ICON_HTML =
  '<div style="width:20px;height:20px;border-radius:50%;background:#f5a8c0;border:3px solid #fff;box-shadow:0 2px 10px rgba(240,150,190,0.5)"></div>';

const pinIcon = L.divIcon({
  className: "mr-leaflet-divicon",
  html: PIN_ICON_HTML,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});

type Position = {
  lat: number;
  lng: number;
};

type LocationMarkerProps = {
  position: Position | null;
  setPosition: Dispatch<SetStateAction<Position | null>>;
  setLatInput: Dispatch<SetStateAction<string>>;
  setLngInput: Dispatch<SetStateAction<string>>;
};

type ButterflyResponse = {
  job_id?: string;
  status?: string;
  image_url?: string;
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

function formatCoordsPill(lat: number, lng: number) {
  const sign = lng < 0 ? "–" : "";
  return `${lat.toFixed(4)}°, ${sign}${Math.abs(lng).toFixed(4)}°`;
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

  return (
    <Marker
      position={[position.lat, position.lng]}
      icon={pinIcon}
      draggable
      eventHandlers={{
        dragend: (event) => {
          const ll = event.target.getLatLng();
          if (!isWithinUS(ll.lat, ll.lng)) {
            alert("Please select a location within the United States 🇺🇸");
            event.target.setLatLng([position.lat, position.lng]);
            return;
          }
          setPosition({ lat: ll.lat, lng: ll.lng });
          setLatInput(ll.lat.toFixed(6));
          setLngInput(ll.lng.toFixed(6));
        },
      }}
    />
  );
}

function RecenterMap({ position }: { position: Position | null }) {
  const map = useMap();

  useEffect(() => {
    if (!position) return;
    map.flyTo([position.lat, position.lng], map.getZoom());
  }, [map, position]);

  return null;
}

function formatApiErrorBody(data: unknown): string {
  if (!data || typeof data !== "object") return "Failed to generate butterfly";
  const detail = (data as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((entry) => {
        if (entry && typeof entry === "object" && "msg" in entry) {
          return String((entry as { msg: unknown }).msg);
        }
        return JSON.stringify(entry);
      })
      .join("; ");
  }
  return "Failed to generate butterfly";
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function CtaButterflyIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="#8a3055"
      strokeWidth="1.45"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M9.5 2.5 8 0.75M14.5 2.5 16 0.75" />
      <path d="M12 3.25v16.5" />
      <path d="M12 5.75C7 4 2.5 7.5 4 11.75c1.25 3.75 5.75 4.25 8 2.75" />
      <path d="M12 5.75C17 4 21.5 7.5 20 11.75c-1.25 3.75-5.75 4.25-8 2.75" />
      <path d="M12 11.25C7.25 11.75 3.5 14.25 5.25 17.75 6.75 20.25 9.75 19.5 12 17.5" />
      <path d="M12 11.25C16.75 11.75 20.5 14.25 18.75 17.75 17.25 20.25 14.25 19.5 12 17.5" />
    </svg>
  );
}

function ExpandCornerIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="#d499b0" strokeWidth="1.2" aria-hidden>
      <path d="M7 1h4v4M5 7L11 1M1 5v6h6" />
    </svg>
  );
}

export default function MapPage() {
  const navigate = useNavigate();

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

  const displayedButterfly = generatedButterflyUrl ?? logoPng;

  const coordsPillText = useMemo(() => {
    if (!position) return "—";
    return formatCoordsPill(position.lat, position.lng);
  }, [position]);

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

  const handleCancelButterfly = () => {
    setIsLoading(false);
    setShowButterflyModal(false);
    setGeneratedButterflyUrl(null);
    setReport(null);
    setErrorMessage(null);
    setPosition({ ...defaultCenter });
    setLatInput(defaultCenter.lat.toFixed(6));
    setLngInput(defaultCenter.lng.toFixed(6));
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
        latitude: position.lat,
        longitude: position.lng,
        visual_features: [] as string[],
      };

      const response = await fetch(`${API_BASE_URL}/api/generate-butterfly`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const text = await response.text();
      let data: unknown = null;

      try {
        data = text ? (JSON.parse(text) as unknown) : null;
      } catch {
        throw new Error(text.slice(0, 200) || "Invalid JSON from server");
      }

      if (!response.ok) {
        throw new Error(formatApiErrorBody(data));
      }

      const result = data as ButterflyResponse;
      const imageUrl = result.image_url;

      if (!imageUrl) {
        throw new Error("Response did not include image_url");
      }

      setGeneratedButterflyUrl(imageUrl);
      setReport(result.report ?? null);
    } catch (error) {
      console.error("Error generating butterfly:", error);
      setErrorMessage(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <div className="map-page-redesign">
      <header className="mr-topbar">
        <div className="mr-logo">
          <span className="mr-flutter">flutter</span>
          <span className="mr-logo-dot" />
          <span className="mr-friends">friends</span>
        </div>
        <button type="button" className="mr-back-btn" onClick={() => navigate("/")}>
          ← Back
        </button>
      </header>

      <div className="mr-main">
        <aside className="mr-sidebar">
          <div className="mr-sidebar-inner">
          <div className="mr-hint-pill">
            <div className="mr-hint-icon">✦</div>
            <div className="mr-hint-text">
              Click anywhere on the map to place your pin — or enter coordinates manually and press{" "}
              <strong>Set coordinates</strong>.
            </div>
          </div>

          <div className="mr-sidebar-coords-block">
            <div className="mr-section-label">Selected location</div>
            <div className="mr-coords-pill">{coordsPillText}</div>
            <div className="mr-coord-grid">
              <div className="mr-coord-field">
                <div className="mr-coord-label">Latitude</div>
                <input
                  className="mr-coord-input"
                  type="number"
                  step="any"
                  value={latInput}
                  onChange={(e) => setLatInput(e.target.value)}
                  placeholder="e.g. 29.6516"
                />
              </div>
              <div className="mr-coord-field">
                <div className="mr-coord-label">Longitude</div>
                <input
                  className="mr-coord-input"
                  type="number"
                  step="any"
                  value={lngInput}
                  onChange={(e) => setLngInput(e.target.value)}
                  placeholder="e.g. -82.3248"
                />
              </div>
            </div>
            <button type="button" className="mr-set-btn" onClick={handleSetCoordinates}>
              Set coordinates
            </button>
          </div>

          {errorMessage ? (
            <div className="mr-error" role="alert">
              {errorMessage}
            </div>
          ) : null}

          <div className="mr-divider" />

          <div className="mr-butterfly-card">
            <div className="mr-section-label">Your butterfly</div>
            <button
              type="button"
              className="mr-cta-btn mr-cta-btn--in-card"
              onClick={handleCustomizeButterfly}
              disabled={isLoading || !position}
            >
              <CtaButterflyIcon />
              {isLoading ? "Generating…" : "Create your butterfly"}
            </button>
            <img
              src={displayedButterfly}
              alt="Generated or sample butterfly"
              className="mr-butterfly-img"
            />
            <button type="button" className="mr-expand-hint" onClick={() => setShowButterflyModal(true)}>
              <ExpandCornerIcon />
              click to enlarge
            </button>
          </div>

          {report?.top_species && report.top_species.length > 0 ? (
            <>
              <div className="mr-divider" />
              <div className="mr-report">
                <div className="mr-section-label">Top species</div>
                <ul>
                  {report.top_species.slice(0, 5).map((item) => (
                    <li key={`${item.rank}-${item.species}`}>
                      {item.rank}. {item.species} ({item.probability.toFixed(4)})
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : null}
          </div>

          <div className="mr-sidebar-footer">
            <button type="button" className="mr-cancel-btn" onClick={handleCancelButterfly}>
              Cancel
            </button>
          </div>
        </aside>

        <section className="mr-map-panel">
          <div className="mr-map-chip">✦ Click map to place pin</div>
          <MapContainer
            center={[defaultCenter.lat, defaultCenter.lng]}
            zoom={7}
            maxBounds={[
              [24.5, -125],
              [49.5, -66.5],
            ]}
            maxBoundsViscosity={1.0}
            className="mr-map-leaflet"
            scrollWheelZoom
          >
            <TileLayer attribution="© OpenStreetMap" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
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

      {showButterflyModal ? (
        <div className="mr-modal-overlay" onClick={() => setShowButterflyModal(false)} role="presentation">
          <div
            className="mr-modal-content"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="Enlarged butterfly"
          >
            <button type="button" className="mr-modal-close" onClick={() => setShowButterflyModal(false)} aria-label="Close">
              ✕
            </button>
            <img src={displayedButterfly} alt="Enlarged butterfly" className="mr-modal-img" />
          </div>
        </div>
      ) : null}
    </div>
  );
}
