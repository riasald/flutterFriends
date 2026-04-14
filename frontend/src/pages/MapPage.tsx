import { LogoButterflyMark } from "../components/LogoButterflyMark";
import "../styles/mapPageRedesign.css";
import logoPng from "../assets/FlutterFriendsLogo.png";
import {
  useEffect,
  useMemo,
  useRef,
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

type SuggestItem = {
  latitude: number;
  longitude: number;
  label: string;
  detail: string | null;
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

function formatApiErrorBody(data: unknown, fallbackMessage = "Failed to generate butterfly"): string {
  if (!data || typeof data !== "object") return fallbackMessage;
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
  return fallbackMessage;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function triggerBlobDownload(blob: Blob, filename: string) {
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(objectUrl);
}

function fileExtensionForImageMime(mime: string): "jpg" | "png" {
  const m = mime.toLowerCase();
  if (m.includes("jpeg") || m.includes("jpg")) return "jpg";
  return "png";
}

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

function EnlargeOutwardArrowsIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="M6 6 2.5 2.5m0 0L4.5 2.5M2.5 2.5 2.5 4.5M8 8l3.5 3.5m0 0L9.5 11.5M11.5 11.5 11.5 9.5"
        stroke="currentColor"
        strokeWidth="1.15"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function DownloadTrayIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

/** Top-down 4-wing loader: fore + hind per side, flat dusty pink + rose veins (reference art). */
function LoadingButterflyPlaceholder() {
  const wingFill = "#f2c4d4";
  const wingStroke = "#b86d88";
  const veinStroke = "#c47a92";
  const bodyStroke = "#4a1f30";
  const bodyFill = "#5c2838";

  return (
    <div className="mr-butterfly-loading-slot" role="status" aria-live="polite" aria-label="Creating your butterfly">
      <div className="mr-loading-bfly-track">
        <div className="mr-loading-bfly-mover">
          <svg className="mr-loading-bfly-svg" viewBox="-38 -24 76 48" fill="none" aria-hidden>
            <g className="mr-loading-bfly-wing-l" strokeLinecap="round" strokeLinejoin="round">
              <path
                d="M 0 -8.5 C -9 -15 -33 -14 -31.5 -4.5 C -29 4 -13 7.5 -2.5 5.5 C -0.5 4.5 0 1 0 -8.5 Z"
                fill={wingFill}
                stroke={wingStroke}
                strokeWidth="0.55"
              />
              <path
                d="M 0 3.5 C -17 5.5 -27 12.5 -23.5 17.5 C -17 20.5 -6.5 15.5 0 11 Z"
                fill={wingFill}
                stroke={wingStroke}
                strokeWidth="0.55"
              />
              <path
                d="M 0 -5 Q -14 -10 -24 -5 M 0 -1.5 Q -16 -2 -22 2 M -6 6 Q -14 8 -18 12"
                stroke={veinStroke}
                strokeWidth="0.38"
              />
            </g>
            <g className="mr-loading-bfly-wing-r" strokeLinecap="round" strokeLinejoin="round">
              <path
                d="M 0 -8.5 C 9 -15 33 -14 31.5 -4.5 C 29 4 13 7.5 2.5 5.5 C 0.5 4.5 0 1 0 -8.5 Z"
                fill={wingFill}
                stroke={wingStroke}
                strokeWidth="0.55"
              />
              <path
                d="M 0 3.5 C 17 5.5 27 12.5 23.5 17.5 C 17 20.5 6.5 15.5 0 11 Z"
                fill={wingFill}
                stroke={wingStroke}
                strokeWidth="0.55"
              />
              <path
                d="M 0 -5 Q 14 -10 24 -5 M 0 -1.5 Q 16 -2 22 2 M 6 6 Q 14 8 18 12"
                stroke={veinStroke}
                strokeWidth="0.38"
              />
            </g>
            <ellipse cx="0" cy="1" rx="1.35" ry="10.5" fill={bodyFill} stroke={bodyStroke} strokeWidth="0.35" />
            <path
              d="M 0 -9.2 Q -1.8 -14 -3.8 -17.2 M 0 -9.2 Q 1.8 -14 3.8 -17.2"
              stroke={bodyStroke}
              strokeWidth="0.65"
            />
            <circle cx="-4.1" cy="-17.4" r="1.05" fill={bodyFill} />
            <circle cx="4.1" cy="-17.4" r="1.05" fill={bodyFill} />
          </svg>
        </div>
      </div>
      <p className="mr-loading-bfly-label">Creating your butterfly…</p>
    </div>
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
  const [placeQuery, setPlaceQuery] = useState("");
  const [isSettingLocation, setIsSettingLocation] = useState(false);
  const [suggestions, setSuggestions] = useState<SuggestItem[]>([]);
  const [suggestOpen, setSuggestOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);
  const [suggestNonce, setSuggestNonce] = useState(0);
  const placeInputRef = useRef<HTMLInputElement>(null);

  const displayedButterfly = generatedButterflyUrl ?? logoPng;

  const coordsPillText = useMemo(() => {
    if (!position) return "—";
    return formatCoordsPill(position.lat, position.lng);
  }, [position]);

  useEffect(() => {
    if (!position) return undefined;
    const ac = new AbortController();
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const response = await fetch(
            `${API_BASE_URL}/api/reverse-geocode?lat=${encodeURIComponent(String(position.lat))}&lon=${encodeURIComponent(String(position.lng))}`,
            { signal: ac.signal },
          );
          if (!response.ok) return;
          const data = (await response.json()) as { label?: unknown };
          if (typeof data.label === "string" && data.label.trim()) {
            setPlaceQuery(data.label.trim());
          }
        } catch {
          /* aborted or network */
        }
      })();
    }, 400);
    return () => {
      window.clearTimeout(timer);
      ac.abort();
    };
  }, [position?.lat, position?.lng]);

  useEffect(() => {
    const input = placeInputRef.current;
    if (!input || document.activeElement !== input) {
      return undefined;
    }
    const raw = placeQuery.trim();
    if (raw.length < 2) {
      setSuggestions([]);
      setSuggestOpen(false);
      return undefined;
    }
    const ac = new AbortController();
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const response = await fetch(
            `${API_BASE_URL}/api/geocode-suggest?q=${encodeURIComponent(raw)}`,
            { signal: ac.signal },
          );
          const text = await response.text();
          let data: unknown = null;
          try {
            data = text ? (JSON.parse(text) as unknown) : null;
          } catch {
            return;
          }
          if (!response.ok) return;
          const payload = data as { results?: unknown };
          const list = Array.isArray(payload.results) ? payload.results : [];
          const next: SuggestItem[] = [];
          for (const entry of list) {
            if (!entry || typeof entry !== "object") continue;
            const o = entry as Record<string, unknown>;
            const lat = Number(o.latitude);
            const lng = Number(o.longitude);
            const label = typeof o.label === "string" ? o.label : "";
            if (!Number.isFinite(lat) || !Number.isFinite(lng) || !label.trim()) continue;
            next.push({
              latitude: lat,
              longitude: lng,
              label: label.trim(),
              detail: typeof o.detail === "string" ? o.detail : null,
            });
          }
          if (ac.signal.aborted) return;
          setSuggestions(next);
          setSuggestOpen(next.length > 0);
          setHighlightIndex(0);
        } catch {
          if (!ac.signal.aborted) {
            setSuggestions([]);
            setSuggestOpen(false);
          }
        }
      })();
    }, 280);
    return () => {
      window.clearTimeout(timer);
      ac.abort();
    };
  }, [placeQuery, suggestNonce]);

  const applySuggestion = (item: SuggestItem) => {
    setPlaceQuery(item.label);
    setLatInput(item.latitude.toFixed(6));
    setLngInput(item.longitude.toFixed(6));
    setPosition({ lat: item.latitude, lng: item.longitude });
    setSuggestions([]);
    setSuggestOpen(false);
    setHighlightIndex(0);
    setErrorMessage(null);
  };

  /** Applies current latitude/longitude fields if they form a valid US map point. */
  const applyLatLngFromInputs = (): boolean => {
    const lat = Number(latInput);
    const lng = Number(lngInput);
    if (Number.isNaN(lat) || Number.isNaN(lng)) return false;
    if (lat < -90 || lat > 90) return false;
    if (lng < -180 || lng > 180) return false;
    if (!isWithinUS(lat, lng)) return false;
    setSuggestions([]);
    setSuggestOpen(false);
    setPosition({ lat, lng });
    return true;
  };

  const refreshPlaceLabelFromCoords = async (lat: number, lng: number) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/reverse-geocode?lat=${encodeURIComponent(String(lat))}&lon=${encodeURIComponent(String(lng))}`,
      );
      if (!response.ok) return;
      const data = (await response.json()) as { label?: unknown };
      if (typeof data.label === "string" && data.label.trim()) {
        setPlaceQuery(data.label.trim());
      }
    } catch {
      /* ignore */
    }
  };

  const handleSetLocation = async () => {
    const trimmed = placeQuery.trim();

    if (trimmed.length < 2) {
      if (!applyLatLngFromInputs()) {
        setErrorMessage("Enter a place above, or valid numbers for latitude and longitude.");
        return;
      }
      setErrorMessage(null);
      return;
    }

    setIsSettingLocation(true);
    setErrorMessage(null);
    setSuggestions([]);
    setSuggestOpen(false);

    try {
      const url = `${API_BASE_URL}/api/geocode?q=${encodeURIComponent(trimmed)}`;
      const response = await fetch(url);
      const text = await response.text();
      let data: unknown = null;
      try {
        data = text ? (JSON.parse(text) as unknown) : null;
      } catch {
        data = null;
      }

      if (response.ok && data && typeof data === "object") {
        const payload = data as { latitude?: unknown; longitude?: unknown; label?: unknown };
        const lat = Number(payload.latitude);
        const lng = Number(payload.longitude);
        if (Number.isFinite(lat) && Number.isFinite(lng) && isWithinUS(lat, lng)) {
          setLatInput(lat.toFixed(6));
          setLngInput(lng.toFixed(6));
          setPosition({ lat, lng });
          if (typeof payload.label === "string" && payload.label.trim()) {
            setPlaceQuery(payload.label.trim());
          }
          return;
        }
      }

      /*
       * Forward geocode often fails for long reverse-geocoded labels (POI + street).
       * If the text search fails but lat/lng already match the map, use those coordinates.
       */
      if (applyLatLngFromInputs()) {
        const lat = Number(latInput);
        const lng = Number(lngInput);
        await refreshPlaceLabelFromCoords(lat, lng);
        return;
      }

      const fallbackMsg =
        data && typeof data === "object"
          ? formatApiErrorBody(data, "Location search failed.")
          : "No places matched that search.";
      setErrorMessage(fallbackMsg);
    } catch (error) {
      console.error("Geocode error:", error);
      if (applyLatLngFromInputs()) {
        const lat = Number(latInput);
        const lng = Number(lngInput);
        await refreshPlaceLabelFromCoords(lat, lng);
        setErrorMessage(null);
        return;
      }
      setErrorMessage(error instanceof Error ? error.message : "Could not look up that location.");
    } finally {
      setIsSettingLocation(false);
    }
  };

  const commitSuggestionOrSetLocation = async () => {
    if (suggestOpen && suggestions.length > 0) {
      const pick = suggestions[Math.min(highlightIndex, suggestions.length - 1)];
      if (pick) {
        applySuggestion(pick);
        return;
      }
    }
    await handleSetLocation();
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
    setPlaceQuery("");
    setSuggestions([]);
    setSuggestOpen(false);
  };

  const handleDownloadButterfly = async () => {
    if (!generatedButterflyUrl) {
      setErrorMessage("Create a butterfly first, then you can download it.");
      return;
    }

    setErrorMessage(null);

    try {
      const response = await fetch(generatedButterflyUrl);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const ext = fileExtensionForImageMime(blob.type || "image/png");
      triggerBlobDownload(blob, `flutterfriends-butterfly.${ext}`);
    } catch {
      setErrorMessage("Could not download the image. Check your connection or try opening it in a new tab.");
    }
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
          <LogoButterflyMark className="mr-logo-butterfly" />
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
              Click the map to move the pin (the place field updates automatically), type a ZIP or city for suggestions, or
              set latitude and longitude and press <strong>Set location</strong>.
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
            <div className="mr-or-divider" aria-hidden>
              <span className="mr-or-line" />
              <span className="mr-or-word">or</span>
              <span className="mr-or-line" />
            </div>

            <div className="mr-place-field">
              <label className="mr-coord-label" htmlFor="mr-place-search">
                City, state, or ZIP
              </label>
              <div className="mr-place-autocomplete" id="mr-place-wrap">
                <input
                  ref={placeInputRef}
                  id="mr-place-search"
                  className="mr-coord-input mr-place-input"
                  type="text"
                  inputMode="text"
                  autoComplete="off"
                  placeholder="Enter zip code or city, state"
                  value={placeQuery}
                  role="combobox"
                  aria-expanded={suggestOpen}
                  aria-controls="mr-place-suggest-list"
                  aria-autocomplete="list"
                  onChange={(e) => setPlaceQuery(e.target.value)}
                  onFocus={() => setSuggestNonce((n) => n + 1)}
                  onBlur={(e) => {
                    const next = e.relatedTarget as Node | null;
                    const wrap = document.getElementById("mr-place-wrap");
                    if (next && wrap?.contains(next)) return;
                    window.setTimeout(() => setSuggestOpen(false), 180);
                  }}
                  onKeyDown={(e) => {
                    if (suggestOpen && suggestions.length > 0) {
                      if (e.key === "ArrowDown") {
                        e.preventDefault();
                        setHighlightIndex((i) => Math.min(i + 1, suggestions.length - 1));
                        return;
                      }
                      if (e.key === "ArrowUp") {
                        e.preventDefault();
                        setHighlightIndex((i) => Math.max(i - 1, 0));
                        return;
                      }
                      if (e.key === "Escape") {
                        e.preventDefault();
                        setSuggestOpen(false);
                        return;
                      }
                      if (e.key === "Enter") {
                        e.preventDefault();
                        const pick = suggestions[Math.min(highlightIndex, suggestions.length - 1)];
                        if (pick) applySuggestion(pick);
                        return;
                      }
                    }
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void commitSuggestionOrSetLocation();
                    }
                  }}
                  disabled={isSettingLocation || isLoading}
                />
                {suggestOpen && suggestions.length > 0 ? (
                  <ul id="mr-place-suggest-list" className="mr-suggest-list" role="listbox">
                    {suggestions.map((item, idx) => (
                      <li key={`${item.latitude},${item.longitude},${item.label}`} role="presentation">
                        <button
                          type="button"
                          role="option"
                          aria-selected={idx === highlightIndex}
                          className={`mr-suggest-item${idx === highlightIndex ? " mr-suggest-item--active" : ""}`}
                          onMouseDown={(ev) => {
                            ev.preventDefault();
                            applySuggestion(item);
                          }}
                          onMouseEnter={() => setHighlightIndex(idx)}
                        >
                          <span className="mr-suggest-item-label">{item.label}</span>
                          {item.detail && item.detail !== item.label ? (
                            <span className="mr-suggest-item-detail">{item.detail}</span>
                          ) : null}
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
              <button
                type="button"
                className="mr-set-btn"
                onClick={() => void commitSuggestionOrSetLocation()}
                disabled={isSettingLocation || isLoading}
              >
                {isSettingLocation ? "Setting location…" : "Set location"}
              </button>
            </div>
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
            {isLoading ? (
              <LoadingButterflyPlaceholder />
            ) : (
              <>
                <button
                  type="button"
                  className="mr-butterfly-preview-btn"
                  onClick={() => setShowButterflyModal(true)}
                  aria-label="Enlarge butterfly image"
                >
                  <img src={displayedButterfly} alt="" className="mr-butterfly-img" />
                </button>
                <p className="mr-enlarge-hint">
                  <EnlargeOutwardArrowsIcon />
                  <span>Click to enlarge</span>
                </p>
              </>
            )}
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
            <button
              type="button"
              className="mr-download-btn"
              onClick={handleDownloadButterfly}
              disabled={!generatedButterflyUrl || isLoading}
            >
              <DownloadTrayIcon />
              Download
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
