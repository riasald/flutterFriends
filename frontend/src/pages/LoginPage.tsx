import { Link } from "react-router-dom";
import { MapContainer, TileLayer } from "react-leaflet";
import "../styles/loginPageRedesign.css";

function DecorativeButterflySvg() {
  return (
    <svg width="110" height="92" viewBox="0 0 150 126" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <ellipse cx="48" cy="44" rx="42" ry="32" fill="#FFD1DC" opacity="0.88" />
      <ellipse cx="102" cy="44" rx="42" ry="32" fill="#FFD1DC" opacity="0.88" />
      <ellipse cx="44" cy="82" rx="30" ry="22" fill="#FFD1DC" opacity="0.75" />
      <ellipse cx="106" cy="82" rx="30" ry="22" fill="#FFD1DC" opacity="0.75" />
      <ellipse cx="48" cy="44" rx="28" ry="20" fill="#FDFD96" opacity="0.72" />
      <ellipse cx="102" cy="44" rx="28" ry="20" fill="#FDFD96" opacity="0.72" />
      <ellipse cx="46" cy="84" rx="18" ry="14" fill="#FDFD96" opacity="0.6" />
      <ellipse cx="104" cy="84" rx="18" ry="14" fill="#FDFD96" opacity="0.6" />
      <ellipse cx="48" cy="44" rx="12" ry="9" fill="#f5a8c0" opacity="0.42" />
      <ellipse cx="102" cy="44" rx="12" ry="9" fill="#f5a8c0" opacity="0.42" />
      <ellipse cx="52" cy="46" rx="5" ry="4" fill="#fff" opacity="0.32" />
      <ellipse cx="98" cy="46" rx="5" ry="4" fill="#fff" opacity="0.32" />
      <rect x="72" y="28" width="6" height="62" rx="3" fill="#3a2010" />
      <rect x="73.5" y="30" width="3" height="58" rx="1.5" fill="#6a4030" opacity="0.55" />
      <line x1="75" y1="32" x2="66" y2="18" stroke="#3a2010" strokeWidth="1.3" strokeLinecap="round" />
      <line x1="75" y1="32" x2="84" y2="18" stroke="#3a2010" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="65" cy="17" r="2.2" fill="#3a2010" />
      <circle cx="85" cy="17" r="2.2" fill="#3a2010" />
    </svg>
  );
}

export default function LoginPage() {
  return (
    <div className="login-page-redesign">
      <div className="login-map-bg">
        <MapContainer
          center={[37.8, -96]}
          zoom={4}
          zoomControl={false}
          dragging={false}
          scrollWheelZoom={false}
          doubleClickZoom={false}
          boxZoom={false}
          keyboard={false}
          touchZoom={false}
          attributionControl={false}
          className="login-bg-map-host"
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        </MapContainer>
        <div className="login-gradient-overlay" aria-hidden />
      </div>

      <header className="login-topbar">
        <div className="login-logo">
          <span className="login-flutter">flutter</span>
          <span className="login-logo-dot" />
          <span className="login-friends">friends</span>
        </div>
        <span className="login-badge">Butterfly Mapper</span>
      </header>

      <main className="login-card">
        <div className="login-butterfly-wrap">
          <DecorativeButterflySvg />
        </div>

        <h1 className="login-headline">
          Explore Butterfly Biodiversity
          <br />
          Across the <em>U.S.</em>
        </h1>

        <div className="login-divider-line" />

        <p className="login-body-text">
          FlutterFriends generates butterfly images inspired by the biodiversity of any location you choose. Pick a spot on
          the map and discover what might be flying there.
        </p>

        <div className="login-spacer" />

        <p className="login-body-text login-body-text-muted">
          Butterflies are indicators of ecosystem health, pollinator activity, and environmental change — explore how species
          vary from place to place.
        </p>

        <div className="login-spacer-lg" />

        <Link className="login-start-btn" to="/map">
          Start exploring
          <span className="login-arrow" aria-hidden>
            →
          </span>
        </Link>
      </main>

      <p className="login-footer-hint">Click anywhere on the map to begin · Powered by OpenStreetMap</p>
    </div>
  );
}
