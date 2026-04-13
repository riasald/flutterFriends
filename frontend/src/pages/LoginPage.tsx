import { Link } from "react-router-dom";
import { MapContainer, TileLayer } from "react-leaflet";
import "../styles/loginPageRedesign.css";
import logoPng from "../assets/FlutterFriendsLogo.png";

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
          <img src={logoPng} alt="FlutterFriends Logo" className="login-logo-img" />
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
