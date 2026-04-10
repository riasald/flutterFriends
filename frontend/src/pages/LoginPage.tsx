import { useNavigate } from "react-router-dom";
import { MapContainer, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "../styles/flutterfriends.css";
import titlePng from "../assets/title.png";

export default function LoginPage() {
  const nav = useNavigate();

  return (
    <div className="ff-page ff-landingPage">
      <div className="ff-landingMapBg">
        <MapContainer
          center={[39.5, -98.35]}
          zoom={4}
          scrollWheelZoom={false}
          dragging={false}
          doubleClickZoom={false}
          boxZoom={false}
          keyboard={false}
          zoomControl={false}
          attributionControl={false}
          className="ff-landingMap"
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        </MapContainer>
        <div className="ff-landingOverlay" />
      </div>

      <header className="ff-headerRow ff-headerSoft">
        <img className="ff-titleImg" src={titlePng} alt="FlutterFriends" />
      </header>

      <main className="ff-landingWrapper">
        <div className="ff-loginCardEnhanced ff-infoCard">
          <h1 className="ff-infoTitle">Explore Butterfly Biodiversity Across the U.S.</h1>

          <p className="ff-infoText">
            FlutterFriends is an interactive app that generates butterfly images based on location.
            Choose a place in the United States and discover a butterfly inspired by the biodiversity
            of that region.
          </p>

          <p className="ff-infoText">
            Butterflies are more than beautiful. They are important indicators of ecosystem health,
            pollinator activity, and environmental change. Exploring where different butterflies may
            appear helps connect people to the diversity of life around them.
          </p>

          <p className="ff-infoText">
            Click around the map, try different coordinates, and see how butterfly species can vary
            from place to place.
          </p>

          <button className="ff-orangeBtn ff-startBtn" onClick={() => nav("/map")}>
            Start Exploring →
          </button>
        </div>
      </main>
    </div>
  );
}