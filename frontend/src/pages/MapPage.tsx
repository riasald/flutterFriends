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
};

function LocationMarker({ position, setPosition }: LocationMarkerProps) {
  useMapEvents({
    click(e: LeafletMouseEvent) {
      setPosition({
        lat: e.latlng.lat,
        lng: e.latlng.lng,
      });
    },
  });

  if (!position) return null;

  return <Marker position={[position.lat, position.lng]} />;
}

export default function MapPage() {
  const defaultCenter: Position = {
    lat: 29.652,
    lng: -82.325,
  };

  const [position, setPosition] = useState<Position | null>(defaultCenter);

  const handleCustomizeButterfly = async () => {
  if (!position) return;

  const payload = {
    latitude: position.lat,
    longitude: position.lng,
  };

    console.log("Sending payload:", payload);

    try {
        await fetch("https://httpbin.org/post", { // temporary api endpoint that works
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        });
    } catch (error) {
        console.error("Error sending request:", error);
    }
    };

  const [showButterflyModal, setShowButterflyModal] = useState(false);

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
            <button className="ff-ctaSoft" onClick={handleCustomizeButterfly}>
              Customize Your Butterfly
            </button>

            <div className="ff-location">
              📍 Location:
              <br />
              {position
                ? `${position.lat.toFixed(4)}°, ${position.lng.toFixed(4)}°`
                : "Click on the map to select a location"}
            </div>

            <div className="ff-butterflyWrapClear">
              <img
                src={butterflyPng}
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
          </section>

          <section className="ff-bubble ff-mapBubble">
            <MapContainer
              center={[defaultCenter.lat, defaultCenter.lng]}
              zoom={10}
              scrollWheelZoom={true}
              className="ff-mapFrameBig"
            >
              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <LocationMarker position={position} setPosition={setPosition} />
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
                src={butterflyPng}
                alt="Enlarged Butterfly"
                className="ff-butterflyModalImg"
            />
            </div>
        </div>
        )}
    </div>
  );
}