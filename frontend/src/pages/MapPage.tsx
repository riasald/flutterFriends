import "../styles/flutterfriends.css";
import butterflyPng from "../assets/butterfly.png";
import titlePng from "../assets/title.png";

export default function MapPage() {
  //const locationName = "Gainesville, FL";
  const lat = 29.652;
  const lng = -82.325;

  const embedUrl = `https://www.google.com/maps?q=${lat},${lng}&z=10&output=embed`;

  return (
    <div className="ff-page ff-babyYellow">
    <header className="ff-headerSoft ff-headerRow">
        <img
            className="ff-titleImg"
            src={titlePng}
            alt="FlutterFriends"
        />

        <button
            className="ff-pill ff-pillBig"
            onClick={() => (window.location.href = "/login")}
        >
            Log Out
        </button>
    </header>

    <main className="ff-mainSoft">
        <div className="ff-twoColumnLayout">

            {/* LEFT BUBBLE */}
            <section className="ff-bubble ff-butterflyCard">
            <button className="ff-ctaSoft">
                Customize Your Butterfly
            </button>

            <div className="ff-location">
                📍 Location: Gainesville, FL <br />
                29.6520° N, 82.3250° W
            </div>

            <div className="ff-butterflyWrapClear">
                <img
                src={butterflyPng}
                alt="Butterfly"
                className="ff-butterflyBig"
                />
            </div>

            <button className="ff-smallBtnSoft">⤢</button>
            </section>

            {/* RIGHT BUBBLE */}
            <section className="ff-bubble ff-mapBubble">
            <iframe
                className="ff-mapFrameBig"
                src={embedUrl}
                title="Map"
            />
            </section>

        </div>
    </main>
    </div>
  );
}