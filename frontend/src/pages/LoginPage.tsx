import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/flutterfriends.css";
import titlePng from "../assets/title.png";

export default function LoginPage() {
  const nav = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    nav("/map");
  }

  return (
    <div className="ff-page">
      
      {/* HEADER */}
      <header className="ff-headerRow ff-headerSoft">
        <img className="ff-titleImg" src={titlePng} alt="FlutterFriends" />
      </header>

      {/* LOGIN CARD */}
      <main className="ff-loginWrapper">
        <div className="ff-loginCardEnhanced">
          <form className="ff-form" onSubmit={onSubmit}>
            <div className="ff-inputRow">
              <span className="ff-icon">👤</span>
              <input
                className="ff-input"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>

            <div className="ff-inputRow">
              <span>🔒</span>
              <input
                className="ff-input"
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <button className="ff-orangeBtn" type="submit">
              Sign In
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}