# flutterFriends
# 🦋 FlutterFriends 🦋

FlutterFriends is a full-stack web app that generates AI-powered butterfly images based on geographic coordinates within the United States.

Users can:

* Click on a map or enter coordinates manually
* Generate a butterfly using a trained model (via RunPod)
* View and expand the generated image
* See predicted species information

---

## Tech Stack

**Frontend**

* React + TypeScript (Vite)
* React Leaflet (map)

**Backend**

* Python + FastAPI
* RunPod Serverless API

---

## Project Structure

```
flutterFriends/
├── frontend/      # React app
├── backend/       # FastAPI server
```

---

## Prerequisites

Make sure you have installed:

* Node.js (v18+ recommended)
* npm
* Python (3.10+ recommended)
* pip

---

## Environment Variables

### Backend (`backend/.env`)

Create a `.env` file inside the `backend` folder:

```
RUNPOD_API_KEY=rp_your_real_api_key_here
RUNPOD_ENDPOINT_ID=m8q4fur9v0n4cu
PORT=8000
```

Never commit this file to GitHub.

---

### Frontend (`frontend/.env`)

Create a `.env` file inside the `frontend` folder:

```
VITE_API_BASE_URL=http://localhost:8000
```

---

## TO RUN APP:

### 1. Start the Backend

Open a terminal:

```
cd backend
pip install -r requirements.txt
python app.py
```

You should see something like:

```
Uvicorn running on http://0.0.0.0:8000
```

Test it:

```
http://localhost:8000/health
```

---

### 2. Start the Frontend

Open a **new terminal**:

```
cd frontend
npm install
npm run dev
```

You should see:

```
Local: http://localhost:5173/
```

Open that URL in your browser.

---

## How to Use

1. Click anywhere on the map **within the United States 🇺🇸**

   * OR manually enter latitude & longitude

2. Click **"Set Coordinates"** (if typing manually)

3. Click **"Customize Your Butterfly"**

4. Wait for the model to generate your butterfly

5. Click ⤢ to expand the image

---

## 🇺🇸 Coordinate Restrictions

The app only supports US coordinates:

* Latitude: `24.5 → 49.5`
* Longitude: `-125 → -66.5`

Enforced at:

* Frontend (UX)
* Backend (security)

---

## Troubleshooting

### ❌ `401 Unauthorized`

* Your RunPod API key is wrong or missing
* Make sure it starts with `rp_`

---

### ❌ `'vite' is not recognized`

Run:

```
npm install
```

---

### ❌ No image generated

Check:

* Backend is running
* API key is valid
* Endpoint ID is correct

---

### ❌ `.env` not working

Make sure:

* File is named exactly `.env`
* No quotes or spaces:

  ```
  RUNPOD_API_KEY=rp_...
  ```

---

## Security Notes

* API key is stored only in backend `.env`
* Frontend never exposes secrets
* Backend validates all coordinates

---

## Future Improvements

* download button for images
* UI revamp
* animation while loading

---

## Author

Built as a full-stack AI + mapping project using RunPod + React + FastAPI.

---
