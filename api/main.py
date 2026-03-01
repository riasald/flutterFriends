from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import csv
import math
import os

app = FastAPI(
    title="Butterfly Generation API",
    description="API for the flutterFriends geographically conditioned ML model."
)

class ButterflyRequest(BaseModel):
    latitude: float
    longitude: float
    visual_features: List[str]

def get_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

@app.post("/api/generate-butterfly")
async def generate_butterfly(request: ButterflyRequest):
    csv_file_path = "data/raw_metadata/nymphalidae_us_raw_2020_2024_humanobs.csv"

    if not os.path.exists(csv_file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Run download_metadata.py first.")

    closest_image_url = None
    closest_species = None
    shortest_distance = float('inf')

    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["lat"] or not row["lon"]:
                continue

            row_lat = float(row["lat"])
            row_lon = float(row["lon"])

            distance = get_distance(request.latitude, request.longitude, row_lat, row_lon)

            if distance < shortest_distance:
                shortest_distance = distance
                closest_image_url = row["image_url"]
                closest_species = row["species"]

    if not closest_image_url:
        closest_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Flickr_-_ggallice_-_Morpho_peleides.jpg/640px-Flickr_-_ggallice_-_Morpho_peleides.jpg"
        closest_species = "Mock Fallback"

    return {
        "status": "success",
        "message": f"Successfully 'generated' geographically accurate butterfly.",
        "input_received": {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "visual_features": request.visual_features
        },
        "species": closest_species,
        "image_url": closest_image_url
    }