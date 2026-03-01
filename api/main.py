from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Butterfly Generation API",
    description="API for the flutterFriends geographically conditioned ML model."
)

# 2. Define the Request Data Model
# This forces the frontend to send the correct data types.
class ButterflyRequest(BaseModel):
    latitude: float
    longitude: float
    visual_features: List[str] # e.g., ["blue wings", "large"]

# 3. Create the Generation Endpoint
@app.post("/api/generate-butterfly")
async def generate_butterfly(request: ButterflyRequest):
    """
    Receives geographic and visual data, and returns a generated butterfly image.
    Currently mocked to unblock frontend development.
    """

    # TODO: In the future, this is where you will pass the request data
    # to your Butterfly Generation Engine (the Purple Box).
    print(f"Received request for lat: {request.latitude}, lon: {request.longitude}")
    print(f"Requested features: {request.visual_features}")

    # Mock Response: Return a hardcoded image URL so the frontend has something to display
    dummy_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Flickr_-_ggallice_-_Morpho_peleides.jpg/640px-Flickr_-_ggallice_-_Morpho_peleides.jpg"

    return {
        "status": "success",
        "message": "This is a mock response. ML engine not yet connected.",
        "input_received": {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "visual_features": request.visual_features
        },
        "image_url": dummy_image_url
    }