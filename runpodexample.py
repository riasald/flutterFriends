import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}

data = {
  "input": {
    "lat": 29.6516,
    "lon": -82.3248,
    "num_candidates": 1,
    "num_outputs": 1,
    "samples_per_batch": 1
  }
}


response = requests.post('https://api.runpod.ai/v2/m8q4fur9v0n4cu/run', headers=headers, json=data)