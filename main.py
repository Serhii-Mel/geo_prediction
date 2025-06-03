from plonk import PlonkPipeline
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json

# Example JSON input (simulating post data)
post_data = {
    "url": "https://example.com/media/example.jpg", # or video and then extract some photos.
    "description": "Somewhere in Ukraine",
    "tags": ["city", "street", "old city"]
}

def download_image(url):
    """Download an image from a URL and return a PIL Image object."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def analyze_image(image, pipeline):
    """Analyze an image using PLONK pipeline and return coordinates with likelihoods."""
    # Run PLONK pipeline to get coordinates
    coordinates = pipeline(
        images=image,
        batch_size=32,
        cfg=2.0
    )
    
    # Compute likelihoods for the coordinates
    likelihood = pipeline.compute_likelihood(
        images=image,
        coordinates=coordinates,
        cfg=0,
        rademacher=False
    )
    
    # Pair coordinates with their likelihoods
    results = [
        {"latitude": float(coord[0]), "longitude": float(coord[1]), "likelihood": float(likelihood[i])}
        for i, coord in enumerate(coordinates)
    ]
    
    # Sort by likelihood (descending)
    results = sorted(results, key=lambda x: x["likelihood"], reverse=True)
    
    return results

def process_post_data(post_json):
    """Process post data and return a dictionary with locations and metadata."""
    # Initialize PLONK pipeline
    pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC")
    
    # Download image
    # image = download_image(post_json["url"])

    # Just for test case take a local image
    image = Image.open("media/img/example.jpg")

    if image is None:
        return {"error": "Failed to download or process image"}
    
    # Analyze image for geolocation
    location_data = analyze_image(image, pipe)
    
    # Construct output dictionary
    output = {
        "locations": location_data,
        # "description": post_json.get("description", ""),
        # "tags": post_json.get("tags", [])
    }
    
    return output

# Process the example post data
result = process_post_data(post_data)

# Print the result as JSON
print(json.dumps(result, indent=2, ensure_ascii=False))
