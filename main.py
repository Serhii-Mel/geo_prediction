from plonk import PlonkPipeline
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
import cv2
from tiktok_scraper.tiktok_video_scraper_mobile import TikTokVideoScraperMobile

# Example JSON input (simulating post data)
post_data = {
    "url": "https://example.com/media/example.jpg",  # or video and then extract some photos
    "description": "Somewhere in Ukraine",
    "tags": ["city", "street", "old city"]
}

def parse_tiktok(url: str):
    """Parse TikTok video and download it."""
    tiktok_video = TikTokVideoScraperMobile()
    video_id = tiktok_video.get_video_id_by_url(url)
    tiktok_video_urls, video_thumbnail, geo_info = tiktok_video.get_video_data_by_video_id(video_id)
    downloaded_video_list = tiktok_video.download(tiktok_video_urls, video_id)
    tiktok_video.tiktok_session.close()
    return tiktok_video_urls, downloaded_video_list, geo_info

def download_image(url):
    """Download an image from a URL and return a PIL Image object."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def extract_frames_from_video(video_path: str):
    """Extract frames from a video at 1-second intervals and return a list of PIL Images."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        # Get frames per second (FPS) and total frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Frames to skip to get one per second
        frames = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at 1-second intervals
            if frame_count % frame_interval == 0:
                # Convert OpenCV BGR frame to RGB and then to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb).convert("RGB")
                frames.append(pil_image)

            frame_count += 1

        cap.release()
        return frames
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

def analyze_image(image, pipeline):
    """Analyze an image or list of images using PLONK pipeline and return coordinates with likelihoods."""
    # Check if input is a single image or a list of images
    is_list = isinstance(image, list)
    images = image if is_list else [image]
    batch_size = len(images) if is_list else 32  # Dynamic batch size for lists

    # Run PLONK pipeline to get coordinates
    coordinates = pipeline(
        images=images,
        batch_size=batch_size,
        cfg=2.0 # from 0 to 5.
    )

    # Compute likelihoods for the coordinates
    likelihood = pipeline.compute_likelihood(
        images=images,
        coordinates=coordinates,
        cfg=0,
        rademacher=False
    )

    # Pair coordinates with their likelihoods
    if len(coordinates) != len(likelihood):
        print(f"Error: Mismatch between coordinates ({len(coordinates)}) and likelihoods ({len(likelihood)})")
        return []

    # Pair coordinates with their likelihoods and percentages
    results = []
    for i, coord in enumerate(coordinates):
        try:
            if not isinstance(coord, (list, tuple, np.ndarray)) or len(coord) != 2:
                print(f"Warning: Invalid coordinate at index {i}: {coord}")
                continue

            result = {
                "latitude": float(coord[0]),
                "longitude": float(coord[1]),
                "likelihood": float(likelihood[i]),
                "perc": convert_likelihood_to_percentage_bounded(float(likelihood[i]))
            }
            results.append(result)
        except (IndexError, TypeError, ValueError) as e:
            print(f"Error processing index {i}: {e}")
            continue

    # Sort by likelihood (descending)
    results = sorted(results, key=lambda x: x["likelihood"], reverse=True)

    return results

def process_post_data(img: str | list[str]):
    """Process post data (image path or list of image paths) and return a dictionary with locations and metadata."""
    # Initialize PLONK pipeline
    pipe = PlonkPipeline("nicolas-dufour/PLONK_YFCC")

    # Handle single image or list of images
    if isinstance(img, str):
        # Single image (local file or URL)
        if img.startswith("http"):
            image = download_image(img)
        else:
            try:
                image = Image.open(img).convert("RGB")
            except Exception as e:
                print(f"Error opening image {img}: {e}")
                return {"error": "Failed to open or process image"}
    else:
        # List of images (e.g., from video frames)
        image = img

    if image is None:
        return {"error": "Failed to download or process image"}

    # Analyze image(s) for geolocation
    location_data = analyze_image(image, pipe)

    # Construct output dictionary
    output = {
        "locations": location_data,
        # "description": post_data.get("description", ""),
        # "tags": post_data.get("tags", [])
    }

    return output


def convert_likelihood_to_percentage_bounded(likelihood, min_likelihood=-5, max_likelihood=9):
    """Convert a single likelihood value to percentage with bounds: -5 or less = 0%, 9 or more = 100%."""
    try:
        bounded_likelihood = max(min(float(likelihood), max_likelihood), min_likelihood)
        percentage = ((bounded_likelihood - min_likelihood) / (max_likelihood - min_likelihood)) * 100
        return round(percentage, 2)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    # Parse TikTok video and download
    tiktok_url = "https://www.tiktok.com/@travelwithbleo/video/7477688351949098262" # just for example
    tiktok_video_urls, downloaded_video_list, geo_info = parse_tiktok(tiktok_url)
    print("TikTok Video URLs:", tiktok_video_urls)
    print("Downloaded Video List:", downloaded_video_list)
    # Assume the first downloaded video is used
    if downloaded_video_list:
        video_path = downloaded_video_list[0]  # Path to the downloaded video
        # Extract frames from the video
        frames = extract_frames_from_video(video_path)
        if not frames:
            print("No frames extracted from video.")
        else:
            # Process the extracted frames
            result = process_post_data(frames)
            print("Geo Info:", geo_info)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("No video downloaded.")
