import requests
import os

def download_model():
    detector_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float32/1/blaze_face_short_range.tflite"
    detector_path = "blaze_face_short_range.tflite"
    
    embedder_url = "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float32/1/face_embedder.tflite"
    embedder_path = "face_embedder.tflite"
    
    for url, path in [(detector_url, detector_path), (embedder_url, embedder_path)]:
        if os.path.exists(path):
            print(f"{path} already exists.")
            continue
            
        print(f"Downloading {path} from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download {path}: {e}")

if __name__ == "__main__":
    download_model()
