from mediapipe.tasks.python import vision
try:
    print(vision.ImageEmbedder)
    print("ImageEmbedder found.")
except AttributeError:
    print("ImageEmbedder NOT found.")
