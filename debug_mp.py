import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
print(dir(vision))
try:
    print(vision.FaceEmbedder)
except AttributeError:
    print("FaceEmbedder not found directly.")
