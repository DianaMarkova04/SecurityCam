import mediapipe as mp
try:
    import mediapipe.python.solutions as solutions
    print("Imported mediapipe.python.solutions successfully")
    print(dir(solutions))
except ImportError as e:
    print(f"Failed to import mediapipe.python.solutions: {e}")

try:
    from mediapipe import solutions
    print("from mediapipe import solutions success")
except ImportError as e:
    print(f"from mediapipe import solutions failed: {e}")
