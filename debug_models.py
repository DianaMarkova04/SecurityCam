import sys
print(sys.executable)
try:
    import face_recognition_models
    print("face_recognition_models imported successfully")
    print(face_recognition_models.__file__)
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

try:
    import face_recognition
    print("face_recognition imported successfully")
except Exception as e:
    print(f"face_recognition error: {e}")
