import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureSystem:
    def __init__(self, model_path='hand_landmarker.task'):
        # Hand Landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE # Using IMAGE mode for simplicity in loop
        )
        try:
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("MediaPipe Hand Landmarker loaded.")
        except Exception as e:
            print(f"Error loading Hand Landmarker: {e}")
            self.detector = None

    def detect_gesture(self, frame):
        if not self.detector:
            return None, None

        # MediaPipe needs RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        try:
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Get first hand
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # Convert to NormalizedLandmark list for drawing (if needed externally) 
                # or just process logic here.
                # Logic: same as before, check Y coordinates.
                
                # Landmarks:
                # 4 = Thumb tip, 2 = Thumb MCP
                # 8 = Index tip, 6 = Index PIP
                # 12 = Middle tip, 10 = Middle PIP
                # 16 = Ring tip, 14 = Ring PIP
                # 20 = Pinky tip, 18 = Pinky PIP
                
                lm = hand_landmarks # list of NormalizedLandmark
                
                index_ext = lm[8].y < lm[6].y
                middle_ext = lm[12].y < lm[10].y
                ring_ext = lm[16].y < lm[14].y
                pinky_ext = lm[20].y < lm[18].y
                
                extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
                
                gesture = None
                if extended_count == 0:
                    gesture = "Fist"
                elif extended_count == 4:
                    gesture = "Open"
                elif extended_count == 2 and index_ext and middle_ext and not ring_ext and not pinky_ext:
                    gesture = "Victory"
                elif extended_count == 1 and index_ext:
                    gesture = "One"
                    
                # Return landmarks compatible with drawing utils? 
                # mp_hands.HandLandmark is needed for drawing utils usually.
                # But Tasks API returns a list of NormalizedLandmark objects.
                # Drawing might be tricky with legacy drawing_utils if types mismatch.
                # We will return the list object, but main.py might need update to draw it manually or convert.
                
                return gesture, hand_landmarks
                
        except Exception as e:
            print(f"Gesture error: {e}")
            
        return None, None
