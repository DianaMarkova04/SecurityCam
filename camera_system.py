import cv2
import face_recognition
import os
import threading
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gesture_recognition import GestureSystem

class CameraSystem:
    def __init__(self, known_faces_dir='known_faces', face_model_path='face_landmarker.task'):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.is_running = True
        self.lock = threading.Lock()
        
        # Thread 1: Face Recognition
        self.frame_to_process_face = None
        self.face_thread = threading.Thread(target=self._face_processing_loop)
        self.face_thread.daemon = True
        
        # Thread 2: MediaPipe (Gestures + Liveness)
        self.frame_to_process_mp = None
        self.mp_thread = threading.Thread(target=self._mediapipe_processing_loop)
        self.mp_thread.daemon = True
        
        # Shared State
        self.latest_face_status = None 
        self.current_face_locations = [] 
        self.motion_detected = False
        
        self.current_ear = 0.0
        self.gesture_message = ""
        self.liveness_verified = False
        
        self.gesture_landmarks = [] 
        self.liveness_landmarks = [] 
        
        # Internal Logic State
        self.gesture_system = GestureSystem()
        self.gesture_sequence = []
        self.target_sequence = ['Open', 'Fist', 'Open']
        self.last_gesture_time = 0
        self.gestures_completed = False 
        
        self.blink_detected = False
        self.ear_threshold = 0.35
        
        self.previous_frame = None

        self.load_known_faces()
        
        base_options = python.BaseOptions(model_asset_path=face_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            num_faces=1)
        try:
            self.liveness_detector = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe Face Landmarker loaded.")
        except Exception as e:
            print(f"Error loading Face Landmarker: {e}")
            self.liveness_detector = None
            
        self.face_thread.start()
        self.mp_thread.start()

    def load_known_faces(self):
        print("Loading known faces...")
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created {self.known_faces_dir} directory.")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    path = os.path.join(self.known_faces_dir, filename)
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        print(f"Loaded face: {name}")
                    else:
                        print(f"No face found in {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        print(f"Loaded {len(self.known_face_names)} faces.")

    def calculate_mp_ear(self, landmarks, indices):
        def dist(i1, i2):
            p1 = np.array([landmarks[i1].x, landmarks[i1].y])
            p2 = np.array([landmarks[i2].x, landmarks[i2].y])
            return np.linalg.norm(p1 - p2)

        p1, p2, p3, p4, p5, p6 = indices
        v1 = dist(p2, p6)
        v2 = dist(p3, p5)
        h = dist(p1, p4)
        if h == 0: return 0
        return (v1 + v2) / (2.0 * h)

    def register_face(self, frame, name):
        if not self.liveness_verified:
             return False, "Check Failed: Please Blink first!"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return False, "No face detected in frame."
        
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not encodings:
            return False, "Could not encode face."
            
        new_encoding = encodings[0]
        filename = f"{name}.jpg"
        filepath = os.path.join(self.known_faces_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame) 
            print(f"Saved {filepath}")
        except Exception as e:
            return False, f"Error saving file: {e}"
            
        with self.lock:
            self.known_face_encodings.append(new_encoding)
            self.known_face_names.append(name)
            self.latest_face_status = 'access_granted' 
            
        return True, f"Successfully registered {name}!"

    def check_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.previous_frame is None:
            self.previous_frame = gray
            return False
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500: 
                continue
            self.motion_detected = True
            break
        self.previous_frame = gray
        return self.motion_detected

    def update_frame(self, frame):
        with self.lock:
            self.frame_to_process_face = frame.copy()
            self.frame_to_process_mp = frame.copy()

    # -----------------------------
    # THREAD 1: Face Recognition (THROTTLED)
    # -----------------------------
    def _face_processing_loop(self):
        while self.is_running:
            frame = None
            with self.lock:
                if self.frame_to_process_face is not None:
                    frame = self.frame_to_process_face
                    self.frame_to_process_face = None 
            
            if frame is not None:
                self._process_faces(frame)
                # THROTTLE: Sleep 0.3s (approx 3 FPS). This drastically reduces CPU.
                time.sleep(0.3) 
            else:
                time.sleep(0.1)

    def _process_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            scaled_locations.append((top*4, right*4, bottom*4, left*4))
        self.current_face_locations = scaled_locations

        if not face_locations:
             # Auto-Logout
             self.latest_face_status = None
             self.liveness_verified = False 
             self.gestures_completed = False
             return

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        status = 'unknown'
        for face_encoding in face_encodings:
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.55:
                    status = 'identified'
                    break 
        
        # State Machine Logic
        if self.latest_face_status == 'access_granted':
            pass 
        elif status == 'identified':
            if self.latest_face_status not in ['identified', 'verifying_liveness']:
                self.liveness_verified = False 
            
            if self.liveness_verified:
                self.latest_face_status = 'access_granted'
            else:
                self.latest_face_status = 'verifying_liveness'
        else: # Unknown
            if self.gestures_completed:
                if self.liveness_verified:
                    self.latest_face_status = 'access_granted'
                else:
                    self.latest_face_status = 'verifying_liveness' 
            else:
                self.latest_face_status = 'unknown'

    # -----------------------------
    # THREAD 2: MediaPipe (CONDITIONAL)
    # -----------------------------
    def _mediapipe_processing_loop(self):
        while self.is_running:
            frame = None
            status = self.latest_face_status # Read once
            
            # CONDITION: If Access Granted, sleep heavily.
            if status == 'access_granted':
                self.current_ear = 0.0
                time.sleep(0.5)
                continue

            with self.lock:
                if self.frame_to_process_mp is not None:
                    frame = self.frame_to_process_mp
                    self.frame_to_process_mp = None
            
            if frame is not None:
                self._process_mediapipe(frame, status)
            else:
                time.sleep(0.01)

    def _process_mediapipe(self, frame, status):
        landmark_list_for_drawing = []

        # 1. Gesture Detection (ONLY IF UNKNOWN)
        if status == 'unknown':
            # This is the heavy part for gestures
            gesture, landmarks = self.gesture_system.detect_gesture(frame)
            
            if hasattr(landmarks, 'landmark'):
                 landmark_list_for_drawing = landmarks.landmark 
            elif isinstance(landmarks, list):
                 landmark_list_for_drawing = landmarks
            
            if gesture:
                now = time.time()
                if now - self.last_gesture_time > 0.5:
                    expected_next = self.target_sequence[len(self.gesture_sequence)]
                    if gesture == expected_next:
                        if len(self.gesture_sequence) == 0 or (now - self.last_gesture_time) > 1.0: 
                             self.gesture_sequence.append(gesture)
                             self.last_gesture_time = now
                             self.gesture_message = f"Gesture {len(self.gesture_sequence)}/3: {gesture}"
                             print(self.gesture_message)
                
                if self.gesture_sequence == self.target_sequence:
                    self.gestures_completed = True
                    self.liveness_verified = False # FORCE BLINK
                    self.gesture_sequence = []
                    self.gesture_message = "Gestures OK. BLINK TO VERIFY."
        else:
             self.gesture_sequence = []
             # Don't run gesture inference at all
             
        self.gesture_landmarks = landmark_list_for_drawing

        # 2. Liveness Detection
        # Run if 'verifying_liveness' OR 'identified' (waiting for blink) OR 'unknown' (registration)
        if self.liveness_detector:
             # Just always run liveness if not granted, it's relatively cheap compared to hands,
             # but we can skip if status is None (no face)
             if status is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                detection = self.liveness_detector.detect(mp_image)
                
                if detection.face_landmarks:
                    landmarks = detection.face_landmarks[0]
                    left_indices = [33, 160, 158, 133, 153, 144]
                    right_indices = [362, 385, 387, 263, 373, 380]
                    
                    ear_left = self.calculate_mp_ear(landmarks, left_indices)
                    ear_right = self.calculate_mp_ear(landmarks, right_indices)
                    avg_ear = (ear_left + ear_right) / 2.0
                    
                    self.current_ear = avg_ear
                    
                    if avg_ear < self.ear_threshold:
                        if not self.liveness_verified:
                             print(f"Blink Detected! EAR: {avg_ear:.2f}")
                             self.liveness_verified = True
                else:
                    self.current_ear = 0.0

    def get_status(self):
        return (self.motion_detected, 
                self.latest_face_status, 
                self.current_face_locations, 
                self.current_ear,
                self.gesture_landmarks,
                self.gesture_message)

    def stop(self):
        self.is_running = False
        if self.face_thread:
            self.face_thread.join(timeout=1.0)
        if self.mp_thread:
            self.mp_thread.join(timeout=1.0)
