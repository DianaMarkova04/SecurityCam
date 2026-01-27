import cv2
import face_recognition
import os
import threading
import numpy as np
import time
from gesture_recognition import GestureSystem

class CameraSystem:
    def __init__(self, known_faces_dir='known_faces'):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_to_process = None
        self.face_processing_thread = None
        self.is_running = True
        self.latest_face_status = None # None, 'identified', 'unknown', 'access_granted'
        self.lock = threading.Lock()
        
        # Motion detection
        self.previous_frame = None
        self.motion_detected = False
        
        # Gesture System
        self.gesture_system = GestureSystem()
        self.current_gesture = None
        self.gesture_sequence = []
        self.target_sequence = ['Open', 'Fist', 'Open']
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0 # Seconds between gestures
        self.gesture_message = ""
        
        self.load_known_faces()
        
        # Start face processing thread
        self.face_processing_thread = threading.Thread(target=self._face_processing_loop)
        self.face_processing_thread.daemon = True
        self.face_processing_thread.start()

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

    def check_motion(self, frame):
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        # Compute difference
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500: # Minimum area
                continue
            self.motion_detected = True
            break
            
        self.previous_frame = gray
        return self.motion_detected

    def update_frame(self, frame):
        """Update the frame available for the processing thread"""
        with self.lock:
            self.frame_to_process = frame.copy()

    def detect_gesture_pass(self, frame):
        """Run gesture detection in main loop for responsiveness"""
        # Only run if face is unknown or we are in gesture entry mode
        if self.latest_face_status == 'identified' or self.latest_face_status == 'access_granted':
            self.gesture_sequence = []
            self.gesture_message = ""
            return None, None

        gesture, landmarks = self.gesture_system.detect_gesture(frame)
        self.current_gesture = gesture
        
        if gesture:
            now = time.time()
            # Simple debounce / state machine
            if now - self.last_gesture_time > 0.5:
                # Add to sequence if it matches next target step logic or just append?
                # Let's enforce the order.
                
                expected_next = self.target_sequence[len(self.gesture_sequence)]
                
                if gesture == expected_next:
                    if len(self.gesture_sequence) == 0 or (now - self.last_gesture_time) > 1.0: # Ensure distinct separate moves
                         self.gesture_sequence.append(gesture)
                         self.last_gesture_time = now
                         self.gesture_message = f"Gesture {len(self.gesture_sequence)}/3 Accepted: {gesture}"
                         print(self.gesture_message)
                
                # Reset if wrong gesture? Or just ignore?
                # Ignoring makes it easier. Resetting makes it secure.
                # Let's reset if it's explicitly wrong but not 'None'
                elif gesture in ['Fist', 'Open', 'Victory'] and gesture != expected_next and gesture != self.gesture_sequence[-1] if self.gesture_sequence else True:
                     # self.gesture_sequence = []
                     # self.gesture_message = "Wrong Gesture. Reset."
                     pass

            if self.gesture_sequence == self.target_sequence:
                self.latest_face_status = 'access_granted'
                self.gesture_message = "ACCESS GRANTED"
                self.gesture_sequence = []
                
        return gesture, landmarks

    def _face_processing_loop(self):
        while self.is_running:
            frame = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None 
            
            if frame is not None:
                self._process_faces(frame)
            else:
                cv2.waitKey(10) 

    def _process_faces(self, frame):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if not face_locations:
            # Keep previous status if access granted? No, security cam should reset if no one is there?
            # User wants: "if the person is in the database a green circle apears if not red circle"
            # If no face, maybe reset status unless access_granted?
            if self.latest_face_status != 'access_granted':
                self.latest_face_status = None
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
        
        # Don't overwrite access_granted with unknown
        if self.latest_face_status != 'access_granted' and self.latest_face_status != 'identified':
            self.latest_face_status = status
        elif status == 'identified':
            self.latest_face_status = 'identified'

    def get_status(self):
        return self.motion_detected, self.latest_face_status

    def stop(self):
        self.is_running = False
        if self.face_processing_thread:
            self.face_processing_thread.join(timeout=1.0)
