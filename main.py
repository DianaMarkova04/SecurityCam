import cv2
import sys
import mediapipe as mp
from camera_system import CameraSystem

def main():
    print("Initializing Security Camera...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Camera warming up...")
        import time
        time.sleep(2.0)
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return

    cam_system = CameraSystem()
    
    print("Camera System Started.")
    print("Press 'q' to quit.")
    print("Press 'n' to register (MUST BLINK FIRST).")
    print("Press 'm' to register BANNED PERSON (MUST BLINK FIRST).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # --- MAIN PROCESSING (ASYNC) ---
            cam_system.check_motion(frame)
            cam_system.update_frame(frame)
            
            # --- GET STATUS ---
            (motion_detected, 
             face_status, 
             face_locations, 
             current_ear, 
             gesture_landmarks, 
             gesture_msg) = cam_system.get_status()

            # --- DRAW UI ---
            height, width = frame.shape[:2]

            # Draw Gestures
            if gesture_landmarks:
                h, w, _ = frame.shape
                for point in gesture_landmarks:
                    if hasattr(point, 'x'):
                        cx, cy = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

            # Box Color Logic
            box_color = (0, 0, 255) # Red (Unknown)
            status_text = "UNKNOWN"
            
            if face_status == 'access_granted':
                box_color = (0, 255, 0) # Green
                status_text = "ACCESS GRANTED"
            elif face_status == 'banned':
                box_color = (0, 0, 255) # Red
                status_text = "BANNED: ACCESS DENIED"
            elif face_status == 'verifying_liveness':
                box_color = (255, 255, 0) # Cyan/Yellowish
                status_text = "BLINK TO VERIFY"
            elif face_status == 'identified':
                 box_color = (0, 255, 255) # Yellow
                 status_text = "IDENTIFIED"

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Big X for banned
                if face_status == 'banned':
                     cv2.line(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                     cv2.line(frame, (right, top), (left, bottom), (0, 0, 255), 3)
            
            # Global Text
            if face_status == 'access_granted':
                 cv2.putText(frame, "ACCESS GRANTED", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif face_status == 'banned':
                 cv2.putText(frame, "SECURITY ALERT: BANNED PERSON", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif face_status == 'verifying_liveness':
                 cv2.putText(frame, "PLEASE BLINK EYES", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif face_status == 'unknown':
                 cv2.putText(frame, "UNKNOWN - GESTURE REQUIRED", (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                 cv2.putText(frame, gesture_msg, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if motion_detected:
                cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Show Resolution
            cv2.putText(frame, f"Res: {width}x{height}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Liveness Feedback
            if cam_system.liveness_verified:
                 cv2.putText(frame, "LIVENESS: OK", (width - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                 cv2.putText(frame, "LIVENESS: WAIT", (width - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Debug EAR value
            ear_text = f"Eye Openness: {current_ear:.2f}"
            ear_color = (0, 255, 0) if current_ear > 0.32 else (0, 0, 255)
            
            if face_status is not None and face_status != 'banned':
                 cv2.putText(frame, ear_text, (width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)

            cv2.imshow('Security Camera', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Registration Mode (Allowed)
                if not cam_system.liveness_verified:
                    print("\n[WARNING] Cannot Register: Liveness Check Failed.")
                    print("Please BLINK at the camera until 'LIVENESS: OK' appears, then press 'n'.\n")
                else:
                    print("\n--- NEW USER REGISTRATION ---")
                    print("Pausing camera...")
                    name = input("Enter name for the new person: ")
                    if name:
                        success, msg = cam_system.register_face(frame, name)
                        print(msg)
                    else:
                        print("Registration cancelled.")
                    print("Resuming camera...\n")
            
            elif key == ord('m'):
                # Registration Mode (BANNED)
                if not cam_system.liveness_verified:
                    print("\n[WARNING] Cannot Register: Liveness Check Failed.")
                    print("Please BLINK at the camera until 'LIVENESS: OK' appears, then press 'm'.\n")
                else:
                    print("\n--- BANNED PERSON REGISTRATION ---")
                    print("Pausing camera...")
                    name = input("Enter name for the BANNED person: ")
                    if name:
                        success, msg = cam_system.register_banned_face(frame, name)
                        print(msg)
                    else:
                        print("Registration cancelled.")
                    print("Resuming camera...\n")
                
    except KeyboardInterrupt:
        pass
    finally:
        cam_system.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

if __name__ == "__main__":
    main()
