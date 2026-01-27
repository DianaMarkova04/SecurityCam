import cv2
import sys
import mediapipe as mp
from camera_system import CameraSystem

def main():
    print("Initializing Security Camera...")
    
    # Initialize Camera Check
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return

    # Initialize System
    cam_system = CameraSystem()
    
    print("Camera System Started.")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 1. Motion Detection
            is_motion = cam_system.check_motion(frame)

            # 2. Face Recognition (Update thread)
            if is_motion:
                cam_system.update_frame(frame)
            
            # 3. Gesture Detection (Main Thread for UI speed)
            landmarks, raw_landmarks_list = cam_system.detect_gesture_pass(frame)
            if raw_landmarks_list:
                # Custom drawing for Tasks API landmarks
                h, w, _ = frame.shape
                for point in raw_landmarks_list:
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

            # Get latest status
            _, face_status = cam_system.get_status()

            # 4. Visual Feedback
            height, width = frame.shape[:2]
            center = (width - 30, height - 30)
            radius = 15
            
            if face_status == 'identified' or face_status == 'access_granted':
                color = (0, 255, 0) # Green
                cv2.circle(frame, center, radius, color, -1)
                
                if face_status == 'access_granted':
                     cv2.putText(frame, "ACCESS GRANTED (GESTURE)", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                     cv2.putText(frame, "ACCESS GRANTED (FACE)", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif face_status == 'unknown':
                color = (0, 0, 255) # Red
                cv2.circle(frame, center, radius, color, -1)
                cv2.putText(frame, "UNKNOWN - GESTURE REQUIRED", (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Show Gesture Prompt/Status
                cv2.putText(frame, cam_system.gesture_message, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if is_motion:
                cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow('Security Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam_system.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

if __name__ == "__main__":
    main()
