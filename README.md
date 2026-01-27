# Security Camera Face Recognition

A python-based security camera system that uses your laptop's webcam to detect motion and recognize faces.

## Features
- **Motion Detection**: Only processes frames when movement is detected.
- **Face Recognition**: Identifies known people (Green Circle) vs intruders (Red Circle).
- **Performance**: Uses threading to keep the video feed smooth.

## Quick Start
1.  **Add Photos**: Place `.jpg` images of known people in the `known_faces` folder. Name the file the person's name (e.g. `Alice.jpg`).
2.  **Run**:
    ```powershell
    .\venv\Scripts\python main.py
    ```
