import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess

class Camera:
    def __init__(self, manual_exposure=False, device_index=0):
        self.device_index = device_index
        self.stream = cv2.VideoCapture(device_index)
        
        # 1. Force MJPG to ensure high bandwidth for 1080p
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        # 2. Set High Resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not self.stream.isOpened():
            raise Exception("Unable to connect to camera")

        # 3. Warm-up: Read a few frames to let auto-exposure find the room level
        print("Warming up camera for 2 seconds...")
        for _ in range(20):
            self.stream.read()
            time.sleep(0.1)

        # 4. Apply Exposure Settings
        if manual_exposure:
            self._lock_camera_controls()
        
        # Get dimensions
        self.width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera initialized at {int(self.width)}x{int(self.height)}")

    def _lock_camera_controls(self):
        """Uses v4l2-ctl to strictly disable auto-adjustments."""
        try:
            # Disable Auto Exposure (1=Manual, 3=Auto)
            subprocess.run(f"v4l2-ctl -d /dev/video{self.device_index} -c exposure_auto=1", shell=True)
            # Disable Auto White Balance (0=Off, 1=On)
            subprocess.run(f"v4l2-ctl -d /dev/video{self.device_index} -c white_balance_automatic=0", shell=True)
            print("  -> Camera exposure and white balance LOCKED.")
        except Exception as e:
            print(f"  -> Warning: Failed to lock controls: {e}")

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def get_frame(self):
        """Captures the latest frame. Flushes buffer to ensure freshness."""
        status = False
        frame = None
        # Loop to clear the internal buffer so we get the *current* reality
        for _ in range(5):
            status, frame = self.stream.read()
        
        if status:
            return frame
        return None

    def get_rgb(self):
        """Returns frame in RGB for Matplotlib."""
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_brightness(self, frame_rgb):
        """Calculates average brightness (0-255)."""
        # Convert RGB back to Gray for calculation (or calculate on BGR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return np.mean(gray)

if __name__ == "__main__":
    # Initialize Camera with lock enabled
    camera = Camera(manual_exposure=True, device_index=0)
    
    # Capture Baseline
    initial_image = camera.get_rgb()
    baseline_brightness = camera.get_brightness(initial_image)
    print(f"Baseline Brightness established: {baseline_brightness:.2f}")

    # Setup Matplotlib
    fig, ax = plt.subplots()
    img = ax.imshow(initial_image)
    ax.set_title(f"Status: Stable (Base: {baseline_brightness:.1f})")
    
    # Hide axis ticks for cleaner view
    ax.axis('off')

    THRESHOLD = 20.0 # Adjust this sensitivity as needed

    def _update_image(frame_idx):
        # 1. Get new image
        image = camera.get_rgb()
        if image is None:
            return [img]

        # 2. Update the display
        img.set_array(image)
        
        # 3. Calculate Change
        curr_brightness = camera.get_brightness(image)
        diff = curr_brightness - baseline_brightness
        
        # 4. Determine Status
        status_text = "Stable"
        color = "black"
        
        if diff > THRESHOLD:
            status_text = "LIGHTS ON DETECTED"
            color = "red"
        elif diff < -THRESHOLD:
            status_text = "LIGHTS OFF DETECTED"
            color = "blue"
            
        # 5. Update Title with status
        ax.set_title(f"Status: {status_text}\nBright: {curr_brightness:.1f} (Diff: {diff:+.1f})", color=color)

        return [img, ax.title]

    # Interval is in milliseconds. 1000ms = 1 update per second.
    ani = animation.FuncAnimation(fig, _update_image, interval=1000, blit=False)
    plt.show()
