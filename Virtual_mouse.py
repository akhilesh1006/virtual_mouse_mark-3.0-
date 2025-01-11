import cv2
import pyautogui
import time
import math
import threading

class HandGestureMouseController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Get screen and frame dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width, self.frame_height = 640, 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Smaller rectangle for movement area (50% of frame size)
        self.rect_width, self.rect_height = int(self.frame_width * 0.5), int(self.frame_height * 0.5)
        self.rect_x, self.rect_y = (self.frame_width - self.rect_width) // 2, (self.frame_height - self.rect_height) // 2

        # Scaling factors
        self.x_scale = self.screen_width / self.rect_width
        self.y_scale = self.screen_height / self.rect_height

        # Smooth mouse movement
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.6

        # Click delays
        self.click_delay = 0.7  # Left click delay
        self.right_click_delay = 0.5  # Right click delay
        self.last_left_click_time = 0
        self.last_right_click_time = 0

    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

    def fingers_up(self, landmarks):
        """Check which fingers are up."""
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [3, 6, 10, 14, 18]

        fingers = []

        # Thumb check (left/right depending on hand orientation)
        fingers.append(landmarks[finger_tips[0]].x < landmarks[finger_bases[0]].x)

        # Other fingers: Tip above base in y-axis
        for tip, base in zip(finger_tips[1:], finger_bases[1:]):
            fingers.append(landmarks[tip].y < landmarks[base].y)

        return [1 if finger else 0 for finger in fingers]

    def move_mouse(self, target_x, target_y):
        """Smooth mouse movement."""
        new_x = int(self.prev_x + self.smoothing_factor * (target_x - self.prev_x))
        new_y = int(self.prev_y + self.smoothing_factor * (target_y - self.prev_y))
        pyautogui.moveTo(new_x, new_y)
        self.prev_x, self.prev_y = new_x, new_y

    def process_frame(self, frame):
        """Process each frame and handle gestures."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Check fingers up
                fingers = self.fingers_up(landmarks)

                # Move cursor: Only when index finger is up and others are down
                if fingers == [0, 1, 0, 0, 0]:
                    index_tip = landmarks[8]

                    # Ensure index tip is within the green rectangle
                    if (self.rect_x <= index_tip.x * self.frame_width <= self.rect_x + self.rect_width and
                        self.rect_y <= index_tip.y * self.frame_height <= self.rect_y + self.rect_height):

                        # Map index finger position to screen coordinates
                        screen_x = int((index_tip.x * self.frame_width - self.rect_x) * self.x_scale)
                        screen_y = int((index_tip.y * self.frame_height - self.rect_y) * self.y_scale)
                        self.move_mouse(screen_x, screen_y)

                # Right-click: Thumb and index fingers up, and close together
                if fingers[0] == 1 and fingers[1] == 1:  # Thumb and index up
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    if self.calculate_distance(thumb_tip, index_tip) < 0.05:
                        current_time = time.time()
                        if current_time - self.last_right_click_time > self.right_click_delay:
                            pyautogui.click(button='right')
                            self.last_right_click_time = current_time

                # Left-click: Index and middle fingers up, and close together
                if fingers[1] == 1 and fingers[2] == 1:  # Index and middle up
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]
                    if self.calculate_distance(index_tip, middle_tip) < 0.05:
                        current_time = time.time()
                        if current_time - self.last_left_click_time > self.click_delay:
                            pyautogui.click(button='left')
                            self.last_left_click_time = current_time
