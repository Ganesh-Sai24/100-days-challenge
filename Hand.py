import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import threading
from pystray import Icon, MenuItem, Menu
from PIL import Image
import os
from pydub import AudioSegment
from pydub.playback import play
from threading import Event

# Define CLSCTX constants
CLSCTX_ALL = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Get the default audio endpoint
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # Corrected this line
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the current volume level at startup
initial_volume = volume.GetMasterVolumeLevelScalar()
volume_step = 0.02
movement_threshold = 0.01
horizontal_movement_threshold = 0.05

class VolumeControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.is_running = True
        self.previous_y = None
        self.previous_x = None
        self.previous_volume = initial_volume
        volume.SetMasterVolumeLevelScalar(initial_volume, None)
        self.playing = False
        self.media_event = Event()
        self.audio_file = 'your_audio_file.mp3'  # Update with your audio file path
        self.volume_thread = threading.Thread(target=self.run_volume_control)
        self.volume_thread.start()

    def initUI(self):
        self.setWindowTitle('Volume Control')
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        self.width = screen_rect.width() // 2
        self.height = screen_rect.height()
        self.setGeometry(0, 0, self.width, self.height)
        self.show()

    def run_volume_control(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Retrying...")
                cap.release()
                cap = cv2.VideoCapture(0)
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                if self.are_fingertips_visible(hand_landmarks):
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    current_y = index_finger_tip.y
                    current_x = index_finger_tip.x

                    # Volume control logic
                    if self.previous_y is not None:
                        vertical_movement = current_y - self.previous_y
                        if vertical_movement > movement_threshold:
                            new_volume = max(0.0, volume.GetMasterVolumeLevelScalar() - volume_step)
                            volume.SetMasterVolumeLevelScalar(new_volume, None)
                        elif vertical_movement < -movement_threshold:
                            new_volume = min(1.0, volume.GetMasterVolumeLevelScalar() + volume_step)
                            volume.SetMasterVolumeLevelScalar(new_volume, None)

                    if self.previous_x is not None:
                        horizontal_movement = current_x - self.previous_x
                        if horizontal_movement < -horizontal_movement_threshold:
                            new_volume = max(0.0, volume.GetMasterVolumeLevelScalar() - volume_step)
                            volume.SetMasterVolumeLevelScalar(new_volume, None)
                        elif horizontal_movement > horizontal_movement_threshold:
                            new_volume = min(1.0, volume.GetMasterVolumeLevelScalar() + volume_step)
                            volume.SetMasterVolumeLevelScalar(new_volume, None)

                    self.previous_y = current_y
                    self.previous_x = current_x

                if self.is_fist_closed(hand_landmarks):
                    volume.SetMasterVolumeLevelScalar(0.0, None)
                    print("Volume muted")
                else:
                    current_volume = volume.GetMasterVolumeLevelScalar()
                    if current_volume == 0.0:
                        volume.SetMasterVolumeLevelScalar(self.previous_volume, None)
                        print(f"Volume restored to: {self.previous_volume * 100:.0f}%")

                if self.is_play_pause_gesture(hand_landmarks):
                    self.playing = not self.playing
                    if self.playing:
                        self.media_event.set()
                        print("Playing media...")
                        threading.Thread(target=self.play_media, daemon=True).start()
                    else:
                        self.media_event.clear()
                        print("Paused media...")

                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            else:
                # Display message at the center of the top part
                message = 'Place your hand properly'
                font_scale = 1.5
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = (frame.shape[1] - text_width) // 2
                text_y = text_height + 20  # Padding from the top
                cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

            current_volume = volume.GetMasterVolumeLevelScalar()
            cv2.putText(frame, f'Volume: {current_volume * 100:.0f}%', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def are_fingertips_visible(self, hand_landmarks):
        if hand_landmarks:
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            for landmark in [mp_hands.HandLandmark.THUMB_TIP,
                             mp_hands.HandLandmark.INDEX_FINGER_TIP,
                             mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                             mp_hands.HandLandmark.RING_FINGER_TIP,
                             mp_hands.HandLandmark.PINKY_TIP]:
                if hand_landmarks.landmark[landmark].y >= wrist_y:
                    return False
            return True
        return False

    def is_fist_closed(self, hand_landmarks):
        if hand_landmarks:
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            return (thumb_tip_y >= wrist_y and index_tip_y >= wrist_y and 
                    middle_tip_y >= wrist_y and ring_tip_y >= wrist_y and 
                    pinky_tip_y >= wrist_y)
        return False

    def is_play_pause_gesture(self, hand_landmarks):
        if hand_landmarks:
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            return abs(index_tip_y - thumb_tip_y) < 0.05 and index_tip_y < wrist_y
        return False

    def closeEvent(self, event):
        self.is_running = False
        event.accept()

    def play_media(self):
        if os.path.isfile(self.audio_file):
            audio = AudioSegment.from_file(self.audio_file)
            while self.playing:
                play(audio)
                self.media_event.wait()
        else:
            print(f"Audio file '{self.audio_file}' not found.")

def setup_tray_icon():
    menu = Menu(MenuItem('Quit', on_quit))
    
    icon_path = 'icon.ico'
    if not os.path.isfile(icon_path):
        print(f"Icon file '{icon_path}' not found. Using default icon.")
        icon_image = Image.new('RGB', (64, 64), color='gray')
    else:
        icon_image = Image.open(icon_path)
    
    icon = Icon('Volume Control', icon=icon_image, menu=menu)
    icon.run()

def on_quit(icon, item):
    icon.stop()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    volume_control_app = VolumeControlApp()

    tray_thread = threading.Thread(target=setup_tray_icon)
    tray_thread.start()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("Application terminated by user.")
        volume_control_app.is_running = False
