import os
import sys
import time
import pickle
import threading
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

import customtkinter as ctk

# Try importing pyttsx3 for text-to-speech, fallback gracefully if unavailable
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Configure CustomTkinter Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ASLTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("RYWASL - ASL Real-Time Sign Language Translator")
        self.geometry("1100x700")
        self.minsize(950, 600)

        # --- Model & ML Pipeline Initialization ---
        self.model_path = os.path.join(os.path.dirname(__file__), "asl_model.p")
        self.model = None
        self.load_model()

        # --- MediaPipe Hands Setup ---
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # --- Webcam Setup ---
        self.cap = cv2.VideoCapture(0)
        self.is_camera_running = True
        self.draw_skeleton = True

        # --- Gesture Hold & Debounce Variables ---
        self.current_prediction = "-"
        self.last_prediction = None
        self.prediction_hold_count = 0
        self.HOLD_THRESHOLD = 12  # frames required to commit letter (~0.4 sec)
        self.committed_letter = None

        # --- Text to Speech Engine ---
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except Exception as e:
                print(f"TTS Initialization Warning: {e}")
                self.tts_engine = None

        # --- Build UI Layout ---
        self.setup_ui()

        # --- Start Video Processing Loop ---
        self.update_frame()

    def load_model(self):
        """Loads the trained pickle RandomForest model."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model', model_data)
                print(" Successfully loaded ASL model!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model file '{self.model_path}' not found!")
            self.model = None

    def setup_ui(self):
        """Builds the main user interface."""
        # Grid weights: Left 65% (Video), Right 35% (Controls & Text)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT PANEL: Video Feed & Camera Controls
        # ==========================================
        self.left_panel = ctk.CTkFrame(self, corner_radius=15)
        self.left_panel.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="nsew")
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=0)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Video Frame Container
        self.video_container = ctk.CTkFrame(self.left_panel, fg_color="#1a1a1a", corner_radius=12)
        self.video_container.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="nsew")
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        # Video Canvas / Label
        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="Initializing Camera Feed...",
            font=ctk.CTkFont(size=16),
            text_color="#888888"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Camera Control Toolbar
        self.cam_toolbar = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.cam_toolbar.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")

        self.toggle_cam_btn = ctk.CTkButton(
            self.cam_toolbar,
            text="⏸ Pause Feed",
            command=self.toggle_camera,
            width=130,
            fg_color="#343a40",
            hover_color="#495057"
        )
        self.toggle_cam_btn.pack(side="left", padx=(0, 10))

        self.toggle_skel_btn = ctk.CTkButton(
            self.cam_toolbar,
            text="🦴 Hide Skeleton",
            command=self.toggle_skeleton,
            width=130,
            fg_color="#343a40",
            hover_color="#495057"
        )
        self.toggle_skel_btn.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(
            self.cam_toolbar,
            text="Model Ready" if self.model else "⚠️ No Model Loaded",
            text_color="#28a745" if self.model else "#dc3545",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.status_label.pack(side="right", padx=10)

        # ==========================================
        # RIGHT PANEL: Predictions & Sentence Builder
        # ==========================================
        self.right_panel = ctk.CTkFrame(self, corner_radius=15)
        self.right_panel.grid(row=0, column=1, padx=(10, 15), pady=15, sticky="nsew")

        # Title Banner
        self.app_title = ctk.CTkLabel(
            self.right_panel,
            text="🤟 RYWASL ASL Translator",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.app_title.pack(pady=(20, 10))

        # --- Current Sign Card ---
        self.pred_card = ctk.CTkFrame(self.right_panel, fg_color="#2b2d31", corner_radius=12)
        self.pred_card.pack(fill="x", padx=15, pady=10)

        self.pred_card_header = ctk.CTkLabel(
            self.pred_card,
            text="DETECTED LETTER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#aaaaaa"
        )
        self.pred_card_header.pack(pady=(12, 0))

        self.pred_letter_label = ctk.CTkLabel(
            self.pred_card,
            text="-",
            font=ctk.CTkFont(size=56, weight="bold"),
            text_color="#3b82f6"
        )
        self.pred_letter_label.pack(pady=(0, 5))

        # Hold Progress Bar
        self.hold_progress = ctk.CTkProgressBar(self.pred_card, height=6, progress_color="#10b981")
        self.hold_progress.set(0)
        self.hold_progress.pack(fill="x", padx=25, pady=(0, 12))

        # --- Sentence Builder ---
        self.sentence_heading = ctk.CTkLabel(
            self.right_panel,
            text="Translated Text Buffer",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        self.sentence_heading.pack(fill="x", padx=15, pady=(15, 5))

        self.textbox = ctk.CTkTextbox(
            self.right_panel,
            font=ctk.CTkFont(size=18),
            corner_radius=10,
            border_width=1,
            border_color="#3f3f46"
        )
        self.textbox.pack(fill="both", expand=True, padx=15, pady=5)

        # --- Text Action Controls ---
        self.action_grid = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.action_grid.pack(fill="x", padx=15, pady=(10, 15))

        self.btn_space = ctk.CTkButton(
            self.action_grid,
            text="␣ Space",
            command=self.insert_space,
            width=80,
            fg_color="#2563eb",
            hover_color="#1d4ed8"
        )
        self.btn_space.grid(row=0, column=0, padx=3, pady=3, sticky="ew")

        self.btn_backspace = ctk.CTkButton(
            self.action_grid,
            text="⌫ Backspace",
            command=self.backspace_text,
            width=90,
            fg_color="#4b5563",
            hover_color="#374151"
        )
        self.btn_backspace.grid(row=0, column=1, padx=3, pady=3, sticky="ew")

        self.btn_clear = ctk.CTkButton(
            self.action_grid,
            text="🗑 Clear",
            command=self.clear_text,
            width=80,
            fg_color="#ef4444",
            hover_color="#dc2626"
        )
        self.btn_clear.grid(row=0, column=2, padx=3, pady=3, sticky="ew")

        self.btn_speak = ctk.CTkButton(
            self.action_grid,
            text="🔊 Speak",
            command=self.speak_sentence,
            width=80,
            fg_color="#10b981",
            hover_color="#059669",
            state="normal" if TTS_AVAILABLE else "disabled"
        )
        self.btn_speak.grid(row=0, column=3, padx=3, pady=3, sticky="ew")

        self.action_grid.grid_columnconfigure((0, 1, 2, 3), weight=1)

    # ==========================================
    # LOGIC & VIDEO LOOP
    # ==========================================
    def update_frame(self):
        """Main update loop capturing frame, detecting sign, and updating UI."""
        if self.is_camera_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                # Flip frame horizontally for natural mirror view
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process hand landmarks
                results = self.hands.process(image_rgb)
                detected_letter = "-"

                if results.multi_hand_landmarks and self.model:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks on frame if enabled
                        if self.draw_skeleton:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                        # Extract Relative Landmarks (wrist point 0 subtracted from points 1..20)
                        wrist = hand_landmarks.landmark[0]
                        features = []
                        for i in range(1, 21):
                            lm = hand_landmarks.landmark[i]
                            features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                        # AI Prediction
                        try:
                            prediction = self.model.predict([features])
                            detected_letter = str(prediction[0]).upper()
                        except Exception as err:
                            print(f"Prediction Error: {err}")

                        # Highlight hand bounding box
                        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                        x_min, y_min = max(0, min(x_coords) - 15), max(0, min(y_coords) - 35)
                        x_max, y_max = min(w, max(x_coords) + 15), min(h, max(y_coords) + 15)

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (59, 130, 246), 2)
                        cv2.putText(
                            frame,
                            f"ASL: {detected_letter}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (59, 130, 246),
                            2
                        )

                # Update Debounce / Hold Logic
                self.process_hold_logic(detected_letter)

                # Convert OpenCV Frame to CustomTkinter Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Dynamic sizing to match widget
                vw = max(320, self.video_container.winfo_width() - 20)
                vh = max(240, self.video_container.winfo_height() - 20)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(vw, vh))

                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

        # Loop again after 15ms (~60 FPS target)
        self.after(15, self.update_frame)

    def process_hold_logic(self, detected_letter):
        """Debounces and holds predictions before appending to sentence box."""
        self.pred_letter_label.configure(text=detected_letter)

        if detected_letter != "-" and detected_letter == self.last_prediction:
            self.prediction_hold_count += 1
            progress = min(1.0, self.prediction_hold_count / self.HOLD_THRESHOLD)
            self.hold_progress.set(progress)

            if self.prediction_hold_count >= self.HOLD_THRESHOLD:
                if self.committed_letter != detected_letter:
                    self.textbox.insert("end", detected_letter)
                    self.committed_letter = detected_letter
                    self.hold_progress.configure(progress_color="#10b981")
        else:
            self.prediction_hold_count = 0
            self.last_prediction = detected_letter
            self.committed_letter = None
            self.hold_progress.set(0)

    # ==========================================
    # EVENT HANDLERS & BUTTON ACTIONS
    # ==========================================
    def toggle_camera(self):
        """Pauses or resumes webcam capture."""
        self.is_camera_running = not self.is_camera_running
        if self.is_camera_running:
            self.toggle_cam_btn.configure(text="⏸ Pause Feed")
        else:
            self.toggle_cam_btn.configure(text="▶ Resume Feed")

    def toggle_skeleton(self):
        """Toggles MediaPipe drawing overlays."""
        self.draw_skeleton = not self.draw_skeleton
        self.toggle_skel_btn.configure(text="🦴 Hide Skeleton" if self.draw_skeleton else "🦴 Show Skeleton")

    def insert_space(self):
        """Inserts space character into text buffer."""
        self.textbox.insert("end", " ")

    def backspace_text(self):
        """Deletes last character from text buffer."""
        content = self.textbox.get("0.0", "end-1c")
        if content:
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", content[:-1])

    def clear_text(self):
        """Clears text buffer."""
        self.textbox.delete("0.0", "end")

    def speak_sentence(self):
        """Speaks sentence using TTS engine in a background thread."""
        text = self.textbox.get("0.0", "end-1c").strip()
        if text and self.tts_engine:
            def _speak():
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Speak Error: {e}")

            threading.Thread(target=_speak, daemon=True).start()

    def on_closing(self):
        """Clean shutdown when closing window."""
        self.is_camera_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = ASLTranslatorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
