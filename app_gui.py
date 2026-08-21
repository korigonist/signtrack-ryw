from PIL import ImagePalette
import os
import sys
import pickle
import threading
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

import customtkinter as ctk

# Try importing keras / tensorflow for Word/Action LSTM model
TF_AVAILABLE = False
KERAS_LOAD_MODEL = None

try:
    import keras
    KERAS_LOAD_MODEL = keras.models.load_model
    TF_AVAILABLE = True
except Exception:
    try:
        import tensorflow as tf
        KERAS_LOAD_MODEL = tf.keras.models.load_model
        TF_AVAILABLE = True
    except Exception:
        TF_AVAILABLE = False

# Try importing pyttsx3 for text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Configure CustomTkinter Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller (_MEIPASS)."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


class ASLTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("SignTrack")
        self.geometry("1150x720")
        self.minsize(980, 620)

        # --- Mode State: "Alphabet" or "Word" ---
        self.translation_mode = "Alphabet"

        # --- 1. Load Alphabet Model (asl_model.p) ---
        self.alphabet_model_path = get_resource_path("asl_model.p")
        self.alphabet_model = None
        self.load_alphabet_model()

        # --- 2. Load Word/Action Model (action.keras / action.h5) ---
        self.word_model = None
        self.actions = np.array(['hello', 'thanks', 'iloveyou'])
        self.load_word_model()

        # --- MediaPipe Pipelines ---
        self.mp_hands = mp.solutions.hands
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Single hand tracker for Alphabet mode
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Holistic tracker for Word/Action mode
        self.holistic_detector = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Webcam Setup ---
        self.cap = cv2.VideoCapture(0)
        self.is_camera_running = True
        self.draw_skeleton = True

        # --- Alphabet Mode Hold/Debounce Variables ---
        self.current_alphabet_pred = "-"
        self.last_alphabet_pred = None
        self.alphabet_hold_count = 0
        self.ALPHABET_HOLD_THRESHOLD = 12  # frames required to commit (~0.4 sec)
        self.committed_letter = None

        # --- Word Mode Sequence Variables ---
        self.word_sequence = []
        self.word_predictions = []
        self.word_threshold = 0.8
        self.current_word_pred = "-"

        # --- Text-to-Speech Engine ---
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

        # --- Start Video Loop ---
        self.update_frame()

    def load_alphabet_model(self):
        """Loads the RandomForest Alphabet model (asl_model.p)."""
        if os.path.exists(self.alphabet_model_path):
            try:
                with open(self.alphabet_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.alphabet_model = model_data.get('model', model_data)
                print(" Successfully loaded Alphabet model (asl_model.p)!")
            except Exception as e:
                print(f"❌ Error loading Alphabet model: {e}")
                self.alphabet_model = None
        else:
            print(f"⚠️ Alphabet model file '{self.alphabet_model_path}' not found!")

    def load_word_model(self):
        # Loads LSTM model
        if not TF_AVAILABLE and KERAS_LOAD_MODEL is None:
            print("⚠️ Keras/TensorFlow not available for Word model.")
            return

        keras_path = get_resource_path("action.keras")
        h5_path = get_resource_path("action.h5")

        model_file = keras_path if os.path.exists(keras_path) else (h5_path if os.path.exists(h5_path) else None)

        if model_file:
            loaded_model = None
            # 1. Try KERAS_LOAD_MODEL
            if KERAS_LOAD_MODEL:
                try:
                    loaded_model = KERAS_LOAD_MODEL(model_file)
                except Exception as e:
                    print(f"Primary load attempt error: {e}")

            # 2. Fallback to keras or tensorflow.keras directly
            if loaded_model is None:
                try:
                    import keras
                    loaded_model = keras.models.load_model(model_file)
                except Exception:
                    try:
                        import tensorflow as tf
                        loaded_model = tf.keras.models.load_model(model_file)
                    except Exception as e:
                        print(f"Fallback load attempt error: {e}")

            if loaded_model is not None:
                self.word_model = loaded_model
                print(f" Successfully loaded Word/Action model ({os.path.basename(model_file)})!")
            else:
                print(f" Error loading Word model from {model_file}")
                self.word_model = None
        else:
            print("No Word model file (action.keras/action.h5) found.")

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        icon_path = get_resource_path("images/creatoricon.png")

        try:
            pil_image = Image.open(icon_path)
            about_icon = ctk.CTkImage(pil_image, size=(20, 20))
        except:
            about_icon = None

        # ==========================================
        # LEFT PANEL: Video Feed & Camera Controls
        # ==========================================
        self.left_panel = ctk.CTkFrame(self, corner_radius=15)
        self.left_panel.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="nsew")
        self.left_panel.grid_rowconfigure(0, weight=0)
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_rowconfigure(2, weight=0)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Top Header Bar (Top Left About Button)
        self.top_header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.top_header.grid(row=0, column=0, padx=15, pady=(15, 0), sticky="ew")

        self.about_btn = ctk.CTkButton(
            self.top_header,
            text="",
            command=self.open_about_window,
            image=about_icon,
            width=16,
            height=16,
            corner_radius=16,
            fg_color="#343a40",
            hover_color="#495057"
        )

        # something to commit once again
        self.about_btn.pack(side="left")
        # self.about_btn = ctk.CTkButton(
        #     self.top_header,
        #     text="i",
        #     command=self.open_about_window,
        #     width=32,
        #     height=32,
        #     corner_radius=16,
        #     font=ctk.CTkFont(size=16),
        #     fg_color="#343a40",
        #     hover_color="#495057"
        # )
        # self.about_btn.pack(side="left")

        # Video Frame Container
        self.video_container = ctk.CTkFrame(self.left_panel, fg_color="#1a1a1a", corner_radius=12)
        self.video_container.grid(row=1, column=0, padx=15, pady=(10, 10), sticky="nsew")
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_container,
            text="Initializing Camera Feed...",
            font=ctk.CTkFont(size=16),
            text_color="#888888"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Toolbar
        self.cam_toolbar = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.cam_toolbar.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="ew")

        self.toggle_cam_btn = ctk.CTkButton(
            self.cam_toolbar,
            text="Pause Feed",
            command=self.toggle_camera,
            width=130,
            fg_color="#343a40",
            hover_color="#495057"
        )
        self.toggle_cam_btn.pack(side="left", padx=(0, 10))

        self.toggle_skel_btn = ctk.CTkButton(
            self.cam_toolbar,
            text="Hide Skeleton",
            command=self.toggle_skeleton,
            width=130,
            fg_color="#343a40",
            hover_color="#495057"
        )
        self.toggle_skel_btn.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(
            self.cam_toolbar,
            text="Models Ready",
            text_color="#28a745",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.status_label.pack(side="right", padx=10)

        # ==========================================
        # RIGHT PANEL: Translation Controls & Text
        # ==========================================
        self.right_panel = ctk.CTkFrame(self, corner_radius=15)
        self.right_panel.grid(row=0, column=1, padx=(10, 15), pady=15, sticky="nsew")

        # Title
        self.app_title = ctk.CTkLabel(
            self.right_panel,
            text="SignTrack",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        self.app_title.pack(pady=(15, 2))

        # Subtitle / Placeholder text
        self.app_subtitle = ctk.CTkLabel(
            self.right_panel,
            text="Sign language translator with hand tracking",
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        )
        self.app_subtitle.pack(pady=(0, 12))

        # --- MODE SELECTOR ---
        self.mode_selector = ctk.CTkSegmentedButton(
            self.right_panel,
            values=["Alphabet Mode", "Word Mode"],
            command=self.on_mode_change,
            selected_color="#2563eb",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.mode_selector.set("Alphabet Mode")
        self.mode_selector.pack(fill="x", padx=15, pady=(0, 10))

        # Detected Output Card
        self.pred_card = ctk.CTkFrame(self.right_panel, fg_color="#2b2d31", corner_radius=12)
        self.pred_card.pack(fill="x", padx=15, pady=5)

        self.pred_card_header = ctk.CTkLabel(
            self.pred_card,
            text="DETECTED LETTER",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#aaaaaa"
        )
        self.pred_card_header.pack(pady=(10, 0))

        self.pred_letter_label = ctk.CTkLabel(
            self.pred_card,
            text="-",
            font=ctk.CTkFont(size=46, weight="bold"),
            text_color="#3b82f6"
        )
        self.pred_letter_label.pack(pady=(0, 2))

        # Hold / Sequence Progress Bar
        self.hold_progress = ctk.CTkProgressBar(self.pred_card, height=6, progress_color="#10b981")
        self.hold_progress.set(0)
        self.hold_progress.pack(fill="x", padx=25, pady=(0, 10))

        # --- Text Buffer Area ---
        self.sentence_heading = ctk.CTkLabel(
            self.right_panel,
            text="Translated Text Output",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        self.sentence_heading.pack(fill="x", padx=15, pady=(10, 3))

        self.textbox = ctk.CTkTextbox(
            self.right_panel,
            font=ctk.CTkFont(size=17),
            corner_radius=10,
            border_width=1,
            border_color="#3f3f46"
        )
        self.textbox.pack(fill="both", expand=True, padx=15, pady=5)

        # action buttons 
        self.action_grid = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.action_grid.pack(fill="x", padx=15, pady=(8, 12))

        self.btn_space = ctk.CTkButton(
            self.action_grid,
            text="Space",
            command=self.insert_space,
            width=75,
            fg_color="#2563eb",
            hover_color="#1d4ed8"
        )
        self.btn_space.grid(row=0, column=0, padx=4, pady=2, sticky="ew")

        self.btn_backspace = ctk.CTkButton(
            self.action_grid,
            text="Backspace",
            command=self.backspace_text,
            width=85,
            fg_color="#4b5563",
            hover_color="#374151"
        )
        self.btn_backspace.grid(row=0, column=1, padx=4, pady=2, sticky="ew")

        self.btn_clear = ctk.CTkButton(
            self.action_grid,
            text="Clear",
            command=self.clear_text,
            width=75,
            fg_color="#ef4444",
            hover_color="#dc2626"
        )
        self.btn_clear.grid(row=0, column=2, padx=4, pady=2, sticky="ew")

        self.action_grid.grid_columnconfigure((0, 1, 2), weight=1)

    def on_mode_change(self, value):
        # toggle between letter mode and word mode
        if "Alphabet" in value:
            self.translation_mode = "Alphabet"
            self.pred_card_header.configure(text="DETECTED LETTER")
            self.pred_letter_label.configure(text="-")
            self.hold_progress.set(0)
            print("Switched to Letter Mode")
        else:
            self.translation_mode = "Word"
            self.pred_card_header.configure(text="DETECTED WORD / ACTION")
            self.pred_letter_label.configure(text="-")
            self.word_sequence = []
            self.hold_progress.set(0)
            print("Switched to Word Mode")

    # VIDEO & PREDICTION PIPELINE
    def update_frame(self):
        if self.is_camera_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                frame = cv2.flip(frame, 1)

                if self.translation_mode == "Alphabet":
                    frame = self.process_alphabet_mode(frame)
                else:
                    frame = self.process_word_mode(frame)

                # Convert frame for CustomTkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                vw = max(320, self.video_container.winfo_width() - 20)
                vh = max(240, self.video_container.winfo_height() - 20)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(vw, vh))

                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

        self.after(15, self.update_frame)

    def process_alphabet_mode(self, frame):
        """Alphabet mode processing using asl_model.p and Hands detector."""
        h, w, c = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(image_rgb)
        detected_letter = "-"

        if results.multi_hand_landmarks and self.alphabet_model:
            for hand_landmarks in results.multi_hand_landmarks:
                if self.draw_skeleton:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                wrist = hand_landmarks.landmark[0]
                features = []
                for i in range(1, 21):
                    lm = hand_landmarks.landmark[i]
                    features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                try:
                    prediction = self.alphabet_model.predict([features])
                    detected_letter = str(prediction[0]).upper()
                except Exception as err:
                    print(f"Alphabet Model Prediction Error: {err}")

                # Bounding box
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, y_min = max(0, min(x_coords) - 15), max(0, min(y_coords) - 35)
                x_max, y_max = min(w, max(x_coords) + 15), min(h, max(y_coords) + 15)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (59, 130, 246), 2)
                cv2.putText(frame, f"Letter: {detected_letter}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2)

        # Debounce & commit letter
        self.process_alphabet_hold_logic(detected_letter)
        return frame

    def process_alphabet_hold_logic(self, detected_letter):
        self.pred_letter_label.configure(text=detected_letter)

        if detected_letter != "-" and detected_letter == self.last_alphabet_pred:
            self.alphabet_hold_count += 1
            progress = min(1.0, self.alphabet_hold_count / self.ALPHABET_HOLD_THRESHOLD)
            self.hold_progress.set(progress)

            if self.alphabet_hold_count >= self.ALPHABET_HOLD_THRESHOLD:
                if self.committed_letter != detected_letter:
                    self.textbox.insert("end", detected_letter)
                    self.committed_letter = detected_letter
                    self.hold_progress.configure(progress_color="#10b981")
        else:
            self.alphabet_hold_count = 0
            self.last_alphabet_pred = detected_letter
            self.committed_letter = None
            self.hold_progress.set(0)

    def extract_holistic_keypoints(self, results):
        """Extracts 1662 keypoints for Holistic model (pose, face, left hand, right hand)."""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def process_word_mode(self, frame):
        """Word / Action mode processing using action.keras / action.h5 and Holistic detector."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic_detector.process(image_rgb)

        if self.draw_skeleton:
            self.draw_holistic_landmarks(frame, results)

        keypoints = self.extract_holistic_keypoints(results)
        self.word_sequence.append(keypoints)
        self.word_sequence = self.word_sequence[-30:]

        seq_len = len(self.word_sequence)

        predicted_word = "-"
        if seq_len < 30:
            self.hold_progress.set(seq_len / 30.0)
            self.pred_letter_label.configure(text=f"Buffering... ({seq_len}/30)")
        elif self.word_model:
            try:
                res = self.word_model.predict(np.expand_dims(self.word_sequence, axis=0), verbose=0)[0]
                best_idx = np.argmax(res)
                confidence = float(res[best_idx])

                self.word_predictions.append(best_idx)
                self.word_predictions = self.word_predictions[-20:]

                recent_preds = self.word_predictions[-10:]
                match_count = sum(1 for p in recent_preds if p == best_idx)
                self.hold_progress.set(match_count / 10.0)

                clean_word = self.actions[best_idx].upper()

                if confidence > 0.5:
                    predicted_word = f"{clean_word} ({int(confidence * 100)}%)"

                # Commit word if consistent across last 10 predictions and above threshold
                if confidence >= self.word_threshold and len(recent_preds) == 10 and all(p == best_idx for p in recent_preds):
                    current_text = self.textbox.get("0.0", "end-1c")
                    words = current_text.strip().split()
                    if not words or words[-1].upper() != clean_word:
                        prefix = " " if current_text and not current_text.endswith(" ") else ""
                        self.textbox.insert("end", f"{prefix}{clean_word}")
                        if self.tts_engine:
                            self.speak_sentence()
            except Exception as e:
                print(f"Word Model Error: {e}")

            self.pred_letter_label.configure(text=predicted_word)

        return frame

    def draw_holistic_landmarks(self, image, results):
        """Draws face, pose, and hand landmarks for Word Mode."""
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

    # ==========================================
    # TOOLBAR & ACTION HANDLERS
    # ==========================================
    def toggle_camera(self):
        self.is_camera_running = not self.is_camera_running
        self.toggle_cam_btn.configure(text="Pause Feed" if self.is_camera_running else "Resume Feed")

    def toggle_skeleton(self):
        self.draw_skeleton = not self.draw_skeleton
        self.toggle_skel_btn.configure(text="Hide Skeleton" if self.draw_skeleton else "Show Skeleton")

    def open_about_window(self):
        if hasattr(self, 'about_window') and self.about_window is not None and self.about_window.winfo_exists():
            self.about_window.focus_force()
            return

        self.about_window = ctk.CTkToplevel(self)
        self.about_window.title("About")
        self.about_window.geometry("400x240")
        self.about_window.after(100, lambda: self.about_window.focus_force())

        # Title Label
        title_label = ctk.CTkLabel(
            self.about_window,
            text="Creators:",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(pady=(0, 0))

        # Bullet List Container
        bullet_frame = ctk.CTkFrame(self.about_window, fg_color="transparent")
        bullet_frame.pack(padx=40, pady=10, fill="x")

        creators = [
            "• Thampapont Maolanont (Class: M.4/18)",
            "• Pisit Boonyingruangrong (Class: M.4/18)",
            "• Gonchawin Chotpiyaanan (Class: M.4/18)"
        ]

        for creator in creators:
            item_label = ctk.CTkLabel(
                bullet_frame,
                text=creator,
                font=ctk.CTkFont(size=14),
                anchor="w"
            )
            item_label.pack(fill="x", pady=2)

        program_label = ctk.CTkLabel(
            self.about_window,
            text="English Program",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        program_label.pack(pady=(0, 0))

        # here's something to commit once again
        school_label = ctk.CTkLabel(
            self.about_window,
            text="Rayongwittayakom School",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        school_label.pack(pady=(0, 0))

        status_label = ctk.CTkLabel(
            self.about_window,
            text="SignTrack program (version 1.0), Released August 17th 2026",
            font=ctk.CTkFont(size=12)
        )
        status_label.pack(pady=(10, 0))

    def insert_space(self):
        self.textbox.insert("end", " ")

    def backspace_text(self):
        content = self.textbox.get("0.0", "end-1c")
        if content:
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", content[:-1])

    def clear_text(self):
        self.textbox.delete("0.0", "end")

    def speak_sentence(self):
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
        self.is_camera_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = ASLTranslatorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
