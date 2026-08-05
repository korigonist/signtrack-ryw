# from kivy.app import App
# from kivy.uix.screenmanager import ScreenManager, Screen
# from kivy.graphics import Color, RoundedRectangle
# from kivy.uix.button import Button
# from kivy.lang import Builder

# # ออกแบบ Interface ด้วย KV Language (ช่วยให้จัดหน้าตาได้ง่ายและตรงบรีฟที่สุด)
# Builder.load_string('''
# <MenuScreen>:
#     canvas.before:
#         Color:
#             rgba: 0.0, 0.0, 0.55, 1  # พื้นหลังสีน้ำเงินเข้มตามรูป (#00008C โดยประมาณ)
#         Rectangle:
#             pos: self.pos
#             size: self.size

#     BoxLayout:
#         orientation: 'vertical'
#         padding: [20, 80, 20, 100]
#         spacing: 30
#         alignment: 'center'

#         # ส่วนของ Logo โรงเรียน
#         Image:
#             source: 'RYWEP.png'  # เปลี่ยนเป็นชื่อไฟล์โลโก้ของคุณ
#             size_hint: (None, None)
#             size: (250, 250)
#             pos_hint: {'center_x': 0.5}

#         # ข้อความต้อนรับ
#         Label:
#             text: 'Welcome to our sign language translator'
#             font_size: '24sp'
#             bold: True
#             color: [1, 1, 1, 1]  # สีขาว
#             size_hint_y: None
#             height: 50

#         Widget:
#             size_hint_y: None
#             height: 20

#         # ปุ่ม Start
#         YellowButton:
#             text: 'Start'
#             on_release: root.manager.current = 'start_screen'

#         # ปุ่ม About us
#         YellowButton:
#             text: 'About us'
#             on_release: root.manager.current = 'about_screen'
            
#         Widget: # ตัวช่วยดันพื้นที่ด้านล่างให้สมดุล
#             size_hint_y: 1


# # หน้าจอจำลองอื่นๆ เมื่อกดปุ่ม
# <StartScreen>:
#     BoxLayout:
#         orientation: 'vertical'
#         Label:
#             text: 'Sign Language Translator Camera Screen (Placeholder)'
#         Button:
#             text: 'Back to Menu'
#             size_hint: (None, None)
#             size: (150, 50)
#             pos_hint: {'center_x': 0.5}
#             on_release: root.manager.current = 'menu'

# <AboutScreen>:
#     BoxLayout:
#         orientation: 'vertical'
#         Label:
#             text: 'Rayongwittayakom School - English Program\\nCreated by...'
#             halign: 'center'
#         Button:
#             text: 'Back to Menu'
#             size_hint: (None, None)
#             size: (150, 50)
#             pos_hint: {'center_x': 0.5}
#             on_release: root.manager.current = 'menu'
# ''')

# # สร้าง Custom Button เพื่อให้ได้ปุ่มขอบมนสีเหลืองตัวหนังสือสีดำตามรูป
# class YellowButton(Button):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.background_color = (0, 0, 0, 0)  # ซ่อนพื้นหลังเดิมของ Kivy
#         self.color = (0, 0, 0, 1)             # ตัวอักษรสีดำ
#         self.font_size = '22sp'
#         self.bold = True
#         self.size_hint = (None, None)
#         self.size = (280, 65)                 # ขนาดของปุ่ม
#         self.pos_hint = {'center_x': 0.5}
        
#         # วาดพื้นหลังปุ่มสีเหลืองขอบมน
#         with self.canvas.before:
#             Color(rgba=(0.98, 0.82, 0.28, 1))  # สีเหลืองสด (#FBD147)
#             self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[32.5])
            
#         self.bind(pos=self.update_rect, size=self.update_rect)

#     def update_rect(self, *args):
#         self.rect.pos = self.pos
#         self.rect.size = self.size

# # หน้าจอหลัก
# class MenuScreen(Screen):
#     pass

# # หน้าจอสำหรับระบบถัดไป
# class StartScreen(Screen):
#     pass

# class AboutScreen(Screen):
#     pass

# class ASLApp(App):
#     def build(self):
#         self.title = "ASL Translator - Rayongwittayakom School"
#         sm = ScreenManager()
#         sm.add_widget(MenuScreen(name='menu'))
#         sm.add_widget(StartScreen(name='start_screen'))
#         sm.add_widget(AboutScreen(name='about_screen'))
#         return sm

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import mediapipe as mp
import pickle
import numpy as np

# ออกแบบ Interface ด้วย KV Language
Builder.load_string('''
<MenuScreen>:
    canvas.before:
        Color:
            rgba: 0.0, 0.0, 0.55, 1  # พื้นหลังสีน้ำเงินเข้ม
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        orientation: 'vertical'
        padding: [20, 80, 20, 100]
        spacing: 30

        Image:
            source: 'RYWEP.png'  # โลโก้โรงเรียน
            size_hint: (None, None)
            size: (250, 250)
            pos_hint: {'center_x': 0.5}

        Label:
            text: 'Welcome to our sign language translator'
            font_size: '24sp'
            bold: True
            color: [1, 1, 1, 1]
            size_hint_y: None
            height: 50

        Widget:
            size_hint_y: None
            height: 20

        YellowButton:
            text: 'Start'
            on_release: root.manager.current = 'start_screen'

        YellowButton:
            text: 'About us'
            on_release: root.manager.current = 'about_screen'
            
        Widget:
            size_hint_y: 1

<StartScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        
        # ส่วนแสดงผลกล้องภายในแอพ Kivy
        Image:
            id: camera_preview
            allow_stretch: True
            keep_ratio: True
            size_hint_y: 0.85

        # ปุ่มกดย้อนกลับหน้าเมนู
        Button:
            text: 'Back to Menu'
            size_hint: (None, None)
            size: (180, 50)
            pos_hint: {'center_x': 0.5}
            on_release: root.go_back()

<AboutScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 20
        Label:
            text: 'Rayongwittayakom School - English Program\\n\\nCreated by Amazing Team'
            font_size: '20sp'
            halign: 'center'
        Button:
            text: 'Back to Menu'
            size_hint: (None, None)
            size: (180, 50)
            pos_hint: {'center_x': 0.5}
            on_release: root.manager.current = 'menu'
''')

# ปุ่มขอบมนสีเหลือง
class YellowButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.color = (0, 0, 0, 1)
        self.font_size = '22sp'
        self.bold = True
        self.size_hint = (None, None)
        self.size = (280, 65)
        self.pos_hint = {'center_x': 0.5}
        
        with self.canvas.before:
            Color(rgba=(0.98, 0.82, 0.28, 1))
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[32.5])
            
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


class MenuScreen(Screen):
    pass


# หน้าจอสำหรับเปิดกล้องแปลภาษามือ
class StartScreen(Screen):
    def on_enter(self):
        """ ทำงานอัตโนมัติเมื่อผู้ใช้กดเข้ามาที่หน้านี้ """
        # 1. โหลด AI Model
        try:
            with open('asl_model.p', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
        except FileNotFoundError:
            print("⚠️ ไม่พบไฟล์โมเดล 'asl_model.p' ระบบจะไม่แปลผลจนกว่าจะเทรนโมเดลเสร็จนะ!")
            self.model = None

        # 2. ตั้งค่า MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.target_ids = [0, 4, 8, 12, 16, 20] # 6 จุดอ้างอิงของคุณ

        # 3. เปิดกล้อง
        self.capture = cv2.VideoCapture(0)
        
        # 4. สั่งให้ฟังก์ชัน update ทำงานทุกๆ 1/30 วินาที (30 FPS)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        """ ดึงภาพจากกล้อง แปลผลภาษามือ และส่งไปแสดงบนหน้าจอ Kivy """
        success, frame = self.capture.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # วาดโครงกระดูกมือบนเฟรม
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # ดึงพิกัด (ดึงครบ 21 จุดก่อน แล้วค่อยกรองเหลือ 6 จุดตามสูตรเดิมของคุณ)
                all_lms = []
                for lm in hand_landmarks.landmark:
                    all_lms.append({
                        'x_norm': lm.x, 'y_norm': lm.y, 'z_norm': lm.z, 
                        'x_px': int(lm.x * w), 'y_px': int(lm.y * h)
                    })
                
                # กรองเฉพาะ 6 จุดสำคัญ
                current_landmarks = []
                x_pixel_list = []
                y_pixel_list = []
                for target_id in self.target_ids:
                    pt = all_lms[target_id]
                    current_landmarks.extend([pt['x_norm'], pt['y_norm'], pt['z_norm']])
                    x_pixel_list.append(pt['x_px'])
                    y_pixel_list.append(pt['y_px'])
                
                # ส่งให้ AI ทายคำตอบ (ถ้ามีโมเดลอยู่)
                if self.model:
                    prediction = self.model.predict([current_landmarks])
                    predicted_letter = prediction[0]
                    
                    # วาดกล่องข้อความแปลผลบนหน้าจอ OpenCV ก่อนแปลงเป็น Texture
                    x_min, y_min = min(x_pixel_list), min(y_pixel_list)
                    cv2.putText(frame, predicted_letter, (x_min, y_min - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # แปลงภาพ OpenCV (BGR) ให้กลายเป็น Texture ที่ Kivy สามารถเอาไปแสดงผลได้
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        
        # ส่งค่า Texture ไปอัปเดตที่วิดเจ็ตตระกูล Image ในไฟล์ KV
        self.ids.camera_preview.texture = texture

    def go_back(self):
        """ หยุดกล้องและย้อนกลับหน้าเมนู """
        Clock.unschedule(self.update)
        if self.capture:
            self.capture.release()
        self.manager.current = 'menu'

    def on_leave(self):
        """ ป้องกันกรณีสลับหน้าจอด้วยวิธีอื่น ให้แน่ใจว่ากล้องจะถูกปิด """
        Clock.unschedule(self.update)
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()


class AboutScreen(Screen):
    pass


class ASLApp(App):
    def build(self):
        self.title = "ASL Translator - Rayongwittayakom School"
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(StartScreen(name='start_screen'))
        sm.add_widget(AboutScreen(name='about_screen'))
        return sm


if __name__ == '__main__':
    ASLApp().run()