import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
from deepface import DeepFace
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import cycle

class AnimatedBackground(tk.Label):
    def __init__(self, parent, gif_path, **kwargs):
        super().__init__(parent, **kwargs)
        im = Image.open(gif_path)
        frames = []
        try:
            while True:
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(len(frames))
        except EOFError:
            pass
        self.frames = cycle(frames)
        self.delay = im.info.get('duration', 100)
        self.config(image=next(self.frames))
        self.after(self.delay, self.next_frame)

    def next_frame(self):
        self.config(image=next(self.frames))
        self.after(self.delay, self.next_frame)

class EmotionObjectApp:
    def __init__(self, root):
        self.root = root
        root.title("Emotion‑Object Interaction")
        root.geometry("1024x720")
        root.configure(bg="#1a1a1a")

        bg = AnimatedBackground(root, "assets/loop.gif")
        bg.place(relwidth=1, relheight=1)

        panel = tk.Frame(root, bg="#242424", bd=2, relief="ridge")
        panel.place(relx=0.05, rely=0.05, relwidth=0.90, relheight=0.90)

        self.cap = None
        self.running = False
        self.csv_file = "data/output.csv"
        self.model = YOLO("yolov8n.pt")
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        with open(self.csv_file, "w") as f:
            f.write("Timestamp,Detected Object,Emotion\n")

        self.frame_label = tk.Label(panel, bg="#000")
        self.frame_label.pack(pady=10, fill=tk.BOTH, expand=False)

        btnframe = ttk.Frame(panel)
        btnframe.pack(pady=10)
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        self.start_btn = ttk.Button(btnframe, text="▶ Start", command=self.start_detection)
        self.stop_btn = ttk.Button(btnframe, text="⏸ Stop", command=self.stop_detection)
        self.analysis_btn = ttk.Button(btnframe, text="📊 Analysis", command=self.show_analysis_popup)
        self.exit_btn = ttk.Button(btnframe, text="❌ Exit", command=self.exit_app)
        for i, btn in enumerate([self.start_btn, self.stop_btn, self.analysis_btn, self.exit_btn]):
            btn.grid(row=0, column=i, padx=8)

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def exit_app(self):
        self.stop_detection()
        self.root.destroy()

    def update_frame(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                results = self.model(frame)[0]
                detected_object = "other"

                for box in results.boxes:
                    cls = int(box.cls[0])
                    name = self.model.names[cls]
                    if name in ["cell phone", "book"]:
                        detected_object = name
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

                faces = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                for face in faces:
                    x, y, w, h = face['region'].values()
                    emotion = face['dominant_emotion']
                    confidence = face['emotion'][emotion]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.csv_file, "a") as f:
                    f.write(f"{ts},{detected_object},{emotion}\n")

            except Exception as e:
                print("Detection error:", e)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.frame_label.imgtk = img
            self.frame_label.config(image=img)

    def show_analysis_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("📊 Emotion Analysis")
        popup.geometry("1000x600")
        popup.configure(bg="#1a1a1a")

        def clear_data():
            with open(self.csv_file, "w") as f:
                f.write("Timestamp,Detected Object,Emotion\n")
            popup.destroy()

        try:
            df = pd.read_csv(self.csv_file)
            grp = df.groupby(['Detected Object','Emotion']).size().unstack().fillna(0)
            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            grp.plot(kind='bar', stacked=True, ax=ax, colormap="tab20")
            ax.set_title("Emotion Distribution per Object", color="#eee", fontsize=13)
            ax.set_ylabel("Frames", color="#ccc")
            ax.tick_params(colors="#aaa")
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            tk.Label(popup, text=f"Error: {e}", bg="#1a1a1a", fg="red").pack()

        btn_frame = tk.Frame(popup, bg="#1a1a1a")
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="🔄 Clear Data", command=clear_data, font=("Segoe UI", 11), bg="#444", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="❌ Close", command=popup.destroy, font=("Segoe UI", 11), bg="#800", fg="white").pack(side=tk.LEFT, padx=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionObjectApp(root)
    root.mainloop()
