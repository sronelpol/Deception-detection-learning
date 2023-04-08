import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import filedialog, ttk, messagebox
from tkinter.ttk import Progressbar

from test_your_video.predict_single_video import run_single_video_process

COLOR = "deep sky blue"


def process_video(video_file):
    # This function represents the background process you want to run
    # Replace this with your actual function that processes the video file
    print("Processing video:", video_file)
    result = run_single_video_process(video_file)  # Simulate processing the video for 5 seconds
    print("Video processing complete")
    return "Detected as " + result


class VideoProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("300x100")
        self.root.iconbitmap(
            str(Path(__file__).parent.parent / "resources" / "gui" / "deception-detector-logo.ico"), False
        )
        self.root.config(bg=COLOR)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("custom.Horizontal.TProgressbar", troughcolor="#AED6F1", background=COLOR)
        self.root.title("Deception Detector")

        # Create a button to select the video file
        self.select_button = tk.Button(self.root, text="Select Video", command=self.select_video)
        self.select_button.pack(pady=10)

        # Create a progress bar to show the progress of the video processing
        self.progress = Progressbar(
            self.root, style="custom.Horizontal.TProgressbar", orient=tk.HORIZONTAL, length=200, mode="indeterminate"
        )

        # Create a label to show the spinner
        self.spinner_label = tk.Label(self.root, bg=COLOR, text="Processing video, please wait...")

        # Create a label to show the result
        self.result_label = tk.Label(self.root, bg=COLOR)

    def select_video(self):
        # Open a file dialog to select a video file
        self.video_file = filedialog.askopenfilename(title="Select Video", filetypes=(("Video Files", "*.mp4"),))
        if not self.video_file:
            messagebox.showerror("Error", "No video file selected")
            return
        # Hide the select button and show the progress bar and spinner
        self.result_label.pack_forget()
        self.select_button.pack_forget()
        self.spinner_label.pack(pady=10)
        self.progress.pack(pady=10)
        self.progress.start()

        # Start a new thread to process the video file
        self.process_thread = Thread(target=self.process_video, args=(self.video_file,))
        self.process_thread.start()

        # Check the status of the process thread every 100ms
        self.check_thread()

    def process_video(self, video_file):
        result = process_video(video_file)
        # Update the result label with the result
        self.result_label.config(text=result)

    def check_thread(self):
        if self.process_thread.is_alive():
            # If the thread is still running, check again after 100ms
            self.root.after(100, self.check_thread)
        else:
            # If the thread has finished, hide the spinner and progress bar and show the select button and result label
            self.progress.stop()
            self.progress.pack_forget()
            self.spinner_label.pack_forget()
            self.select_button.pack(pady=10)
            self.result_label.pack(pady=10)

    def start(self):
        self.root.mainloop()


# Create a VideoProcessor instance and start the program
processor = VideoProcessor()
processor.start()
