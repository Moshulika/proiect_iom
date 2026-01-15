import os
import threading
import subprocess
import scipy.signal
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont
import noisereduce as nr
import soundfile as sf
import librosa
import numpy as np

# --- Configuration ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def open_in_explorer(path: str):
    """Open path in Windows Explorer (cross-platform fallback)."""
    try:
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        try:
            subprocess.Popen(["explorer", path])
        except Exception:
            pass


def _generate_fallback_icon(size: tuple, text: str = "♪") -> Image.Image:
    """Return a generated (PIL) icon. Updated to work with new Pillow versions."""
    w, h = size
    img = Image.new("RGBA", (w, h), (40, 40, 40, 255))
    draw = ImageDraw.Draw(img)

    # circle
    r = min(w, h) // 2 - 4
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(70, 130, 180, 255))

    # draw glyph text
    try:
        fnt = ImageFont.truetype("arial.ttf", int(r * 1.2))
    except Exception:
        fnt = ImageFont.load_default()

    # FIX: textsize is deprecated in Pillow 10+, use textbbox
    try:
        bbox = draw.textbbox((0, 0), text, font=fnt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        tw, th = draw.textsize(text, font=fnt)

    draw.text((cx - tw / 2, cy - th / 2 - (th / 4)), text, font=fnt, fill=(255, 255, 255, 255))
    return img


# --- Main App ---
class NoiseReducerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Noise Reducer — Modern")
        self.geometry("1050x600")
        self.minsize(width=1050, height=600)

        # State
        self.file_queue = []
        self.file_cards = {}
        self.recent_folders = []
        self.recent_files = []
        self.cancel_flags = {}
        self.threads = {}
        self.saved_outputs = []
        self.stop_all_flag = False
        self.updating_completed_ui = False

        # UI layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(1, weight=1)  # Recent folders scroll
        sidebar.grid_rowconfigure(3, weight=1)  # Completed files scroll

        # Recent Folders Section
        lbl = ctk.CTkLabel(sidebar, text="Recent Folders", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.grid(row=0, column=0, pady=(18, 4), padx=12, sticky="w")

        self.recent_scroll = ctk.CTkScrollableFrame(sidebar, width=240, height=200)
        self.recent_scroll.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

        # Completed Files Section
        # We use a separator or just spacing
        lbl_completed = ctk.CTkLabel(sidebar, text="Completed Files", font=ctk.CTkFont(size=18, weight="bold"))
        lbl_completed.grid(row=2, column=0, pady=(18, 4), padx=12, sticky="w")

        self.recent_files_scroll = ctk.CTkScrollableFrame(sidebar, width=240)
        self.recent_files_scroll.grid(row=3, column=0, padx=8, pady=8, sticky="nsew")

        # --- Main Area ---
        main = ctk.CTkFrame(self, corner_radius=12)
        main.grid(row=0, column=1, sticky="nswe", padx=18, pady=12)
        main.grid_rowconfigure(2, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Top controls
        top_controls = ctk.CTkFrame(main, fg_color="transparent")
        top_controls.grid(row=0, column=0, sticky="ew", pady=(10, 6))
        top_controls.grid_columnconfigure(2, weight=1)

        self.add_btn = ctk.CTkButton(top_controls, text="Add Audio Files", command=self.add_files, height=36)
        self.add_btn.grid(row=0, column=0, padx=8)

        self.remove_selected_btn = ctk.CTkButton(top_controls, text="Remove Selected", command=self.remove_selected,
                                                 height=36,
                                                 fg_color="#A63232", hover_color="#8A2727")
        self.remove_selected_btn.grid(row=0, column=1, padx=8)

        # Bottom controls (start/stop)
        bottom_controls = ctk.CTkFrame(main, fg_color="transparent")
        bottom_controls.grid(row=3, column=0, sticky="ew", pady=(6, 10))
        bottom_controls.grid_columnconfigure(0, weight=1)

        self.start_btn = ctk.CTkButton(bottom_controls, text="Start Noise Reduction", command=self.start_all, height=44,
                                       fg_color="#1f6aa5", hover_color="#155b8a")
        self.start_btn.grid(row=0, column=0, padx=8, sticky="w")

        self.stop_btn = ctk.CTkButton(bottom_controls, text="Stop All", command=self.stop_all, height=44,
                                      fg_color="#A63232", hover_color="#8A2727")
        self.stop_btn.grid(row=0, column=1, padx=8, sticky="e")

        # Scrollable area for file cards
        self.cards_frame = ctk.CTkScrollableFrame(main, label_text="Files in Queue", height=520)
        self.cards_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        self.cards_frame.grid_columnconfigure(0, weight=1)

        # Assets
        self.icon_size = (36, 36)
        try:
            self.icon_folder_img = Image.open("folder-icon.png")  # User must provide these or they fallback
            self.icon_audio_img = Image.open("soundfile-icon.png")
            self.icon_close_img = Image.open("cancel.png")

            self.icon_folder = ctk.CTkImage(self.icon_folder_img, size=(36, 36))
            self.icon_audio = ctk.CTkImage(self.icon_audio_img, size=(36, 36))
            self.icon_close = ctk.CTkImage(self.icon_close_img, size=(18, 18))
        except Exception:
            self.icon_folder = None
            self.icon_audio = None
            self.icon_close = None

        self.selected_card = None

        # Instructions
        instr = ctk.CTkLabel(main, text="Add files, then press Start. Click ❌ on a card to remove/cancel that file.",
                             anchor="w", fg_color="transparent")
        instr.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 0))

        self.update_recent_folders_ui()
        self.update_completed_files_ui()

    # -----------------------------
    # UI helpers
    # -----------------------------
    def update_recent_folders_ui(self):
        for w in self.recent_scroll.winfo_children():
            w.destroy()

        for folder in list(self.recent_folders):
            row = ctk.CTkFrame(self.recent_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4, padx=6)

            # Simple button
            btn = ctk.CTkButton(row, text=os.path.basename(folder) or folder, fg_color="transparent",
                                hover_color="#2C2C2C", anchor="w",
                                command=lambda f=folder: open_in_explorer(f))
            btn.pack(side="left", fill="x", expand=True)

    def update_completed_files_ui(self):
        # Safety check for widget existence
        if not self.recent_files_scroll.winfo_exists():
            return

        # Clear existing widgets
        for w in self.recent_files_scroll.winfo_children():
            w.destroy()

        # Create new widgets
        for folder in list(self.recent_files):
            try:
                row = ctk.CTkFrame(self.recent_files_scroll, fg_color="transparent")
                row.pack(fill="x", pady=4, padx=6)

                btn = ctk.CTkButton(row, text=os.path.basename(folder) or folder, fg_color="transparent",
                                    hover_color="#2C2C2C", anchor="w",
                                    command=lambda f=folder: open_in_explorer(f))
                btn.pack(side="left", fill="x", expand=True)
            except Exception as e:
                print(f"Error creating UI for completed file: {e}")

    def add_completed_files(self, path):
        if path not in self.recent_files:
            self.recent_files.insert(0, path)
            self.recent_files = self.recent_files[:20]
            # UI Update must happen on main thread
            self.after(0, self.update_completed_files_ui)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select audio files",
                                            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a")])
        added = 0
        for f in files:
            if f not in self.file_queue:
                self.file_queue.append(f)
                self._add_file_card(f)
                folder = os.path.dirname(f)
                if folder not in self.recent_folders:
                    self.recent_folders.insert(0, folder)
                    self.recent_folders = self.recent_folders[:20]
                    self.update_recent_folders_ui()
                added += 1

        if added == 0 and len(files) > 0:
            # Only show if not adding anything new
            pass

    def _add_file_card(self, file_path: str):
        card = ctk.CTkFrame(self.cards_frame, corner_radius=12, fg_color="#252525")
        card.pack(fill="x", pady=8, padx=12)

        # Left: Icon
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.pack(side="left", padx=(10, 8), pady=10)

        if self.icon_audio:
            icon_lbl = ctk.CTkLabel(left, image=self.icon_audio, text="")
        else:
            # Generate fallback if image failed
            fb_img = ctk.CTkImage(_generate_fallback_icon((36, 36)), size=(36, 36))
            icon_lbl = ctk.CTkLabel(left, image=fb_img, text="")
        icon_lbl.pack()

        # Center: Info
        center = ctk.CTkFrame(card, fg_color="transparent")
        center.pack(side="left", fill="both", expand=True, padx=(6, 8), pady=8)

        name_lbl = ctk.CTkLabel(center, text=os.path.basename(file_path), anchor="w",
                                font=ctk.CTkFont(size=14, weight="bold"))
        name_lbl.pack(fill="x", anchor="w")

        subt = ctk.CTkLabel(center, text=file_path, anchor="w", font=ctk.CTkFont(size=10))
        subt.pack(fill="x", anchor="w", pady=(2, 6))

        # Right: Remove
        right = ctk.CTkFrame(card, fg_color="transparent")
        right.pack(side="right", padx=(10, 10), pady=10)

        rem = ctk.CTkButton(right, text="X", width=28, height=28,
                            fg_color="#A63232", hover_color="#8A2727",
                            command=lambda p=file_path: self._on_card_remove_clicked(p))
        if self.icon_close:
            rem.configure(image=self.icon_close, text="")
        rem.pack()

        self.file_cards[file_path] = {
            "card": card,
            "center": center,
            "name_lbl": name_lbl,
            "sub_lbl": subt,
            "progress": None,
            "remove_btn": rem,
            "status": "pending",
            "thread": None
        }

        def on_card_click(ev, p=file_path):
            self._select_card(p)

        card.bind("<Button-1>", on_card_click)
        name_lbl.bind("<Button-1>", on_card_click)
        subt.bind("<Button-1>", on_card_click)

    def _select_card(self, file_path: str):
        if self.selected_card and self.selected_card in self.file_cards:
            try:
                prev_card = self.file_cards[self.selected_card]["card"]
                prev_card.configure(fg_color="#252525")
            except Exception:
                pass  # Widget might be dead

        self.selected_card = file_path
        if file_path in self.file_cards:
            self.file_cards[file_path]["card"].configure(fg_color="#2C2C2C")

    def remove_selected(self):
        if not self.selected_card:
            messagebox.showinfo("Remove", "Select a file first.")
            return
        self._on_card_remove_clicked(self.selected_card)
        self.selected_card = None

    def _on_card_remove_clicked(self, file_path: str):
        if file_path not in self.file_cards:
            return

        state = self.file_cards[file_path]["status"]
        if state == "pending" or state == "done" or state == "cancelled":
            self._remove_card(file_path)
        elif state == "processing":
            self._cancel_file(file_path)

    def _remove_card(self, file_path: str):
        if file_path in self.file_cards:
            card = self.file_cards[file_path]["card"]

            # 1. Hide immediately
            card.pack_forget()

            # 2. Schedule destruction safely on main thread
            # This prevents the "Resize Storm" / TclError
            self.after(100, lambda c=card: c.destroy())

            del self.file_cards[file_path]

        if file_path in self.file_queue:
            self.file_queue.remove(file_path)

        self.cancel_flags.pop(file_path, None)
        self.threads.pop(file_path, None)

    # -----------------------------
    # Processing
    # -----------------------------
    def start_all(self):
        if not self.file_cards:
            messagebox.showwarning("No files", "Add some audio files first.")
            return

        self.stop_all_flag = False
        started_any = False

        for path, data in list(self.file_cards.items()):
            if data["status"] == "pending":
                # Create progress bar
                prog = ctk.CTkProgressBar(data["center"], width=320)
                prog.set(0.0)
                prog.pack(fill="x", pady=(2, 4))

                self.cancel_flags[path] = False
                data["progress"] = prog
                data["status"] = "processing"

                t = threading.Thread(target=self._process_single_file, args=(path,), daemon=True)
                self.threads[path] = t
                data["thread"] = t
                t.start()
                started_any = True

        if started_any:
            self.start_btn.configure(state="disabled")

    def _process_single_file(self, file_path: str):

        def update_prog(val):
            if file_path in self.file_cards:
                w = self.file_cards[file_path].get("progress")
                if w and w.winfo_exists():
                    w.set(val)

        try:
            # 1. Load Audio
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Load error: {e}")
            self.after(0, lambda: self._handle_processing_error(file_path))
            return

        # CAPTURE NOISE PROFILE
        # We take a slightly longer sample (0.8s) for better accuracy
        if len(y) > sr:
            noise_part = y[0:int(0.8 * sr)]
        else:
            noise_part = y

        stationary_args = {
            "n_std_thresh_stationary": 1.3,
            "prop_decrease": 1,
            "n_fft": 8192,
            "stationary": True
        }

        length = len(y)
        # Larger chunks reduce "seams" in the audio
        n_chunks = max(8, min(50, int(length / 300000) + 8))
        chunk_size = length // n_chunks
        reduced_parts = []
        cancelled = False

        for i in range(n_chunks):
            if self.stop_all_flag or self.cancel_flags.get(file_path, False):
                cancelled = True
                break

            start = i * chunk_size
            end = length if i == n_chunks - 1 else (i + 1) * chunk_size
            chunk = y[start:end]

            try:
                # Apply reduction with profile
                reduced_chunk = nr.reduce_noise(y=chunk, sr=sr, y_noise=noise_part, **stationary_args)
                reduced_parts.append(reduced_chunk)
            except Exception as e:
                reduced_parts.append(chunk)

            p = (i + 1) / n_chunks
            self.after(0, lambda v=p: update_prog(v))

        if cancelled:
            self.after(0, lambda: self._handle_cancellation(file_path))
            return

        try:
            reduced_full = np.concatenate(reduced_parts) if reduced_parts else np.array([], dtype=np.float32)

            nyquist = 0.5 * sr
            cutoff = 10000 / nyquist

            if cutoff < 1.0:
                b, a = scipy.signal.butter(8, cutoff, btype='low', analog=False)
                reduced_full = scipy.signal.filtfilt(b, a, reduced_full)

            # Save
            out_path = self._get_output_path(file_path)
            sf.write(out_path, reduced_full, sr)
            self.saved_outputs.append(out_path)

            self.after(0, lambda: self._handle_success(file_path, out_path))

        except Exception as e:
            print(f"Write error: {e}")
            self.after(0, lambda: self._handle_processing_error(file_path))
        finally:
            self.after(0, self._maybe_enable_start)

    # --- Thread-Safe Completion Handlers ---

    def _handle_success(self, file_path, out_path):
        """Called by thread on main loop when done."""
        print(f"Done: {file_path}")
        self.add_completed_files(out_path)
        # Remove card SAFELY using our soft delete method
        self._remove_card(file_path)

    def _handle_cancellation(self, file_path):
        if file_path in self.file_cards:
            self.file_cards[file_path]["status"] = "cancelled"
            if self.file_cards[file_path]["progress"]:
                self.file_cards[file_path]["progress"].set(0.0)

        # cleanup partial file
        out_path = self._get_output_path(file_path)
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except:
                pass

    def _handle_processing_error(self, file_path):
        if file_path in self.file_cards:
            self.file_cards[file_path]["status"] = "error"
            if self.file_cards[file_path]["progress"]:
                self.file_cards[file_path]["progress"].set(0.0)
        messagebox.showerror("Error", f"Failed to process {os.path.basename(file_path)}")

    def _maybe_enable_start(self):
        # Check if any threads are still running
        still_running = False
        for v in self.file_cards.values():
            if v["status"] == "processing":
                still_running = True
                break
        if not still_running:
            self.start_btn.configure(state="normal")

    def _get_output_path(self, file_path: str) -> str:
        folder, name = os.path.split(file_path)
        base, ext = os.path.splitext(name)
        return os.path.join(folder, f"{base}_cleaned{ext}")

    def _cancel_file(self, file_path: str):
        self.cancel_flags[file_path] = True
        # Thread will pick this up in the loop and call _handle_cancellation

    def stop_all(self):
        processing = [v for v in self.file_cards.values() if v["status"] == "processing"]
        if not processing:
            if messagebox.askyesno("Clear", "Clear queue?"):
                for p in list(self.file_cards.keys()):
                    self._remove_card(p)
            return

        if not messagebox.askyesno("Stop", "Stop all processing?"):
            return

        self.stop_all_flag = True
        for k in self.cancel_flags:
            self.cancel_flags[k] = True

        self.after(1000, self._cleanup_after_stop)

    def _cleanup_after_stop(self):
        # Clean up files created during this session
        for out in self.saved_outputs:
            if os.path.exists(out):
                try:
                    os.remove(out)
                except:
                    pass
        self.saved_outputs.clear()

        # Reset UI
        for p, data in list(self.file_cards.items()):
            if data["status"] == "processing":
                data["status"] = "cancelled"
                if data["progress"]: data["progress"].set(0.0)

        self.start_btn.configure(state="normal")
        self.stop_all_flag = False
        messagebox.showinfo("Stopped", "Processing stopped.")


if __name__ == "__main__":
    app = NoiseReducerApp()
    app.mainloop()