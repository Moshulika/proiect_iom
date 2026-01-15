import os
import threading
import subprocess
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
        # best-effort
        try:
            subprocess.Popen(["explorer", path])
        except Exception:
            pass

def _generate_fallback_icon(size: tuple, text: str = "♪") -> Image.Image:
    """Return a generated (PIL) icon with a simple glyph — fallback for non-Windows or failure."""
    w, h = size
    img = Image.new("RGBA", (w, h), (40, 40, 40, 255))
    draw = ImageDraw.Draw(img)
    # circle
    r = min(w, h) // 2 - 4
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(70, 130, 180, 255))
    # draw glyph text
    try:
        # try a system font
        fnt = ImageFont.truetype("arial.ttf", int(r * 1.2))
    except Exception:
        fnt = ImageFont.load_default()
    tw, th = draw.textsize(text, font=fnt)
    draw.text((cx - tw / 2, cy - th / 2), text, font=fnt, fill=(255, 255, 255, 255))
    return img


# --- Main App ---
class NoiseReducerApp(ctk.CTk):
    def __init__(self):

        super().__init__()
        self.title("Noise Reducer — Modern")
        self.geometry("1050x600")
        self.minsize(width=1050, height=600)

        # State
        self.file_queue = []               # list of file paths (order)
        self.file_cards = {}               # file_path -> card widgets and state
        self.recent_folders = []           # recent folder paths
        self.recent_files = []
        self.cancel_flags = {}             # file_path -> boolean flag (set True to cancel this file)
        self.threads = {}                  # file_path -> thread object
        self.saved_outputs = []            # list of saved output files (for cleanup on stop-all)
        self.stop_all_flag = False         # when set, stop everything
        self.updating_completed_ui = False # flag to prevent concurrent UI updates

        # UI layout: 2 columns: left sidebar, main
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left sidebar: recent folders
        sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(1, weight=1)

        lbl = ctk.CTkLabel(sidebar, text="Recent Folders", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.grid(row=0, column=0, pady=(18, 4), padx=12, sticky="w")

        self.recent_scroll = ctk.CTkScrollableFrame(sidebar, width=240)
        self.recent_scroll.grid(row=1, column=0, padx=8, pady=8, sticky="nswe")

        recent_files = ctk.CTkFrame(sidebar, width=260, corner_radius=0)
        recent_files.grid(row=2, column=0, sticky="nswe")
        recent_files.grid_rowconfigure(2, weight=1)

        lbl = ctk.CTkLabel(recent_files, text="Completed files", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.grid(row=0, column=0, pady=(18, 4), padx=12, sticky="w")

        self.recent_files_scroll = ctk.CTkScrollableFrame(recent_files, width=240)
        self.recent_files_scroll.grid(row=2, column=0, padx=8, pady=8, sticky="nswe")

        # Main area
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

        self.remove_selected_btn = ctk.CTkButton(top_controls, text="Remove Selected", command=self.remove_selected, height=36,
                                                 fg_color="#A63232", hover_color="#8A2727")
        self.remove_selected_btn.grid(row=0, column=1, padx=8)

        # Spacer column
        spacer = ctk.CTkLabel(top_controls, text="")
        spacer.grid(row=0, column=2, sticky="ew")

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

        # Scrollable area for file cards (rounded cards)
        self.cards_frame = ctk.CTkScrollableFrame(main, label_text="Files in Queue (cards)", height=520)
        self.cards_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        self.cards_frame.grid_columnconfigure(0, weight=1)

        # Preload icons
        # Folder icon, audio icon, close icon (system attempts); size chosen to fit card layout
        self.icon_size = (36, 36)

        try:

            # try several resources for audio icons: extension .wav, .mp3, or imageres.dll indices
            self.icon_folder_img = Image.open("folder-icon.png")
            # Try extension icons for audio
            self.icon_audio_img = Image.open("soundfile-icon.png")
            self.icon_close_img = Image.open("cancel.png")  # fallback small

        except Exception:
            self.icon_folder_img = _generate_fallback_icon(self.icon_size, "📁")
            self.icon_audio_img = _generate_fallback_icon(self.icon_size, "♪")
            self.icon_close_img = _generate_fallback_icon((20, 20), "x")

        # Convert to CTkImage
        try:

            self.icon_folder = ctk.CTkImage(self.icon_folder_img, size=(36, 36))
            self.icon_audio = ctk.CTkImage(self.icon_audio_img, size=(36, 36))
            close_img_small = self.icon_close_img.resize((18, 18), Image.LANCZOS)
            self.icon_close = ctk.CTkImage(close_img_small, size=(18, 18))

        except Exception:
            # fallback to no-image use
            self.icon_folder = None
            self.icon_audio = None
            self.icon_close = None

        # Track selected card (for Remove Selected)
        self.selected_card = None

        # Add some instructions label
        instr = ctk.CTkLabel(main, text="Add files, then press Start. Click ❌ on a card to remove/cancel that file.",
                             anchor="w", fg_color="transparent")
        instr.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 0))

        # initialize recent folders if any (empty right now)
        self.update_recent_folders_ui()
        self.update_completed_files_ui()

    # -----------------------------
    # UI helpers
    # -----------------------------
    def update_recent_folders_ui(self):
        # clear
        for w in self.recent_scroll.winfo_children():
            w.grid_forget()
            w.destroy()

        for folder in list(self.recent_folders):

            row = ctk.CTkFrame(self.recent_scroll, fg_color="transparent")
            row.pack(fill="x", pady=4, padx=6)

            if self.icon_folder:

                icon_lbl = ctk.CTkLabel(row, image=self.icon_folder, text="")
                icon_lbl.pack(side="left", padx=(2, 8))

            btn = ctk.CTkButton(row, text=os.path.basename(folder) or folder, fg_color="transparent",
                                hover_color="#2C2C2C", anchor="w", command=lambda f=folder: open_in_explorer(f))
            btn.pack(side="left", fill="x", expand=True)

        # -----------------------------
        # UI helpers
        # -----------------------------

    def update_completed_files_ui(self):
        # Prevent concurrent updates
        if self.updating_completed_ui:
            return
        self.updating_completed_ui = True
        
        try:
            # Check if widget exists
            if not hasattr(self, 'recent_files_scroll') or not self.recent_files_scroll.winfo_exists():
                return
            
            # Clear existing widgets
            for w in self.recent_files_scroll.winfo_children():
                try:
                    w.grid_forget()
                    w.destroy()
                except:
                    pass

            # Create new widgets
            for folder in list(self.recent_files):
                try:
                    row = ctk.CTkFrame(self.recent_files_scroll, fg_color="transparent")
                    row.pack(fill="x", pady=4, padx=6)

                    if self.icon_audio:
                        icon_lbl = ctk.CTkLabel(row, image=self.icon_audio, text="")
                        icon_lbl.pack(side="left", padx=(2, 8))

                    btn = ctk.CTkButton(row, text=os.path.basename(folder) or folder, fg_color="transparent",
                                        hover_color="#2C2C2C", anchor="w", command=lambda f=folder: open_in_explorer(f))
                    btn.pack(side="left", fill="x", expand=True)
                except Exception as e:
                    print(f"Error creating UI element for {folder}: {e}")
                    continue
        except Exception as e:
            print(f"Error in update_completed_files_ui: {e}")
        finally:
            self.updating_completed_ui = False

    def add_completed_files(self, path):
        if path not in self.recent_files:
            print("Path is not in recent files, adding it")
            self.recent_files.insert(0, path)
            self.recent_files = self.recent_files[:20]
            # Update UI on main thread
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
                    # cap recent folders to 20
                    self.recent_folders = self.recent_folders[:20]
                    self.update_recent_folders_ui()
                added += 1
        if added == 0 and len(files) > 0:
            messagebox.showinfo("Files", "Selected files were already in the queue.")

    def _add_file_card(self, file_path: str):
        """Create a rounded card representing the file."""
        card = ctk.CTkFrame(self.cards_frame, corner_radius=12, fg_color="#252525")
        card.pack(fill="x", pady=8, padx=12)

        # inner layout: left icon, center content (filename + progress), right remove button
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.pack(side="left", padx=(10, 8), pady=10)

        if self.icon_audio:
            icon_lbl = ctk.CTkLabel(left, image=self.icon_audio, text="")
        else:
            icon_lbl = ctk.CTkLabel(left, text="♪", width=36)
        icon_lbl.pack()

        center = ctk.CTkFrame(card, fg_color="transparent")

        center.pack(side="left", fill="both", expand=True, padx=(6, 8), pady=8)

        name_lbl = ctk.CTkLabel(center, text=os.path.basename(file_path), anchor="w",
                                font=ctk.CTkFont(size=14, weight="bold"))
        name_lbl.pack(fill="x", anchor="w")


        subt = ctk.CTkLabel(center, text=file_path, anchor="w", font=ctk.CTkFont(size=10))
        subt.pack(fill="x", anchor="w", pady=(2, 6))

        # right remove button
        right = ctk.CTkFrame(card, fg_color="transparent")
        right.pack(side="right", padx=(10, 10), pady=10)

        if self.icon_close:
            rem = ctk.CTkButton(right, image=self.icon_close, text="", width=28, height=28,
                               fg_color="#A63232", hover_color="#8A2727",
                               command=lambda p=file_path: self._on_card_remove_clicked(p))
        else:
            rem = ctk.CTkButton(right, text="X", width=28, height=28,
                               fg_color="#A63232", hover_color="#8A2727",
                               command=lambda p=file_path: self._on_card_remove_clicked(p))
        rem.pack()

        # store card state
        self.file_cards[file_path] = {
            "card": card,
            "center": center,
            "name_lbl": name_lbl,
            "sub_lbl": subt,
            "progress": None,
            "remove_btn": rem,
            "status": "pending",   # pending, processing, done, cancelled
            "thread": None
        }

        # clicking the card selects it (for Remove Selected)
        def on_card_click(ev, p=file_path):
            self._select_card(p)

        card.bind("<Button-1>", on_card_click)
        name_lbl.bind("<Button-1>", on_card_click)
        subt.bind("<Button-1>", on_card_click)

    def _select_card(self, file_path: str):
        # visually mark selection
        if self.selected_card and self.selected_card in self.file_cards:
            # reset previous
            prev_card = self.file_cards[self.selected_card]["card"]
            prev_card.configure(fg_color="#252525")
        self.selected_card = file_path
        this_card = self.file_cards[file_path]["card"]
        this_card.configure(fg_color="#2C2C2C")

    def remove_selected(self):
        """Remove the currently selected card (only if pending)."""
        if not self.selected_card:
            messagebox.showinfo("Remove Selected", "Click a card to select it, then press Remove Selected.")
            return
        path = self.selected_card
        state = self.file_cards[path]["status"]
        if state == "pending":
            self._remove_card(path)
        else:
            # If it's processing, treat it as cancelling only that file
            self._cancel_file(path)
        self.selected_card = None

    def _on_card_remove_clicked(self, file_path: str):
        """If pending: remove from queue. If processing: cancel that file only."""
        state = self.file_cards[file_path]["status"]
        if state == "pending":
            self._remove_card(file_path)
        elif state == "processing":
            # cancel just this file (no confirmation)
            self._cancel_file(file_path)
        else:
            # done/cancelled -> remove the card
            self._remove_card(file_path)

    def _remove_card(self, file_path: str):
        """Remove card UI and remove from queue/state. Only for pending or done items."""
        if file_path in self.file_cards:
            card = self.file_cards[file_path]["card"]
            card.grid_forget()
            card.destroy()
            del self.file_cards[file_path]
        if file_path in self.file_queue:
            self.file_queue.remove(file_path)
        # also clear flags / threads
        self.cancel_flags.pop(file_path, None)
        th = self.threads.pop(file_path, None)
        # Note: if thread was running, we don't forcibly kill it here; we set cancel flags elsewhere.

    # -----------------------------
    # Processing & threading
    # -----------------------------
    def start_all(self):

        if not self.file_cards:
            messagebox.showwarning("No files", "Add some audio files first.")
            return
        # Reset global stop flag and saved outputs
        self.stop_all_flag = False

        # For each pending file, start a thread that processes it and updates its own progress
        for path, data in list(self.file_cards.items()):
            if data["status"] == "pending":

                prog = ctk.CTkProgressBar(data["center"], width=320)
                prog.set(0.0)
                prog.pack(fill="x", pady=(2, 4))

                # mark and start thread
                self.cancel_flags[path] = False
                data["progress"] = prog
                data["status"] = "processing"
                t = threading.Thread(target=self._process_single_file, args=(path,), daemon=True)
                self.threads[path] = t
                data["thread"] = t
                t.start()

        # disable start button while processing
        self.start_btn.configure(state="disabled")

    def _process_single_file(self, file_path: str):
        """Process one file in chunks, update progress bar. Respect cancel_flags and stop_all_flag."""
        data = self.file_cards.get(file_path)
        if not data:
            return
        prog_widget = data["progress"]
        # Chunking parameters
        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # mark done/error
            self.after(0, lambda: data["progress"].set(0.0))
            data["status"] = "cancelled"
            return

        length = len(y)
        # number of chunks determines progress smoothness
        n_chunks = max(8, min(50, int(length / 200000) + 8))  # heuristic
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
            # process chunk with noisereduce - small chunks still okay
            try:
                reduced_chunk = nr.reduce_noise(y=chunk, sr=sr)
            except Exception as e:
                # If chunk processing fails, fallback to copying chunk as-is and continue
                reduced_chunk = chunk
                print("Chunk reduce failed:", e)
            reduced_parts.append(reduced_chunk)
            # update progress visually
            p = (i + 1) / n_chunks
            self.after(0, lambda val=p, w=prog_widget: w.set(val))

        # If cancelled, do cleanup
        if cancelled:
            # if partially saved file exists, remove it
            out_path = self._get_output_path(file_path)
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            # mark status
            data["status"] = "cancelled"
            self.after(0, lambda: prog_widget.set(0.0))
            return

        # Merge reduced parts and write
        try:
            reduced_full = np.concatenate(reduced_parts) if reduced_parts else np.array([], dtype=np.float32)
            out_path = self._get_output_path(file_path)
            sf.write(out_path, reduced_full, sr)
            # track saved output for potential global cleanup
            self.saved_outputs.append(out_path)
            data["status"] = "done"
            self.after(0, lambda: prog_widget.set(1.0))

            print(f"Finished processing {file_path}")
            self.add_completed_files(file_path)
            print(f"Removing {file_path} card")
            self._remove_card(file_path)

        except Exception as e:
            print("Failed to write output for", file_path, e)
            data["status"] = "cancelled"
            self.after(0, lambda: prog_widget.set(0.0))
        finally:
            # If all threads finished, re-enable start button (check global)
            self._maybe_enable_start()

    def _maybe_enable_start(self):
        # Called from worker threads via after; enable start when no processing remains
        def check():
            for v in self.file_cards.values():
                if v["status"] == "processing":
                    return
            self.start_btn.configure(state="normal")
        self.after(0, check)

    def _get_output_path(self, file_path: str) -> str:
        folder, name = os.path.split(file_path)
        base, ext = os.path.splitext(name)
        return os.path.join(folder, f"{base}_out{ext}")

    def _cancel_file(self, file_path: str):
        """Cancel processing of a single file in-progress (no confirmation)."""
        # mark the cancel flag; worker thread will detect and cleanup
        self.cancel_flags[file_path] = True
        # delete any partial output if exists (thread also removes after detection)
        out_path = self._get_output_path(file_path)
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        # set UI state to cancelled and reset progress
        if file_path in self.file_cards:
            self.file_cards[file_path]["status"] = "cancelled"
            self.after(0, lambda: self.file_cards[file_path]["progress"].set(0.0))

    def stop_all(self):
        """Prompt for confirmation, then stop everything and remove created outputs."""
        if not any(v["status"] == "processing" for v in self.file_cards.values()):
            # nothing running; just clear pending if user wants
            if messagebox.askyesno("Stop All", "No processing currently running. Do you want to clear the queue?"):
                # clear all cards
                for p in list(self.file_cards.keys()):
                    self._remove_card(p)
            return

        if not messagebox.askyesno("Stop All", "Are you sure you want to stop all processing? Any processed files will be deleted."):
            return

        # Set global flag to stop all
        self.stop_all_flag = True
        # Set individual cancel flags as well
        for k in list(self.cancel_flags.keys()):
            self.cancel_flags[k] = True

        # Wait briefly for threads to notice
        # Use after to run cleanup shortly after
        self.after(700, self._cleanup_after_stop)

    def _cleanup_after_stop(self):
        # delete any outputs created
        for out in list(self.saved_outputs):
            if os.path.exists(out):
                try:
                    os.remove(out)
                except Exception:
                    pass
        self.saved_outputs.clear()
        # reset UI states and progress bars
        for p, data in list(self.file_cards.items()):
            data["status"] = "cancelled" if data["status"] == "processing" else data["status"]
            self.after(0, lambda pw=data["progress"]: pw.set(0.0))
        # re-enable start
        self.start_btn.configure(state="normal")
        self.stop_all_flag = False
        # notify user
        messagebox.showinfo("Stopped", "All processing stopped and outputs removed.")

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _get_pretty_name(path: str) -> str:
        return os.path.basename(path)

# Run the app
if __name__ == "__main__":
    app = NoiseReducerApp()
    app.mainloop()
