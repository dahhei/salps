# Welcome to the Segmentation Analysis Labeling Program! (SALP)
# This script is designed to help you analyze and label images with advanced object detection capabilities.
#
# --- VERSION 2.4 UPGRADES (from v2.3) ---
# 1. Color Picker Tool: A new eyedropper tool to automatically set HSV thresholds by clicking on the image.
# 2. UI for Color Picker: New buttons to start, undo, and finish color selection.
# 3. State Management: The app now cleanly handles mutually exclusive modes (e.g., Drawing vs. Color Picking).

# ==============================================================================
#  IMPORTS
# ==============================================================================
import cv2
import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Toplevel, Menu, simpledialog, ttk, PanedWindow, StringVar
from PIL import Image, ImageTk
import json
import math
import sys

# ==============================================================================
#  HELPER FUNCTION FOR PACKAGING
# ==============================================================================
# This function helps to get the absolute path of resources, especially when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==============================================================================
#  Advanced Scale Calibration Window Class
# ==============================================================================
# This window allows users to select two points on an image and define a scale based on the distance between those points.
class ScaleCalibrationWindow(Toplevel):
    # Provides a GUI for advanced scale calibration by selecting two points on an image.
    def __init__(self, parent, image):
        super().__init__(parent)
        self.title("Advanced Scale Calibration")
        self.transient(parent)
        self.geometry("900x700")
        self.original_image = image
        self.h, self.w = image.shape[:2]
        self.zoom_level, self.view_x, self.view_y = 1.0, 0, 0
        self.p1, self.p2 = None, None
        self.scale_factor, self.scale_unit, self.is_confirmed = 1.0, "px", False
        self.drag_state, self.last_drag_x, self.last_drag_y = None, 0, 0
        self.min_zoom = 1.0
        self.loupe_active, self.last_mouse_pos, self.selected_point = True, (0, 0), None
        self.setup_widgets()
        self.bind_events()
        self.canvas.focus_set()
        self.after(100, self.reset_view)

    def setup_widgets(self):
        controls_frame = tk.Frame(self)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.info_label = tk.Label(controls_frame, text="Click a point, then use Arrow Keys to nudge. Loupe is in top-right.", font=("Helvetica", 10))
        self.info_label.pack(side=tk.LEFT, expand=True)
        tk.Button(controls_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="Clear Line", command=self.clear_line).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="Confirm Scale", command=self.confirm_scale, font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=5)
        self.canvas = tk.Canvas(self, bg="gray", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)

    def bind_events(self):
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel); self.canvas.bind("<Button-4>", self.on_mouse_wheel); self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_press); self.canvas.bind("<ButtonPress-3>", self.on_pan_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag); self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<KeyPress-Left>", self.on_key_press); self.canvas.bind("<KeyPress-Right>", self.on_key_press)
        self.canvas.bind("<KeyPress-Up>", self.on_key_press); self.canvas.bind("<KeyPress-Down>", self.on_key_press)

    def update_display(self):
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: self.after(50, self.update_display); return
        view_w, view_h = int(canvas_w / self.zoom_level), int(canvas_h / self.zoom_level)
        self.view_x = max(0, min(self.view_x, self.w - view_w))
        self.view_y = max(0, min(self.view_y, self.h - view_h))
        img_slice = self.original_image[self.view_y:self.view_y + view_h, self.view_x:self.view_x + view_w]
        display_img = cv2.resize(img_slice, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

        if self.p1:
            p1_c = self.image_to_canvas_coords(self.p1)
            highlight_color = (0, 255, 255) if self.selected_point == 'p1' else (255, 255, 255)
            cv2.circle(display_img, p1_c, 7, (0, 0, 255), -1); cv2.circle(display_img, p1_c, 8, highlight_color, 2)
        if self.p2:
            p1_c, p2_c = self.image_to_canvas_coords(self.p1), self.image_to_canvas_coords(self.p2)
            cv2.line(display_img, p1_c, p2_c, (0, 255, 0), 2)
            highlight_color = (0, 255, 255) if self.selected_point == 'p2' else (255, 255, 255)
            cv2.circle(display_img, p2_c, 7, (0, 255, 0), -1); cv2.circle(display_img, p2_c, 8, highlight_color, 2)
        
        if self.loupe_active: self.draw_loupe(display_img)

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_loupe(self, display_img):
        LOUPE_SIZE, LOUPE_RADIUS, LOUPE_ZOOM = 150, 75, 8
        img_coords = self.canvas_to_image_coords(self.last_mouse_pos)
        region_size = LOUPE_SIZE // LOUPE_ZOOM
        x1, y1 = max(0, img_coords[0] - region_size // 2), max(0, img_coords[1] - region_size // 2)
        x2, y2 = min(self.w, x1 + region_size), min(self.h, y1 + region_size)
        region = self.original_image[y1:y2, x1:x2]
        if region.size > 0:
            zoomed_region = cv2.resize(region, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)
            lx, ly = display_img.shape[1] - LOUPE_SIZE - 10, 10
            mask = np.zeros((LOUPE_SIZE, LOUPE_SIZE), dtype=np.uint8)
            cv2.circle(mask, (LOUPE_RADIUS, LOUPE_RADIUS), LOUPE_RADIUS, 255, -1)
            display_img[ly:ly+LOUPE_SIZE, lx:lx+LOUPE_SIZE] = cv2.bitwise_and(zoomed_region, zoomed_region, mask=mask)
            cv2.circle(display_img, (lx + LOUPE_RADIUS, ly + LOUPE_RADIUS), LOUPE_RADIUS, (255, 255, 255), 2)
            cv2.line(display_img, (lx + LOUPE_RADIUS, ly), (lx + LOUPE_RADIUS, ly + LOUPE_SIZE), (0, 0, 255), 1)
            cv2.line(display_img, (lx, ly + LOUPE_RADIUS), (lx + LOUPE_SIZE, ly + LOUPE_RADIUS), (0, 0, 255), 1)
    # Converts canvas coordinates to image coordinates and vice versa.
    def canvas_to_image_coords(self, c): return (int((c[0]/self.zoom_level)+self.view_x), int((c[1]/self.zoom_level)+self.view_y))
    def image_to_canvas_coords(self, i): return (int((i[0]-self.view_x)*self.zoom_level), int((i[1]-self.view_y)*self.zoom_level))
    # Handles mouse wheel events to zoom in or out of the image.
    def on_mouse_wheel(self, e):
        factor = 1.1 if (e.num == 4 or e.delta > 0) else 1 / 1.1
        self.zoom_level = max(self.min_zoom, self.zoom_level * factor)
        img_coords = self.canvas_to_image_coords((e.x, e.y))
        self.view_x = int(img_coords[0] - (e.x / self.zoom_level))
        self.view_y = int(img_coords[1] - (e.y / self.zoom_level))
        self.update_display()

    # Handles mouse clicks to select points for the scale line.
    def on_press(self, e):
        if self.p1 and math.dist(self.image_to_canvas_coords(self.p1),(e.x,e.y))<10:
            self.drag_state='p1'; self.selected_point='p1'; self.loupe_active=False; return
        if self.p2 and math.dist(self.image_to_canvas_coords(self.p2),(e.x,e.y))<10:
            self.drag_state='p2'; self.selected_point='p2'; self.loupe_active=False; return
        coords=self.canvas_to_image_coords((e.x,e.y))
        if not self.p1:
            self.p1=coords; self.selected_point='p1'; self.info_label.config(text="Left-click END point or use Arrow Keys.")
        elif not self.p2:
            self.p2=coords; self.selected_point='p2'; self.info_label.config(text="Drag points or use Arrow Keys. Then Confirm.")
        self.update_display()

    # Handles panning the view when pressing the middle or right mouse button.
    def on_pan_press(self, e):
        self.drag_state='pan'; self.loupe_active=False; self.last_drag_x,self.last_drag_y=e.x,e.y; self.canvas.config(cursor="fleur")

    # Handles dragging points to adjust their positions.
    def on_drag(self, e):
        if self.drag_state in ['p1', 'p2']:
            coords = self.canvas_to_image_coords((e.x, e.y))
            if self.drag_state == 'p1': self.p1 = coords
            else: self.p2 = coords
            self.update_display()

    # Handles panning the view when dragging with the middle or right mouse button.    
    def on_pan_drag(self, e):
        if self.drag_state=='pan':
            dx,dy=e.x-self.last_drag_x,e.y-self.last_drag_y
            self.view_x-=int(dx/self.zoom_level); self.view_y-=int(dy/self.zoom_level)
            self.last_drag_x,self.last_drag_y=e.x,e.y
            self.update_display()

    # Handles mouse hover events to update the cursor and activate loupe mode.
    def on_hover(self, e):
        self.last_mouse_pos = (e.x, e.y)
        cursor="crosshair"
        if self.p1 and math.dist(self.image_to_canvas_coords(self.p1),(e.x,e.y))<10: cursor="hand2"
        elif self.p2 and math.dist(self.image_to_canvas_coords(self.p2),(e.x,e.y))<10: cursor="hand2"
        if self.drag_state and not e.state & (256 | 1024):
            self.drag_state, self.loupe_active = None, True
        if self.drag_state != 'pan': self.canvas.config(cursor=cursor)
        self.update_display()

    # Handles key presses for nudging the selected point.
    def on_key_press(self, event):
        if not self.selected_point: return
        dx, dy = 0, 0
        if event.keysym == "Left": dx = -1
        elif event.keysym == "Right": dx = 1
        elif event.keysym == "Up": dy = -1
        elif event.keysym == "Down": dy = 1
        if dx != 0 or dy != 0:
            if self.selected_point == 'p1' and self.p1: self.p1 = (self.p1[0] + dx, self.p1[1] + dy)
            elif self.selected_point == 'p2' and self.p2: self.p2 = (self.p2[0] + dx, self.p2[1] + dy)
            self.update_display()

    # Clears the drawn line and resets the state.
    def clear_line(self):
        self.p1, self.p2, self.drag_state, self.selected_point = None, None, None, None
        self.info_label.config(text="Line cleared. Click START point."); self.update_display()

    # Resets the view to the original image size and zoom level. 
    def reset_view(self):
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1: self.after(50, self.reset_view); return
        self.min_zoom = min(canvas_w / self.w, canvas_h / self.h)
        self.zoom_level = self.min_zoom
        self.view_x, self.view_y = 0, 0
        self.update_display()

    # Confirms the scale based on the drawn line and user input.   
    def confirm_scale(self):
        if not self.p1 or not self.p2: messagebox.showerror("Error","Please draw a line first.", parent=self); return
        dist_px = math.dist(self.p1, self.p2)
        length_real = simpledialog.askfloat("Enter Scale", f"The line is {dist_px:.2f} pixels long.\nWhat is the REAL length of the line?", parent=self)
        if length_real is not None and length_real > 0:
            unit = simpledialog.askstring("Enter Unit", "What is the unit of measurement? (e.g., cm, mm, µm)", parent=self)
            if unit and unit.strip():
                self.scale_factor = dist_px / length_real
                self.scale_unit = unit.strip()
                self.is_confirmed = True
                self.destroy()

    # Handles the window closing event, setting the confirmation flag to False.
    def on_closing(self):
        self.is_confirmed=False
        self.destroy()

# ==============================================================================
#  Side Annotation Window
# ==============================================================================

# This window allows users to annotate the long and short axes of an object in an image, providing options for texture classification.
class SideAnnotationWindow(Toplevel):

    # Provides a GUI for annotating the long and short axes of an object in an image.
    def __init__(self, parent, image, roi, scale_factor, scale_unit):
        super().__init__(parent)
        self.title("Annotate Sides")
        self.transient(parent)
        self.is_confirmed, self.annotation_data = False, {}
        rect = cv2.minAreaRect(roi)
        box = cv2.boxPoints(rect)
        self.box = np.int0(box)
        w, h = rect[1]
        self.long_axis_px, self.short_axis_px = max(w, h), min(w, h)
        self.long_axis_scaled, self.short_axis_scaled = self.long_axis_px / scale_factor, self.short_axis_px / scale_factor
        self.scale_unit = scale_unit
        self.long_axis_texture, self.short_axis_texture = StringVar(value="N/A"), StringVar(value="N/A")
        x, y, w, h = cv2.boundingRect(roi)
        padding = 20
        self.roi_img = image[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]
        self.roi_adjusted = roi - (x-padding, y-padding)
        self.box_adjusted = self.box - (x-padding, y-padding)
        self.setup_widgets()

    # Sets up the GUI widgets for the annotation window.
    def setup_widgets(self):
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main_frame, bg="gray")
        self.canvas.pack(pady=5)
        self.draw_roi_on_canvas()
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, expand=True)
        options = ["N/A", "Smooth", "Rough"]
        tk.Label(controls_frame, text=f"Long Axis: {self.long_axis_scaled:.2f} {self.scale_unit}").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(controls_frame, textvariable=self.long_axis_texture, values=options, state="readonly").grid(row=0, column=1, sticky="ew", padx=5)
        tk.Label(controls_frame, text=f"Short Axis: {self.short_axis_scaled:.2f} {self.scale_unit}").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(controls_frame, textvariable=self.short_axis_texture, values=options, state="readonly").grid(row=1, column=1, sticky="ew", padx=5)
        controls_frame.columnconfigure(1, weight=1)
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(10, 0), fill=tk.X)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Confirm", command=self.confirm, bg="#4CAF50", fg="white").pack(side=tk.RIGHT)
    # Draws the ROI and bounding box on the canvas.
    def draw_roi_on_canvas(self):
        display_img = self.roi_img.copy()
        cv2.drawContours(display_img, [self.roi_adjusted], -1, (0, 255, 0), 2)
        cv2.drawContours(display_img, [np.int0(self.box_adjusted)], 0, (255, 0, 255), 2)
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    # Confirms the annotation and saves the data.
    def confirm(self):
        self.is_confirmed = True
        self.annotation_data = {
            'Long_Axis_Length': self.long_axis_scaled,
            'Short_Axis_Length': self.short_axis_scaled,
            'Long_Axis_Texture': self.long_axis_texture.get(),
            'Short_Axis_Texture': self.short_axis_texture.get()
        }
        self.destroy()

# ==============================================================================
#  Main Application Class
# ==============================================================================
# This class manages the main application window, image processing, and user interactions.
class HumanInTheLoopProcessor:
    # Initializes the main application with the root window and sets up the GUI.
    def __init__(self, root_window):
        # --- Application State Variables ---
        self.root = root_window
        self.root.title("SALP v3.5")
        try:
            icon_path = resource_path("app_icon.ico")
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Icon not found (app_icon.ico), skipping: {e}")
        self.root.geometry("1450x950") # Increased width for new controls
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.input_dir, self.output_dir = "", ""
        self.image_paths, self.output_subdirs = [], {}
        self.current_image_index, self.total_images = 0, 0
        self.results_df = None
        self.manual_annotations = {}

        # --- Image Processing State Variables ---
        self.scale_factor, self.scale_unit = 1.0, "px"
        self.original_image, self.processed_image, self.final_rois = None, None, []
        self.preview_zoom_factor = 1.0
        self.last_render_info = {'scale': 1.0, 'offset_x': 0, 'offset_y': 0, 'img_w': 1, 'img_h': 1}
        self.selected_roi_index = -1
        self.drawing_mode = False
        self.new_roi_points = []
        self.contrast_value = None # For contrast slider

        ### NEW FEATURE: Color Picker State Variables
        self.color_picker_active = False
        self.color_picker_history = [] # To store slider states for the undo function

        # --- GUI Element Variables ---
        self.mask_window, self.mask_label, self.results_tree, self.image_label = None, None, None, None
        self.delete_roi_btn, self.annotate_roi_btn = None, None
        self.draw_roi_btn, self.finish_draw_btn, self.cancel_draw_btn = None, None, None
        self.status_label = None
        ### NEW FEATURE: Color Picker Buttons
        self.start_color_pick_btn, self.undo_color_pick_btn, self.finish_color_pick_btn = None, None, None
        
        # --- Application Startup Sequence ---
        self.setup_menu()
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<space>", self.handle_accept)
        self.root.bind("<Key-s>", self.handle_skip); self.root.bind("<Key-S>", self.handle_skip)
        self.root.bind("<Escape>", self.cancel_drawing)

        self.show_welcome_message()
        if self.prompt_for_directories():
            self.start_calibration_or_processing()
        else:
            self.root.destroy()

    # Prompts the user to select input and output directories for image processing.
    def start_calibration_or_processing(self):

        # Prompts the user to set a physical scale for measurements.
        if not self.image_paths: return
        if messagebox.askyesno("Set Scale", "Do you want to set a physical scale for your measurements?\n\n(If you choose 'No', all measurements will be in pixels.)"):
            calib_image = cv2.imread(self.image_paths[0])
            if calib_image is not None:
                calib_window = ScaleCalibrationWindow(self.root, calib_image)
                self.root.wait_window(calib_window)
                if calib_window.is_confirmed:
                    self.scale_factor, self.scale_unit = calib_window.scale_factor, calib_window.scale_unit
                    messagebox.showinfo("Scale Set", f"Scale confirmed: 1 {self.scale_unit} = {self.scale_factor:.4f} pixels.")
                else:
                    messagebox.showwarning("Calibration Canceled", "No scale was set. Measurements will be in pixels.")
            else:
                messagebox.showerror("Error", f"Failed to load first image for calibration:\n{self.image_paths[0]}")
        
        self.setup_results_table()
        self.update_live_results_columns()
        self.process_next_image()

    def setup_results_table(self):
        # Initializes the results DataFrame with appropriate columns.
        columns = [
            'Session_ID', 'Image_Number', 'Filename', 'ROI_ID', 'Centroid_X_px', 'Centroid_Y_px',
            f'Area_({self.scale_unit}^2)', f'Perimeter_({self.scale_unit})', f'Equiv_Diameter_({self.scale_unit})',
            'Aspect_Ratio', 'Circularity_Ratio', 'Solidity_Ratio', 'Orientation_Angle',
            'Long_Axis_Length', 'Short_Axis_Length', 'Long_Axis_Texture', 'Short_Axis_Texture'
        ]
        self.results_df = pd.DataFrame(columns=columns)
    
    # Sets up the application menu with options for saving/loading settings and toggling the mask preview.
    def setup_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Settings...", command=self.save_settings)
        file_menu.add_command(label="Load Settings...", command=self.load_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Live Mask Preview", command=self.toggle_mask_window)

    # Shows a welcome message in the status bar.
        # Shows a welcome message in the status bar.
    def setup_gui(self):
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(status_frame, text="Ready. Left-click to select an ROI.", anchor='w', padx=5, pady=3)
        self.status_label.pack(fill=tk.X)
        main_pane = PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        ### UI SCROLLBAR FEATURE: Create a scrollable container for the control panel ###
        # This outer frame holds both the canvas and the scrollbar
        outer_control_frame = tk.Frame(main_pane, bd=2, relief=tk.SUNKEN)
        main_pane.add(outer_control_frame, width=420) # A bit wider for the scrollbar

        # Create a canvas and a scrollbar
        canvas_controls = tk.Canvas(outer_control_frame, highlightthickness=0)
        scrollbar_controls = ttk.Scrollbar(outer_control_frame, orient="vertical", command=canvas_controls.yview)
        canvas_controls.configure(yscrollcommand=scrollbar_controls.set)

        # This is the actual frame that will contain all the widgets
        control_frame = tk.Frame(canvas_controls, padx=10, pady=10)
        
        # Add the content frame to the canvas
        canvas_controls.create_window((0, 0), window=control_frame, anchor="nw")

        # Update the scroll region when the content frame size changes
        def on_frame_configure(event):
            canvas_controls.configure(scrollregion=canvas_controls.bbox("all"))

        control_frame.bind("<Configure>", on_frame_configure)
        
        # Pack the canvas and scrollbar into the outer frame
        scrollbar_controls.pack(side="right", fill="y")
        canvas_controls.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling on the canvas (cross-platform)
        def on_mouse_wheel(event):
            if event.num == 5 or event.delta < 0:
                canvas_controls.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                canvas_controls.yview_scroll(-1, "units")
        
        # Bind the mouse wheel event to the canvas and the frame inside it
        for widget in [canvas_controls, control_frame]:
            widget.bind("<MouseWheel>", on_mouse_wheel)
            widget.bind("<Button-4>", on_mouse_wheel) # For Linux scroll up
            widget.bind("<Button-5>", on_mouse_wheel) # For Linux scroll down
        ### END UI SCROLLBAR FEATURE ###

        # -- The rest of the setup is the same, but widgets are packed into 'control_frame' --

        right_pane = PanedWindow(main_pane, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        main_pane.add(right_pane)
        image_frame = tk.Frame(right_pane, bg="gray")
        right_pane.add(image_frame, height=650)
        self.image_label = tk.Label(image_frame, bg="gray")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        self.image_label.bind("<MouseWheel>", self.on_preview_zoom)
        self.image_label.bind("<Button-1>", self.handle_image_left_click)
        results_frame = tk.LabelFrame(right_pane, text="Live ROI Measurements", padx=5, pady=5)
        right_pane.add(results_frame, height=250)
        self.results_tree = ttk.Treeview(results_frame, show='headings')
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y'); hsb.pack(side='bottom', fill='x')
        self.results_tree.pack(fill='both', expand=True)
        self.progress_label = tk.Label(control_frame, text="Progress: N/A", font=("Helvetica", 10))
        self.progress_label.pack(pady=(0, 10), anchor='w')
        
        # UI Font for section headers
        ui_font = ("Helvetica", 9, "bold")

        # --- Image Adjustments Frame ---
        img_adj_frame = tk.LabelFrame(control_frame, text="Image Adjustments", padx=5, pady=5, font=ui_font)
        img_adj_frame.pack(fill=tk.X, pady=5)
        self.contrast_value = tk.DoubleVar(value=1.0)
        contrast_slider = Scale(img_adj_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, resolution=0.1,
                                label="Contrast", variable=self.contrast_value, command=self.on_slider_change)
        contrast_slider.pack(fill=tk.X)

        # --- HSV Sliders ---
        def create_hsv_section(parent, text, hsv_type):
            frame = tk.LabelFrame(parent, text=text, padx=5, pady=5, font=ui_font)
            frame.pack(fill=tk.X, pady=2)
            canvas = tk.Canvas(frame, width=300, height=20, bg='black', highlightthickness=0); canvas.pack()
            min_slider = Scale(frame, from_=0, to=179 if hsv_type=='h' else 255, orient=tk.HORIZONTAL, showvalue=1, command=self.on_slider_change); min_slider.pack(fill=tk.X)
            max_slider = Scale(frame, from_=0, to=179 if hsv_type=='h' else 255, orient=tk.HORIZONTAL, showvalue=1, command=self.on_slider_change); max_slider.pack(fill=tk.X)
            return canvas, min_slider, max_slider
        hsv_frame = tk.LabelFrame(control_frame, text="HSV Color Thresholding", padx=5, pady=5, font=ui_font)
        hsv_frame.pack(fill=tk.X, pady=5)
        self.hue_canvas, self.h_min, self.h_max = create_hsv_section(hsv_frame, "Hue", 'h')
        self.sat_canvas, self.s_min, self.s_max = create_hsv_section(hsv_frame, "Saturation", 's')
        self.val_canvas, self.v_min, self.v_max = create_hsv_section(hsv_frame, "Value", 'v')
        self._create_hsv_bars()
        
        # --- ROI Controls ---
        adj_frame = tk.LabelFrame(control_frame, text="ROI Post-Processing", padx=5, pady=5, font=ui_font); adj_frame.pack(fill=tk.X, pady=5)
        self.roi_expansion = Scale(adj_frame, from_=-50, to=50, orient=tk.HORIZONTAL, label="Expand/Shrink (px)", command=self.on_slider_change); self.roi_expansion.pack(fill=tk.X)
        self.min_area = Scale(adj_frame, from_=0, to=50000, orient=tk.HORIZONTAL, label="Min Area (px²)", command=self.on_slider_change); self.min_area.pack(fill=tk.X)
        self.max_area = Scale(adj_frame, from_=0, to=500000, orient=tk.HORIZONTAL, label="Max Area (px²)", command=self.on_slider_change); self.max_area.pack(fill=tk.X)

        # --- Color Picker Tools ---
        picker_tools_frame = tk.LabelFrame(control_frame, text="Color Picker Tool", padx=5, pady=5, font=ui_font)
        picker_tools_frame.pack(fill=tk.X, pady=5)
        picker_grid = tk.Frame(picker_tools_frame)
        picker_grid.pack(fill=tk.X)
        self.start_color_pick_btn = tk.Button(picker_grid, text="Start Color Picking", command=self.enter_color_picker_mode)
        self.undo_color_pick_btn = tk.Button(picker_grid, text="Undo Last Pick", command=self.undo_last_color_pick)
        self.finish_color_pick_btn = tk.Button(picker_grid, text="Finish Picking", command=self.exit_color_picker_mode)
        self.start_color_pick_btn.grid(row=0, column=0, columnspan=2, sticky='ew')
        picker_grid.columnconfigure(0, weight=1); picker_grid.columnconfigure(1, weight=1)

        # --- Tool Buttons ---
        roi_tools_frame = tk.LabelFrame(control_frame, text="ROI Tools", padx=5, pady=5, font=ui_font); roi_tools_frame.pack(fill=tk.X, pady=5)
        tools_grid = tk.Frame(roi_tools_frame); tools_grid.pack(fill=tk.X)
        self.draw_roi_btn = tk.Button(tools_grid, text="Draw New ROI", command=self.enter_drawing_mode)
        self.finish_draw_btn = tk.Button(tools_grid, text="Finish Drawing", command=self.finalize_roi, state=tk.DISABLED, bg="#4CAF50", fg="white")
        self.cancel_draw_btn = tk.Button(tools_grid, text="Cancel Drawing", command=self.cancel_drawing)
        self.draw_roi_btn.grid(row=0, column=0, columnspan=2, sticky='ew')
        tools_grid.columnconfigure(0, weight=1); tools_grid.columnconfigure(1, weight=1)
        
        sel_action_frame = tk.LabelFrame(control_frame, text="Selected ROI Actions", padx=5, pady=5, font=ui_font); sel_action_frame.pack(fill=tk.X, pady=5)
        action_grid = tk.Frame(sel_action_frame); action_grid.pack(fill=tk.X)
        self.delete_roi_btn = tk.Button(action_grid, text="Delete Selected ROI", command=self.delete_selected_roi, state=tk.DISABLED)
        self.delete_roi_btn.grid(row=0, column=0, sticky='ew', padx=(0,2))
        self.annotate_roi_btn = tk.Button(action_grid, text="Annotate Selected ROI", command=self.annotate_selected_roi, state=tk.DISABLED)
        self.annotate_roi_btn.grid(row=0, column=1, sticky='ew', padx=(2,0))
        action_grid.columnconfigure(0, weight=1); action_grid.columnconfigure(1, weight=1)

        # --- Final Actions ---
        action_frame = tk.LabelFrame(control_frame, text="Image Actions", padx=5, pady=5, font=ui_font); action_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        tk.Button(action_frame, text="Reset HSV Controls", command=self.reset_hsv_defaults).pack(fill=tk.X, pady=2)
        tk.Button(action_frame, text="Accept & Next (Space)", command=self.handle_accept, bg="#4CAF50", fg="black", height=2).pack(fill=tk.X, pady=2)
        tk.Button(action_frame, text="Skip Image (S)", command=self.handle_skip, bg="#FF9800", fg="black", height=2).pack(fill=tk.X, pady=2)
        self.reset_hsv_defaults()

    # Creates the HSV bars for visualizing the color thresholds.
    def _create_hsv_bars(self):
        hue_bar = np.zeros((1, 180, 3), dtype=np.uint8); hue_bar[0, :, 0] = np.arange(180); hue_bar[0, :, 1] = 255; hue_bar[0, :, 2] = 255
        self.hue_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(cv2.cvtColor(hue_bar, cv2.COLOR_HSV2RGB), (300, 20), interpolation=cv2.INTER_NEAREST)))
        self.hue_canvas.create_image(0, 0, anchor='nw', image=self.hue_img)
        grad_bar = np.tile(np.arange(256, dtype=np.uint8), (1, 1)).reshape(1, 256)
        sat_bar = np.zeros((1, 256, 3), dtype=np.uint8); sat_bar[0,:,0]=90; sat_bar[0,:,1]=grad_bar; sat_bar[0,:,2]=200
        val_bar = np.zeros((1, 256, 3), dtype=np.uint8); val_bar[0,:,0]=90; val_bar[0,:,1]=200; val_bar[0,:,2]=grad_bar
        self.sat_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(cv2.cvtColor(sat_bar, cv2.COLOR_HSV2RGB), (300, 20))))
        self.val_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(cv2.cvtColor(val_bar, cv2.COLOR_HSV2RGB), (300, 20))))
        self.sat_canvas.create_image(0, 0, anchor='nw', image=self.sat_img); self.val_canvas.create_image(0, 0, anchor='nw', image=self.val_img)
        self.hue_overlay1 = self.hue_canvas.create_rectangle(0,0,0,20, fill='white', stipple='gray50', outline="")
        self.hue_overlay2 = self.hue_canvas.create_rectangle(0,0,0,20, fill='white', stipple='gray50', outline="")
        self.sat_overlay1 = self.sat_canvas.create_rectangle(0,0,0,20, fill='white', stipple='gray50', outline="")
        self.sat_overlay2 = self.sat_canvas.create_rectangle(0,0,0,20, fill='white', stipple='gray50', outline="")
        self.val_overlay1 = self.val_canvas.create_rectangle(0,0,0,20, fill='white', stipple='gray50', outline="")
        self.val_overlay2 = self.val_canvas.create_rectangle(0,0,0,20, fill='white', stipple='gray50', outline="")
        
    # Updates the HSV overlay bars based on current slider values
    def _update_hsv_bars(self):
        w, h = 300, 20
        h_min_pos = self.h_min.get()/179*w; h_max_pos = self.h_max.get()/179*w
        self.hue_canvas.coords(self.hue_overlay1, 0,0, h_min_pos, h); self.hue_canvas.coords(self.hue_overlay2, h_max_pos, 0, w, h)
        s_min_pos = self.s_min.get()/255*w; s_max_pos = self.s_max.get()/255*w
        self.sat_canvas.coords(self.sat_overlay1, 0,0, s_min_pos, h); self.sat_canvas.coords(self.sat_overlay2, s_max_pos, 0, w, h)
        v_min_pos = self.v_min.get()/255*w; v_max_pos = self.v_max.get()/255*w
        self.val_canvas.coords(self.val_overlay1, 0,0, v_min_pos, h); self.val_canvas.coords(self.val_overlay2, v_max_pos, 0, w, h)

    # Handles slider changes for contrast and HSV values, updating the image display.
    def reset_hsv_defaults(self):
        self.contrast_value.set(1.0)
        self.h_min.set(2); self.h_max.set(91); self.s_min.set(43); self.s_max.set(164); self.v_min.set(48); self.v_max.set(196)
        self.on_slider_change(None)

    # Runs the entire detection pipeline, including contrast adjustment, HSV thresholding, contour detection, and ROI post-processing.
    def run_detection_pipeline(self):
        if self.original_image is None: return
        
        # 1. Apply contrast adjustment
        contrast = self.contrast_value.get()
        self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=contrast, beta=0)

        # 2. Perform HSV thresholding on the contrast-adjusted image
        hsv_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2HSV)
        lower=np.array([self.h_min.get(),self.s_min.get(),self.v_min.get()]); upper=np.array([self.h_max.get(),self.s_max.get(),self.v_max.get()])
        mask = cv2.inRange(hsv_image, lower, upper)
        
        # 3. Find initial contours from the raw mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Post-process the contours
        expansion = self.roi_expansion.get()
        if expansion != 0:
            # Create a mask from contours to expand/shrink
            temp_mask = np.zeros_like(mask)
            cv2.drawContours(temp_mask, contours, -1, 255, -1)
            kernel = np.ones((abs(expansion), abs(expansion)), np.uint8)
            processed_mask = cv2.dilate(temp_mask, kernel) if expansion > 0 else cv2.erode(temp_mask, kernel)
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter by area
        min_a, max_a = self.min_area.get(), self.max_area.get()
        self.final_rois = [roi for roi in contours if min_a <= cv2.contourArea(roi) <= max_a]
        
        self.deselect_roi()
        self.update_image_display()

    # Handles slider changes for contrast and HSV values, updating the image display.
    def update_image_display(self):
        if self.processed_image is None: return
        # Display image is now always the contrast-adjusted one
        display_image = self.processed_image.copy()
        
        cv2.drawContours(display_image, self.final_rois, -1, (0, 255, 0), 2)
        if self.selected_roi_index != -1:
            cv2.drawContours(display_image, self.final_rois, self.selected_roi_index, (0, 255, 255), 3)
        for i, roi in enumerate(self.final_rois):
            M = cv2.moments(roi); cx = int(M["m10"] / (M["m00"] + 1e-6)); cy = int(M["m01"] / (M["m00"] + 1e-6))
            id_text = f"{i+1}"; (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_image, (cx, cy), (cx + text_w + 4, cy - text_h - 6), (255, 0, 255), -1)
            cv2.putText(display_image, id_text, (cx + 2, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.drawing_mode and self.new_roi_points:
            pts = np.array(self.new_roi_points, dtype=np.int32)
            cv2.polylines(display_image, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            for point in self.new_roi_points: cv2.circle(display_image, point, 5, (0, 0, 255), -1)
            
        self._update_tkinter_label(self.image_label, display_image)
        self.update_live_results_table()
        if self.mask_window and self.mask_window.winfo_exists():
            # Generate refrshing mask on-the-fly for preview
            preview_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(preview_mask, self.final_rois, -1, 255, -1)
            self._update_tkinter_label(self.mask_label, preview_mask, is_bgr=False)

    # Converts the OpenCV image to a format suitable for Tkinter display.
    def _update_tkinter_label(self, tk_label, cv_image, is_bgr=True):
        
        if cv_image is None: return
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) if is_bgr else cv_image
        h, w = img_rgb.shape[:2]; label_w, label_h = tk_label.winfo_width(), tk_label.winfo_height()
        if label_w <= 1 or label_h <= 1: tk_label.after(50, lambda: self._update_tkinter_label(tk_label, cv_image, is_bgr)); return
        scale_fit = min(label_w / w, label_h / h); final_scale = scale_fit * self.preview_zoom_factor
        new_w, new_h = int(w * final_scale), int(h * final_scale)
        if tk_label == self.image_label:
            self.last_render_info = {'scale': final_scale, 'offset_x': (label_w-new_w)//2, 'offset_y': (label_h-new_h)//2, 'img_w': new_w, 'img_h': new_h}
        if new_w > 0 and new_h > 0:
            resized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_pil = Image.fromarray(resized_img); img_tk = ImageTk.PhotoImage(image=img_pil)
            tk_label.config(image=img_tk); tk_label.image = img_tk

    # Sets up the columns for the live results table.
    def update_live_results_columns(self):
        
        if self.results_tree is None: return
        cols = ('ID', f'Area ({self.scale_unit}²)', f'Perimeter ({self.scale_unit})', 'Aspect Ratio', 'Circularity')
        self.results_tree["columns"] = cols
        for col in cols: self.results_tree.heading(col, text=col); self.results_tree.column(col, width=120, anchor='center')
        self.results_tree.column('ID', width=40)

    # Updates the live results table with measurements for each ROI.
    def update_live_results_table(self):
        if self.results_tree is None: return
        for row in self.results_tree.get_children(): self.results_tree.delete(row)
        for i, roi in enumerate(self.final_rois):
            pixel_area=cv2.contourArea(roi); pixel_perimeter=cv2.arcLength(roi, True); x,y,w,h=cv2.boundingRect(roi)
            scaled_area=pixel_area/(self.scale_factor**2); scaled_perimeter=pixel_perimeter/self.scale_factor
            aspect_ratio=float(w)/h if h!=0 else 0; circularity=(4*math.pi*pixel_area)/(pixel_perimeter**2) if pixel_perimeter!=0 else 0
            values=(i+1, f"{scaled_area:.2f}", f"{scaled_perimeter:.2f}", f"{aspect_ratio:.3f}", f"{circularity:.3f}")
            item = self.results_tree.insert("", "end", values=values)
            if i == self.selected_roi_index: self.results_tree.selection_set(item)

    # Handles the acceptance of the current image and saves measurements.
    def handle_accept(self, event=None):
        if not self.image_paths: return
        current_path = self.image_paths[self.current_image_index]; current_filename = os.path.basename(current_path)
        image_results = []
        for i, roi in enumerate(self.final_rois):
            pixel_area=cv2.contourArea(roi); pixel_perimeter=cv2.arcLength(roi, True); M=cv2.moments(roi)
            cx=int(M['m10']/(M['m00']+1e-6)); cy=int(M['m01']/(M['m00']+1e-6)); x,y,w,h=cv2.boundingRect(roi)
            aspect_ratio=float(w)/h if h!=0 else 0; hull=cv2.convexHull(roi); hull_area=cv2.contourArea(hull)
            solidity=float(pixel_area)/hull_area if hull_area!=0 else 0
            circularity=(4*math.pi*pixel_area)/(pixel_perimeter**2) if pixel_perimeter!=0 else 0
            orientation=cv2.fitEllipse(roi)[-1] if len(roi)>=5 else np.nan
            scaled_area=pixel_area/(self.scale_factor**2); scaled_perimeter=pixel_perimeter/self.scale_factor
            scaled_equiv_diameter=(np.sqrt(4*pixel_area/np.pi))/self.scale_factor
            roi_id = i + 1; annotation = self.manual_annotations.get(roi_id, {})
            image_results.append({
                'Session_ID': self.session_id, 'Image_Number': self.current_image_index + 1, 'Filename': current_filename, 'ROI_ID': roi_id,
                'Centroid_X_px': cx, 'Centroid_Y_px': cy, f'Area_({self.scale_unit}^2)': scaled_area, f'Perimeter_({self.scale_unit})': scaled_perimeter,
                f'Equiv_Diameter_({self.scale_unit})': scaled_equiv_diameter, 'Aspect_Ratio': aspect_ratio, 'Circularity_Ratio': circularity, 'Solidity_Ratio': solidity,
                'Orientation_Angle': orientation, 'Long_Axis_Length': annotation.get('Long_Axis_Length', ''), 'Short_Axis_Length': annotation.get('Short_Axis_Length', ''),
                'Long_Axis_Texture': annotation.get('Long_Axis_Texture', ''), 'Short_Axis_Texture': annotation.get('Short_Axis_Texture', '')
            })
        if image_results: self.results_df = pd.concat([self.results_df, pd.DataFrame(image_results)], ignore_index=True)
        temp_csv_path = os.path.join(self.output_subdirs['temp_measurements'], f"temp_results_{self.session_id}.csv")
        self.results_df.to_csv(temp_csv_path, index=False, float_format='%.4f')
        base_name = os.path.splitext(current_filename)[0]
        
        # --- KEY FIX: Generate final mask and ROI image from the final ROI list ---
        # 1. Create the final binary mask for saving
        final_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        if self.final_rois:
            cv2.drawContours(final_mask, self.final_rois, -1, 255, -1)
        
        # 2. Create the final ROI image for saving (on the original image, not the contrast-adjusted one)
        final_roi_image = self.original_image.copy()
        if self.final_rois:
            cv2.drawContours(final_roi_image, self.final_rois, -1, (0, 255, 0), 2)
            
        cv2.imwrite(os.path.join(self.output_subdirs['image_with_roi'], f"{base_name}_roi.jpg"), final_roi_image)
        cv2.imwrite(os.path.join(self.output_subdirs['filled_masks'], f"{base_name}_mask.png"), final_mask)
        
        self.current_image_index += 1; self.process_next_image()

    # Handles skipping the current image, copying it to the skipped directory if not corrupt.
    def handle_skip(self, event=None, is_corrupt=False):
        if not self.image_paths: return
        current_path = self.image_paths[self.current_image_index]
        destination_path = os.path.join(self.output_subdirs['skipped'], os.path.basename(current_path))
        print(f"Skipping image: {os.path.basename(current_path)}")
        if not is_corrupt:
            try: shutil.copy(current_path, destination_path)
            except Exception as e: print(f"Could not copy skipped file: {e}")
        self.current_image_index += 1; self.process_next_image()

    # Handles left-click events on the image label to select or draw ROIs.
    def handle_image_left_click(self, event):
        info = self.last_render_info; img_x = int((event.x-info['offset_x'])/info['scale']); img_y = int((event.y-info['offset_y'])/info['scale'])
        
        ### NEW FEATURE: Divert click to color picker if active
        if self.color_picker_active:
            self.handle_color_pick(img_x, img_y)
            return
            
        if self.drawing_mode:
            self.new_roi_points.append((img_x, img_y))
            if len(self.new_roi_points) >= 3:
                self.finish_draw_btn.config(state=tk.NORMAL)
            self.update_image_display()
            return
            
        clicked_roi_index = -1
        for i in range(len(self.final_rois) - 1, -1, -1):
            if cv2.pointPolygonTest(self.final_rois[i], (img_x, img_y), False) >= 0:
                clicked_roi_index = i; break
        if clicked_roi_index != -1: self.select_roi(clicked_roi_index)
        else: self.deselect_roi()

    # Enable/disable buttons based on whether an ROI is selected or not
    def update_roi_action_buttons(self):
        state = tk.NORMAL if self.selected_roi_index != -1 else tk.DISABLED
        self.delete_roi_btn.config(state=state); self.annotate_roi_btn.config(state=state)
        if self.selected_roi_index != -1:
            self.status_label.config(text=f"ROI #{self.selected_roi_index + 1} selected. Use action buttons or click background to deselect.")
        elif not self.drawing_mode and not self.color_picker_active:
            self.status_label.config(text="Ready. Left-click to select an ROI or use tools.")

    # Deselect any currently selected ROI
    def select_roi(self, index):
        if not self.drawing_mode and not self.color_picker_active:
            self.selected_roi_index = index; self.update_roi_action_buttons(); self.update_image_display()
            print(f"Selected ROI #{index + 1}")

    # Deselects the currently selected ROI and updates the display
    def deselect_roi(self):
        self.selected_roi_index = -1; self.update_roi_action_buttons(); self.update_image_display()

    # Deletes the currently selected ROI and updates the display.
    def delete_selected_roi(self):
        if self.selected_roi_index != -1:
            index_to_delete = self.selected_roi_index
            self.final_rois.pop(index_to_delete)
            if (index_to_delete + 1) in self.manual_annotations: del self.manual_annotations[index_to_delete + 1]
            self.deselect_roi()

    # Opens a side annotation window for the selected ROI, allowing manual input of additional measurements.
    def annotate_selected_roi(self):
        if self.selected_roi_index != -1:
            roi = self.final_rois[self.selected_roi_index]; roi_id = self.selected_roi_index + 1
            annot_window = SideAnnotationWindow(self.root, self.original_image, roi, self.scale_factor, self.scale_unit)
            if roi_id in self.manual_annotations:
                existing_data = self.manual_annotations[roi_id]
                annot_window.long_axis_texture.set(existing_data.get("Long_Axis_Texture", "N/A"))
                annot_window.short_axis_texture.set(existing_data.get("Short_Axis_Texture", "N/A"))
            self.root.wait_window(annot_window)
            if annot_window.is_confirmed:
                self.manual_annotations[roi_id] = annot_window.annotation_data
                messagebox.showinfo("Saved", f"Annotation for ROI {roi_id} saved.", parent=self.root)

    # Enters drawing mode for manually adding new ROIs
    def enter_drawing_mode(self):
        self.exit_color_picker_mode() # Ensure color picker is off
        self.drawing_mode = True; self.deselect_roi(); self.new_roi_points = []
        self.image_label.config(cursor="crosshair")
        self.status_label.config(text="DRAWING MODE: Left-click to add points. Use buttons to Finish or Cancel.")
        self.draw_roi_btn.grid_remove()
        self.finish_draw_btn.grid(row=0, column=0, sticky='ew', padx=(0,2))
        self.cancel_draw_btn.grid(row=0, column=1, sticky='ew', padx=(2,0))

    # Cancels the drawing mode
    def cancel_drawing(self, event=None):
        self.drawing_mode = False; self.new_roi_points = []
        self.image_label.config(cursor="")
        self.finish_draw_btn.grid_remove(); self.cancel_draw_btn.grid_remove()
        self.draw_roi_btn.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.finish_draw_btn.config(state=tk.DISABLED)
        self.update_roi_action_buttons(); self.update_image_display()

    # Finalizes the drawing of a new ROI
    def finalize_roi(self):
        if len(self.new_roi_points) >= 3:
            new_contour = np.array(self.new_roi_points, dtype=np.int32).reshape((-1, 1, 2))
            self.final_rois.append(new_contour)
            print(f"Manually added new ROI with {len(self.new_roi_points)} points.")
        else:
            messagebox.showwarning("Drawing Error", "An ROI must have at least 3 points.", parent=self.root)
        self.cancel_drawing()

    ### NEW FEATURE: Color Picker Methods
    def enter_color_picker_mode(self):
        self.cancel_drawing() # Ensure drawing mode is off
        self.color_picker_active = True
        self.color_picker_history = []
        self.image_label.config(cursor="tcross")
        self.status_label.config(text="COLOR PICKER MODE: Click on the image to select a color range. Click multiple times to expand.")
        self.start_color_pick_btn.grid_remove()
        self.undo_color_pick_btn.grid(row=0, column=0, sticky='ew', padx=(0,2))
        self.finish_color_pick_btn.grid(row=0, column=1, sticky='ew', padx=(2,0))
        self.undo_color_pick_btn.config(state=tk.DISABLED)

    def exit_color_picker_mode(self):
        self.color_picker_active = False
        self.image_label.config(cursor="")
        self.undo_color_pick_btn.grid_remove()
        self.finish_color_pick_btn.grid_remove()
        self.start_color_pick_btn.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.update_roi_action_buttons()

    def handle_color_pick(self, img_x, img_y):
        if not (0 <= img_y < self.original_image.shape[0] and 0 <= img_x < self.original_image.shape[1]):
            return # Click was outside the image bounds
        
        # Save current state for undo
        current_state = (self.h_min.get(), self.h_max.get(), self.s_min.get(), self.s_max.get(), self.v_min.get(), self.v_max.get())
        self.color_picker_history.append(current_state)
        self.undo_color_pick_btn.config(state=tk.NORMAL)
        
        # Get color and convert to HSV
        bgr_color = self.original_image[img_y, img_x]
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_color[0], hsv_color[1], hsv_color[2]
        
        # Set ranges based on pick
        H_TOL, SV_TOL = 5, 25 # Tolerances for Hue and Sat/Val
        if len(self.color_picker_history) == 1: # First pick
            self.h_min.set(max(0, h - H_TOL)); self.h_max.set(min(179, h + H_TOL))
            self.s_min.set(max(0, s - SV_TOL)); self.s_max.set(min(255, s + SV_TOL))
            self.v_min.set(max(0, v - SV_TOL)); self.v_max.set(min(255, v + SV_TOL))
        else: # Subsequent picks
            self.h_min.set(min(self.h_min.get(), h)); self.h_max.set(max(self.h_max.get(), h))
            self.s_min.set(min(self.s_min.get(), s)); self.s_max.set(max(self.s_max.get(), s))
            self.v_min.set(min(self.v_min.get(), v)); self.v_max.set(max(self.v_max.get(), v))

        self.on_slider_change(None) # Update the display with new values

    def undo_last_color_pick(self):
        if self.color_picker_history:
            last_state = self.color_picker_history.pop()
            self.h_min.set(last_state[0]); self.h_max.set(last_state[1])
            self.s_min.set(last_state[2]); self.s_max.set(last_state[3])
            self.v_min.set(last_state[4]); self.v_max.set(last_state[5])
            self.on_slider_change(None)
        if not self.color_picker_history:
            self.undo_color_pick_btn.config(state=tk.DISABLED)

    # Handles the closing of the main application window, prompting the user to confirm if they want to exit.
    def on_preview_zoom(self, event=None, reset=False):
        if reset: self.preview_zoom_factor = 1.0
        elif event:
            factor = 1.1 if (event.delta > 0 or event.num == 4) else (1 / 1.1)
            self.preview_zoom_factor = max(0.1, self.preview_zoom_factor * factor)
        self.update_image_display()

    # Toggles the live mask preview window, creating it if it doesn't exist or closing it if it does.
    def toggle_mask_window(self):
        if self.mask_window and self.mask_window.winfo_exists():
            self.mask_window.destroy(); self.mask_window = None
        else:
            self.mask_window = Toplevel(self.root); self.mask_window.title("Live Mask Preview"); self.mask_window.geometry("600x600")
            self.mask_label = tk.Label(self.mask_window, bg="gray"); self.mask_label.pack(expand=True, fill=tk.BOTH)
            self.update_image_display() # This will trigger the mask preview update

    # Saves the current settings to a JSON file, allowing the user to create presets for future sessions.
    def save_settings(self):
        settings = {'contrast': self.contrast_value.get(), 'h_min': self.h_min.get(), 'h_max': self.h_max.get(), 's_min': self.s_min.get(), 's_max': self.s_max.get(), 'v_min': self.v_min.get(), 'v_max': self.v_max.get(), 'roi_expansion': self.roi_expansion.get(), 'min_area': self.min_area.get(), 'max_area': self.max_area.get()}
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Settings Preset")
        if filepath:
            try:
                with open(filepath, 'w') as f: json.dump(settings, f, indent=4)
                messagebox.showinfo("Success", f"Settings saved to {os.path.basename(filepath)}")
            except Exception as e: messagebox.showerror("Error", f"Failed to save settings file.\n\nError: {e}")

    # Loads settings from a JSON file, allowing the user to apply previously saved presets.
    def load_settings(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], title="Load Settings Preset")
        if not filepath: return
        try:
            with open(filepath, 'r') as f: settings = json.load(f)
            self.contrast_value.set(settings.get('contrast', 1.0))
            self.h_min.set(settings.get('h_min',0)); self.h_max.set(settings.get('h_max',179)); self.s_min.set(settings.get('s_min',0)); self.s_max.set(settings.get('s_max',255)); self.v_min.set(settings.get('v_min',0)); self.v_max.set(settings.get('v_max',255))
            self.roi_expansion.set(settings.get('roi_expansion',0)); self.min_area.set(settings.get('min_area',1000)); self.max_area.set(settings.get('max_area',40000))
            self.on_slider_change(None)
            messagebox.showinfo("Success", "Settings loaded and applied.")
        except Exception as e: messagebox.showerror("Error", f"Failed to load settings file.\n\nError: {e}")

    #  Displays a welcome message with instructions for using the application.
    def show_welcome_message(self):
        messagebox.showinfo("Welcome!", "Welcome to SALP v2.4!\n\n- Use the new 'Color Picker Tool' to quickly set HSV values.\n- Select input and output folders.\n- Use the controls to adjust detection.\n- Left-click an ROI to select it, then use the action buttons.")

    # Prompts the user to select input and output directories, initializes the session, and loads the image list.
    def prompt_for_directories(self):
        self.input_dir = filedialog.askdirectory(title="Select Input Image Folder")
        if not self.input_dir: return False
        self.output_dir = filedialog.askdirectory(title="Select Main Output Folder")
        if not self.output_dir: return False
        if len(os.listdir(self.output_dir)) > 0:
            if not messagebox.askyesno("Warning", "The output folder is not empty. Session data will be in a new subfolder. Continue?"):
                return False
        self.root.title(f"SALP v3.5 - Session {self.session_id}")
        self.setup_output_structure(); self.load_image_list()
        return self.total_images > 0

    # Initializes the output directory structure for the current session, creating necessary subdirectories.
    def setup_output_structure(self):
        base_path = os.path.join(self.output_dir, f"session_{self.session_id}")
        dir_names = ["filled_masks", "image_with_roi", "temp_measurements", "completed_measurements", "skipped"]
        for name in dir_names:
            path = os.path.join(base_path, name); os.makedirs(path, exist_ok=True)
            self.output_subdirs[name] = path
        print(f"Session '{self.session_id}' started. Output will be saved in: {base_path}")

    # Loads the list of images from the input directory, filtering by valid image file extensions.
    def load_image_list(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        self.image_paths = sorted([os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.lower().endswith(valid_extensions)])
        self.total_images = len(self.image_paths)
        if self.total_images == 0: messagebox.showerror("Error", "No valid images found in the selected directory.")

    # Processes the next image in the list, resetting the state and updating the display.
    def process_next_image(self):
        if self.current_image_index >= self.total_images: self.finalize_session(); return
        self.manual_annotations.clear(); self.cancel_drawing(); self.exit_color_picker_mode(); self.deselect_roi(); self.preview_zoom_factor = 1.0
        self.progress_label.config(text=f"Image {self.current_image_index + 1} of {self.total_images}")
        image_path = self.image_paths[self.current_image_index]
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            messagebox.showwarning("File Error", f"Could not read image file:\n{os.path.basename(image_path)}\nIt will be skipped.")
            self.handle_skip(is_corrupt=True); return
        self.reset_hsv_defaults() # Resets sliders for each new image

    # Updates the HSV bars and runs the detection pipeline when sliders are changed.
    def on_slider_change(self, _):
        self._update_hsv_bars()
        if not self.drawing_mode: self.run_detection_pipeline()
        
    # Handles the closing of the main application window, prompting the user to confirm if they want to exit.
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to exit? Your progress will be saved."):
            self.finalize_session(is_manual_exit=True)

    # Finalizes the session by saving results, generating a summary, and cleaning up temporary files.
    def finalize_session(self, is_manual_exit=False):
        if self.results_df is None or not self.output_subdirs or self.results_df.empty:
            if not is_manual_exit: messagebox.showinfo("Session End", "Session closed. No measurements were saved.")
            if self.root: self.root.destroy()
            return
        final_csv_path = os.path.join(self.output_subdirs['completed_measurements'], f"final_results_{self.session_id}.csv")
        summary_path = os.path.join(self.output_subdirs['completed_measurements'], f"summary_{self.session_id}.txt")
        self.results_df.to_csv(final_csv_path, index=False, float_format='%.4f')
        scale_info = "No physical scale set (measurements are in pixels)." if self.scale_unit=="px" else f"Scale: 1 {self.scale_unit} = {self.scale_factor:.4f} pixels."
        summary_text = f"""--- Analysis Session Summary ---
Session ID: {self.session_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
--- Scale Information ---
{scale_info}
--- Directories ---
Input Directory: {self.input_dir}
Output Directory: {os.path.dirname(self.output_subdirs['completed_measurements'])}
--- Results ---
Total Images in Batch: {self.total_images}
Images Processed: {self.current_image_index}
Total Objects Detected: {len(self.results_df)}
--- Output Files ---
Final Data: {final_csv_path}
Session Summary: {summary_path}
"""
        with open(summary_path, 'w') as f: f.write(summary_text)
        message = "Batch processing complete!" if not is_manual_exit else "Session exited."
        messagebox.showinfo("Session Finished", f"{message}\n\nResults and summary saved to 'completed_measurements' folder.")
        temp_dir = self.output_subdirs.get('temp_measurements')
        if temp_dir and os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir); print(f"Removed temporary directory: {temp_dir}")
            except Exception as e: print(f"Could not remove temp directory: {e}")
        if self.root: self.root.destroy()

# ==============================================================================
#  Application Entry Point
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = HumanInTheLoopProcessor(root)
    root.mainloop()