import cv2
import mediapipe as mp
import time
import os
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)  # Detect 1 hand

# Variables for time tracking
object_start_time = None
capture_duration = 3  # Seconds to wait before capture
count = 1  # For naming captured images
camera_running = True  # Control the camera state
cropped_image = None  # Store cropped image globally

# Set folder path for saving images
capture_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Testing 360", "Captures")
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

# Initialize tkinter root window
root = tk.Tk()
root.title("Hand Detection and Image Capture")

# Set the background color for the window
root.configure(bg='#1f1f1f')

# Video label where camera frames will be shown
video_label = Label(root, bg='#1f1f1f')
video_label.pack(pady=20)

# Instruction label for step-by-step instructions
instructions_label = Label(root, text="Put your hand inside the detection box", font=("Helvetica", 14), fg="white", bg="#1f1f1f")
instructions_label.pack(pady=10)

# Function to restart the camera
def restart_camera():
    global camera_running
    camera_running = True
    crop_button.pack_forget()  # Hide the Crop Image button
    try_again_button.pack_forget()  # Hide the Try Again button
    analyze_button.pack_forget()  # Hide Analyze Image button
    try_another_button.pack_forget()  # Hide Try Another button
    instructions_label.config(text="Put your hand inside the detection box")  # Reset instructions
    start_video()

# Function to crop around the hand landmarks
def crop_hand(image, hand_x, hand_y, margin=30):
    # Get the bounding box of the hand
    x_min, x_max = max(0, int(min(hand_x) - margin)), min(image.shape[1], int(max(hand_x) + margin))
    y_min, y_max = max(0, int(min(hand_y) - margin)), min(image.shape[0], int(max(hand_y) + margin))

    # Crop the image around the hand
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

# Function to handle when the user presses the "Crop Image" button
def crop_image_action():
    global cropped_image
    if cropped_image is not None:
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=cropped_pil)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Show the Analyze button
        analyze_button.pack(side="bottom", pady=10)

# Function to save the cropped image and move to the next step
def analyze_image_action():
    global cropped_image, count
    # Save cropped image
    capture_path = os.path.join(capture_folder, f"capture{count}_cropped.png")
    cv2.imwrite(capture_path, cropped_image)
    print(f"Cropped image saved as {capture_path}")
    count += 1
    
    # Update instructions and hide all buttons except "Try Another"
    instructions_label.config(text="Image saved! Try another one.")
    crop_button.pack_forget()
    try_again_button.pack_forget()
    analyze_button.pack_forget()
    
    # Show the Try Another button
    try_another_button.pack(side="bottom", pady=10)

# Function to start or resume the camera
def start_video():
    global object_start_time, count, camera_running, cropped_image
    
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    def update_frame():
        global object_start_time, count, camera_running, cropped_image
        
        if not camera_running:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return
        
        # Convert frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        # Mediapipe detection
        results = hands.process(frame_rgb)

        # Define bounding box size and location
        square_size = 500
        top_left_x = (width - square_size) // 2
        top_left_y = (height - square_size) // 2
        bottom_right_x = top_left_x + square_size
        bottom_right_y = top_left_y + square_size
        
        # Draw the bounding box
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

        hand_inside_square = False  # To check if hand is in the bounding box
        hand_x, hand_y = [], []  # Store hand coordinates for cropping later

        # Check if any hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand bounding box from landmarks
                hand_x = [landmark.x * width for landmark in hand_landmarks.landmark]
                hand_y = [landmark.y * height for landmark in hand_landmarks.landmark]
                
                # Check if hand is inside the bounding box
                if (top_left_x < min(hand_x) < bottom_right_x and
                    top_left_x < max(hand_x) < bottom_right_x and
                    top_left_y < min(hand_y) < bottom_right_y and
                    top_left_y < max(hand_y) < bottom_right_y):
                    hand_inside_square = True

                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Capture image if hand is inside the square and stationary
        if hand_inside_square:
            if object_start_time is None:
                object_start_time = time.time()  # Start timer
                instructions_label.config(text="Remain stationary for 3 seconds")  # Update instructions
            else:
                elapsed_time = time.time() - object_start_time
                if elapsed_time >= capture_duration:
                    # Crop around the hand
                    cropped_image = crop_hand(frame, hand_x, hand_y, margin=30)  # Crop around hand
                    
                    instructions_label.config(text="Hand detected! Choose an option.")  # Update instructions
                    
                    # Show the buttons for next actions
                    crop_button.pack(side="left", padx=20, pady=10)
                    try_again_button.pack(side="right", padx=20, pady=10)
                    camera_running = False  # Stop the camera
                    cap.release()
                    return
        else:
            object_start_time = None  # Reset if no hand is detected
            instructions_label.config(text="Put your hand inside the detection box")  # Update instructions if no hand detected

        # Convert the frame to ImageTk format to display in the tkinter label
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        imgtk = ImageTk.PhotoImage(image=img_pil)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Continue updating the frame every 10ms
        video_label.after(10, update_frame)
    
    update_frame()

# Improved Button styling for the UI
def style_button(button, bg_color, fg_color, hover_bg, hover_fg, font_size):
    button.config(
        font=("Helvetica", font_size),
        bg=bg_color,
        fg=fg_color,
        activebackground=hover_bg,
        activeforeground=hover_fg,
        relief="flat",
        bd=0,
        padx=15,
        pady=10,
    )
    button.bind("<Enter>", lambda e: button.config(bg=hover_bg, fg=hover_fg))
    button.bind("<Leave>", lambda e: button.config(bg=bg_color, fg=fg_color))

# Buttons for the UI
crop_button = Button(root, text="Crop Image", command=crop_image_action)
try_again_button = Button(root, text="Try Again", command=restart_camera)
analyze_button = Button(root, text="Analyze Image", command=analyze_image_action)
try_another_button = Button(root, text="Try Another", command=restart_camera)

# Styling the buttons with a dark theme and hover effects
style_button(crop_button, bg_color="#333", fg_color="white", hover_bg="#555", hover_fg="white", font_size=12)
style_button(try_again_button, bg_color="#333", fg_color="white", hover_bg="#555", hover_fg="white", font_size=12)
style_button(analyze_button, bg_color="#333", fg_color="white", hover_bg="#555", hover_fg="white", font_size=12)
style_button(try_another_button, bg_color="#333", fg_color="white", hover_bg="#555", hover_fg="white", font_size=12)

# Start the camera when the app launches
start_video()

# Run the tkinter main loop
root.mainloop()