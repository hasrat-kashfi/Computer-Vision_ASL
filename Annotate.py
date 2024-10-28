import os
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set directories for reading and saving images
dataset_dir = './dataset 1/asl_alphabet_train/asl_alphabet_train'  # Updated to match folder structure
output_dir = './Updated files'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each class folder in the dataset directory
for class_folder in os.listdir(dataset_dir):
    class_folder_path = os.path.join(dataset_dir, class_folder)
    
    # Check if it's a folder (i.e., a class folder like 'A', 'B', etc.)
    if not os.path.isdir(class_folder_path):
        continue

    output_class_folder = os.path.join(output_dir, class_folder)

    # Ensure output class folder exists
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)

    print(f"Processing folder: {class_folder}")  # Debugging statement

    # Process each image in the class folder
    for img_file in os.listdir(class_folder_path):
        img_path = os.path.join(class_folder_path, img_file)
        
        # Check if it is a file and has an image extension
        if not os.path.isfile(img_path) or not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Warning: Skipping {img_file} - not a valid image file.")
            continue

        # Read the image
        img = cv2.imread(img_path)
        
        # Skip files that failed to load
        if img is None:
            print(f"Warning: Skipping {img_file} - unable to load.")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe Hands and process the image
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
            results = hands.process(img_rgb)

            # Annotate the image if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

        # Save the annotated image to the output folder
        output_img_path = os.path.join(output_class_folder, img_file)
        cv2.imwrite(output_img_path, img)
        print(f"Saved annotated image: {output_img_path}")  # Debugging statement

print("All images have been processed and saved in the 'Updated files' folder.")
