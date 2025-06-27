from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Load the trained YOLOv8 model (use your 'best.pt' or 'last.pt' weights)
model_path = r'D:/Detection-and-recognition-of-multi-column-text-in-Hindi-newspaper--main/best (1).pt'  # Replace with your local path to the model
model = YOLO(model_path)

# Specify the path to your single image
image_path = r'C:\Users\daksh\OneDrive\Desktop\Hindi 3.png'  # Replace with your local image path

# Read the image
im = cv2.imread(image_path)

if im is None:
    print(f"Error: Couldn't read the image file {image_path}")
else:
    # Run prediction on the image
    results = model(im)

    # Extract results and plot them on the image
    result_img = results[0].plot()  # Plot draws boxes, labels, and scores directly on the image

    # Ensure image is in RGB format for display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Show the image with predictions
    plt.figure(figsize=(10, 8))  # Set figure size
    plt.imshow(result_img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

    # Print predicted classes and bounding boxes
    pred_classes = results[0].boxes.cls.cpu().numpy()  # Convert to numpy array
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert to numpy array

    print("Predicted Classes:", pred_classes)
    print("Predicted Boxes:", pred_boxes)
