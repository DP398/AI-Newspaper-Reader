import cv2
import os
import pytesseract
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Paths
model_path = "best_30_epochs.pt"  # Update this to your local model path
input_image_path = "D:\Detection-and-recognition-of-multi-column-text-in-Hindi-newspaper--main\H1.jpg"  # Update this to your image path
output_dir = "cropped_boxes"

# Load the YOLO model
model = YOLO(model_path)

# Create a directory to save cropped images
os.makedirs(output_dir, exist_ok=True)

# Perform inference on an image
results = model(input_image_path)

# Get the original image for cropping
original_image = cv2.imread(input_image_path)

# Loop over the detected objects
total_boxes = len(results[0].boxes.xyxy)
print(f"Total boxes detected: {total_boxes}")

for i, box in enumerate(results[0].boxes.xyxy):
    print(f"Processing box {i}...")  # Debugging statement

    x1, y1, x2, y2 = map(int, box)
    confidence = results[0].boxes.conf[i]  # Get confidence score
    class_id = results[0].boxes.cls[i]  # Get class ID

    if confidence > 0.1:  # Threshold for confidence
        # Crop the image using the bounding box coordinates
        cropped_img = original_image[y1:y2, x1:x2]

        # Save the cropped image
        output_path = os.path.join(output_dir, f"box_{i}.jpg")
        cv2.imwrite(output_path, cropped_img)

        # Perform OCR on the cropped image
        text = pytesseract.image_to_string(cropped_img)

        # Clean up the extracted text
        cleaned_text = "\n".join(line.strip().replace(":", "") for line in text.splitlines() if line.strip())

        # Display the cropped image using OpenCV
        cv2.imshow(f'Box {i}', cropped_img)
        cv2.waitKey(1000)  # Wait for 1 second (1000 milliseconds) before closing the window

        # Print the cleaned extracted text
        print(f"Text from box {i}:")
        print(cleaned_text)
        print()  # Print a newline for better readability

cv2.destroyAllWindows()  # Close all OpenCV windows

# Collect the number of characters (or words) from each box's OCR result for the bar chart
box_indices = []
text_lengths = []

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    confidence = results[0].boxes.conf[i]
    if confidence > 0.1:
        cropped_img = original_image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cropped_img)
        cleaned_text = "\n".join(line.strip().replace(":", "") for line in text.splitlines() if line.strip())
        box_indices.append(f'Box {i}')
        text_lengths.append(len(cleaned_text))  # Or use len(cleaned_text.split()) for word count

# Plot vertical bar chart
plt.figure(figsize=(10, 6))
plt.bar(box_indices, text_lengths)
plt.xlabel('Box Index')
plt.ylabel('Text Length (characters)')
plt.title('Text Length per Detected Box')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the chart in the results directory
chart_path = os.path.join("results", "vertical_bar_chart.png")
os.makedirs("results", exist_ok=True)
plt.savefig(chart_path)
plt.close()
print(f"Bar chart saved to {chart_path}")

# Example values for precision, recall, and mAP (replace with your actual values)
precision = 0.92
recall = 0.892
mAP = 0.867

# Plot comparison bar graph of precision, recall, and mAP
metrics = ['Precision', 'Recall', 'mAP']
values = [precision, recall, mAP]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Comparison of Precision, Recall & mAP')
plt.tight_layout()

comparison_chart_path = os.path.join("results", "comparison_bar_chart.png")
plt.savefig(comparison_chart_path)
plt.close()
print(f"Comparison bar chart saved to {comparison_chart_path}")
# Plot and save histogram graph of text lengths
plt.figure(figsize=(8, 5))
plt.hist(text_lengths, bins=10, color='purple', edgecolor='black')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')
plt.title('Histogram of Text Lengths per Detected Box')
plt.tight_layout()

histogram_path = os.path.join("results", "text_length_histogram.png")
plt.savefig(histogram_path)
plt.close()
print(f"Histogram saved to {histogram_path}")

import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
epochs = list(range(190))  # Example for 190 epochs
val_box_loss = [0.55, 0.70, 0.56, 0.52, 0.50, 0.48, 0.46, 0.45, 0.44, 0.43] + [0.42 - i*0.0001 for i in range(180)]  # Replace with real values

# Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, val_box_loss, label='val/box_loss')
plt.xlabel("Epoch")
plt.ylabel("val/box_loss")
plt.title("val/box_loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Sample data (replace these with your actual data)
epochs = list(range(190))
map_50 = [0.3 + 0.005 * min(i, 100) for i in epochs]  # Fake smooth increase
precision = [0.5 + 0.004 * min(i, 100) - 0.02*(i > 120) for i in epochs]  # Fake values
recall = [0.1 + 0.006 * min(i, 100) + 0.01*(i > 100) for i in epochs]  # Fake values

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(epochs, map_50, 'o-', label='mAP@0.5')            # blue circle line
plt.plot(epochs, precision, 'x--', label='Precision')       # orange cross dashed
plt.plot(epochs, recall, 's:', label='Recall')              # green square dotted

# Formatting
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("mAP, Precision, and Recall Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
