import os
import cv2
from gtts import gTTS
from PIL import Image
import pytesseract
from telegram import Update
from telegram.ext import Updater, MessageHandler, CommandHandler, Filters, CallbackContext
from ultralytics import YOLO

# Initialize YOLO model
model_path = "best_30_epochs.pt"  # Update this to your local YOLO model path
output_dir = "cropped_boxes"
model = YOLO(model_path)
os.makedirs(output_dir, exist_ok=True)

# Set Tesseract path for Hindi OCR
pytesseract.pytesseract.tesseract_cmd = r"D:\MAJOR_PROJECT\test\Tesseract-OCR\tesseract.exe"

# Function to perform inference with YOLO, crop objects, extract text, generate audio, and display images
def extract_text_from_image(image_path, update, context):
    original_image = cv2.imread(image_path)

    # Perform inference using YOLO to detect text-containing regions
    results = model(image_path)
    
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        confidence = results[0].boxes.conf[i]  # Confidence score

        if confidence > 0.1:  # Consider boxes with sufficient confidence
            # Crop the detected region
            cropped_img = original_image[y1:y2, x1:x2]
            cropped_img_path = os.path.join(output_dir, f"cropped_box_{i}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)

            # Display the cropped image on the monitor
            cv2.imshow(f'Cropped Box {i}', cropped_img)
            cv2.waitKey(500)  # Show the image for 500 ms (adjust as needed)
            cv2.destroyWindow(f'Cropped Box {i}')

            # Perform OCR to extract text from the cropped image
            img = Image.open(cropped_img_path)
            custom_config = r'--oem 3 --psm 6 -l hin'  # Hindi language config
            text = pytesseract.image_to_string(img, config=custom_config).strip()

            # Generate and save separate audio file for each extracted text
            audio_file_path = f"audio_box_{i}.mp3"
            generate_audio_for_text(text, audio_file_path)

            # Send the cropped image and corresponding audio back to Telegram
            send_image_and_audio_to_telegram(update, context, cropped_img_path, audio_file_path)

# Function to generate audio for a given text using gTTS
def generate_audio_for_text(text, audio_file_path):
    if text.strip():  # Ensure text is not empty
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(audio_file_path)

# Function to send cropped images and corresponding audios to Telegram
def send_image_and_audio_to_telegram(update: Update, context: CallbackContext, image_path, audio_file_path):
    # Send the cropped image
    with open(image_path, 'rb') as img:
        update.message.reply_photo(photo=img)

    # Send the corresponding audio
    with open(audio_file_path, 'rb') as audio:
        update.message.reply_audio(audio=audio)

# Start command handler
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hi! I am the Newspaper Reader Bot. Send me a newspaper image, and I will extract text from it, "
        "convert it into Hindi audio, and send the corresponding cropped images and audios back to you."
    )

# Function to handle image input
def process_image(update: Update, context: CallbackContext):
    photo = update.message.photo[-1]  # Get the highest resolution photo sent by the user
    file_id = photo.file_id
    file = context.bot.get_file(file_id)
    file.download('input_image.jpg')  # Save the photo locally

    # Extract text from image and generate corresponding audios
    extract_text_from_image('input_image.jpg', update, context)

# Initialize the Telegram bot
def main():
    updater = Updater(token="6364780215:AAF6ehmZzkl7mEeUECg6S3i6Zh5sY7ab5_0", use_context=True)
    dispatcher = updater.dispatcher

    # Add start command handler
    dispatcher.add_handler(CommandHandler('start', start))

    # Add image handler
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, process_image))

    # Start polling Telegram for messages
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
