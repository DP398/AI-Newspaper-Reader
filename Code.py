from gtts import gTTS
from telegram import Update
from telegram.ext import Updater, MessageHandler, CommandHandler, Filters, CallbackContext
from PIL import Image
import pytesseract
import base64
import os
import cv2
import google.generativeai as genai
from ultralytics import YOLO
import pygame  # For playing audio on the monitor

# Initialize the YOLO model 
model_path = "best_30_epochs.pt"  # Update this to your local YOLO model path
output_dir = "cropped_boxes"
model = YOLO(model_path)
os.makedirs(output_dir, exist_ok=True)

# Set Tesseract path for Hindi
pytesseract.pytesseract.tesseract_cmd = r"D:\MAJOR_PROJECT\test\Tesseract-OCR\tesseract.exe"

# Initialize pygame for audio playback on the system
pygame.mixer.init()

# Function to perform inference using YOLO, crop objects, and extract text with Tesseract
def extract_text_from_image(image_path, update, context):  # Added update and context parameters
    original_image = cv2.imread(image_path)

    # Perform inference using YOLO to detect text-containing regions
    results = model(image_path)
    
    extracted_texts = []
    
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        confidence = results[0].boxes.conf[i]  # Confidence score

        if confidence > 0.1:
            cropped_img = original_image[y1:y2, x1:x2]
            cropped_img_path = os.path.join(output_dir, f"box_{i}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)

            # Perform OCR using Tesseract
            img = Image.open(cropped_img_path)
            custom_config = r'--oem 3 --psm 6 -l hin'
            text = pytesseract.image_to_string(img, config=custom_config).strip()

            extracted_texts.append(text)

            # Show cropped images on the monitor
            cv2.imshow(f'Cropped Image {i}', cropped_img)
            cv2.waitKey(0)  # Wait until a key is pressed to proceed

            # Send image back to Telegram
            send_image_to_telegram(update, context, cropped_img_path)  # Added update and context

    return "\n\n".join(extracted_texts)

# Function to handle the starting message
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hi! I am the Newspaper Reader Bot. Just send me a newspaper image or text, "
        "and I will convert the text to a response for you. Don't use any special characters as "
        "I have to convert it into Hindi audio, saying article 1 heading and text all in Hindi, then move to article 2."
    )

start_handler = CommandHandler('start', start)

# Function to handle image input, process using YOLO + Tesseract, and respond
def process_image(update: Update, context: CallbackContext):
    photo = update.message.photo[-1]
    file_id = photo.file_id
    file = context.bot.get_file(file_id)
    file.download('input_image.jpg')

    # Extract text using YOLO + Tesseract
    extracted_text = extract_text_from_image('input_image.jpg', update, context)  # Added update and context
    response = process_text_and_generate_response(extracted_text)

    send_audio_response(update, context, response)

# Function to process and generate a response using Google Gemini
def process_text_and_generate_response(input_text):
    genai.configure(api_key='AIzaSyB0gluTaJ9Hfv5nBt1q43lRV_hCQ11KtUY')  # Make sure to replace with your key
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = "You will be provided with Hindi newspaper text. Return each article in correct order, only in Hindi, without skipping any part."

    response = model.generate_content(prompt + "\n\n" + input_text)

    print(response.text)
    return response.text

# Function to send the response as Hindi audio using TTS and play it on the monitor
def send_audio_response(update: Update, context: CallbackContext, response: str):
    tts = gTTS(text=response, lang='hi', slow=False)
    audio_file = "response.mp3"
    tts.save(audio_file)

    # Send audio to Telegram
    with open(audio_file, 'rb') as audio:
        update.message.reply_audio(audio=audio)

    # Play the audio on the monitor
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():  # Wait until the audio finishes playing
        pygame.time.Clock().tick(10)

    os.remove(audio_file)

# Function to send cropped images to Telegram
def send_image_to_telegram(update: Update, context: CallbackContext, image_path):
    with open(image_path, 'rb') as img:
        update.message.reply_photo(photo=img)

# Handler for images
image_handler = MessageHandler(Filters.photo & ~Filters.command, process_image)

# Initialize the Telegram bot
updater = Updater(token="6364780215:AAF6ehmZzkl7mEeUECg6S3i6Zh5sY7ab5_0", use_context=True)
dispatcher = updater.dispatcher

# Add handlers to the dispatcher
dispatcher.add_handler(start_handler)
dispatcher.add_handler(image_handler)

# Start polling for Telegram messages
updater.start_polling()
updater.idle()
