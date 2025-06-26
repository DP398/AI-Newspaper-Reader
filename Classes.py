import os
import cv2
import pytesseract
import matplotlib.pyplot as plt
from gtts import gTTS
from ultralytics import YOLO
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import logging

# Paths
model_path = "best_30_epochs.pt"  # Update this to your local model path
output_dir = "cropped_boxes"

# Initialize YOLO model
model = YOLO(model_path)

# Set up logging for the bot
logging.basicConfig(level=logging.INFO)

def set_language(update: Update, context: CallbackContext) -> None:
    reply_keyboard = [['English', 'Hindi']]
    update.message.reply_text(
        "Please choose the language of the newspaper:",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

def handle_language_choice(update: Update, context: CallbackContext) -> None:
    choice = update.message.text.lower()
    
    if choice in ['english', 'hindi']:
        context.user_data['language'] = choice
        update.message.reply_text(f"Great! You've selected {choice.title()}. Now, please upload the newspaper image as a file for better results.")
    else:
        update.message.reply_text("Invalid choice. Please select either 'English' or 'Hindi'.")

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "Hi! Send me an image of a newspaper, and I'll process it. "
        "For better reading results, please upload the image as a file instead of a photo. "
        "To start, please set the language of the newspaper by using the /setlanguage command."
    )

def handle_image(update: Update, context: CallbackContext) -> None:
    if 'language' not in context.user_data:
        update.message.reply_text("Please choose the newspaper language first by using /setlanguage.")
        return
    
    image = update.message.photo[-1].get_file()
    input_image_path = os.path.join(output_dir, "D:\major_project_python - Copy\major_project_python - Copy\OIP.jpeg")
    image.download(input_image_path)
    
    process_image(update, context, input_image_path)

def handle_document(update: Update, context: CallbackContext) -> None:
    if 'language' not in context.user_data:
        update.message.reply_text("Please choose the newspaper language first by using /setlanguage.")
        return
    
    file = update.message.document
    if file.mime_type.startswith("image/"):
        input_image_path = os.path.join(output_dir, file.file_name)
        file = file.get_file()
        file.download(input_image_path)
        
        process_image(update, context, input_image_path)
    else:
        update.message.reply_text("Please upload a valid image file.")

def process_image(update: Update, context: CallbackContext, input_image_path: str) -> None:
    try:
        results = model(input_image_path)
        original_image = cv2.imread(input_image_path)
        extracted_texts = []
        selected_language = context.user_data.get('language', 'hindi')
        
        classified_results = {}

        for i, box in enumerate(results[0].boxes.xyxy):
            try:
                x1, y1, x2, y2 = map(int, box)
                confidence = results[0].boxes.conf[i]
                class_id = int(results[0].boxes.cls[i])
                class_name = model.names[class_id]

                if confidence > 0.1:
                    cropped_img = original_image[y1:y2, x1:x2]
                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    
                    if selected_language == 'english':
                        text = pytesseract.image_to_string(cropped_img_rgb, lang='eng')
                    else:
                        text = pytesseract.image_to_string(cropped_img_rgb, lang='hin')
                    
                    cleaned_text = "\n".join(line.strip().replace(":", "") for line in text.splitlines() if line.strip())
                    extracted_texts.append(cleaned_text)
                    
                    if class_name not in classified_results:
                        classified_results[class_name] = []
                    classified_results[class_name].append(cleaned_text)
            except Exception as e:
                print(f"Error processing box {i}: {e}")
        
        output_classified_path = os.path.join(output_dir, "classified_news.jpg")
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig(output_classified_path)
        
        context.bot.send_photo(chat_id=update.message.chat_id, photo=open(output_classified_path, 'rb'))
        
        for category, texts in classified_results.items():
            update.message.reply_text(f"Category: {category}\n" + "\n".join(texts))
            
    except Exception as e:
        print(f"Error processing image: {e}")
        update.message.reply_text("There was an error processing the image.")

def main() -> None:
    TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
    updater = Updater(TELEGRAM_BOT_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("setlanguage", set_language))
    dp.add_handler(MessageHandler(Filters.text & Filters.regex('^(English|Hindi)$'), handle_language_choice))
    dp.add_handler(MessageHandler(Filters.photo, handle_image))
    dp.add_handler(MessageHandler(Filters.document.category("image"), handle_document))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    main()
