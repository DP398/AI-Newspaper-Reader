import os
import pytesseract
from PIL import Image
from gtts import gTTS

def ocr_english_image(image_path, folder_path):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Open the image file
    image = Image.open(image_path)
    
    # Perform OCR with English language support
    text = pytesseract.image_to_string(image, lang='eng')
    
    # Save the OCR result to a text file inside the folder
    output_text_path = os.path.join(folder_path, 'ocr_output.txt')
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    return output_text_path, text

def convert_text_to_audio(text, output_audio_folder):
    # Create folder if it doesn't exist
    if not os.path.exists(output_audio_folder):
        os.makedirs(output_audio_folder)
    
    # Convert text to audio using gTTS (English)
    tts = gTTS(text=text, lang='en')
    audio_output_path = os.path.join(output_audio_folder, 'ocr_output_audio.mp3')
    
    # Save the audio file
    tts.save(audio_output_path)
    
    return audio_output_path

# Example usage

# Path to the image for OCR
image_path = r'D:\newspaper_bott_deployed_Paid\English1.jpg'

# Folder where OCR text file will be stored
folder_path = r'D:\newspaper_bott_deployed_Paid\ocr_results'

# Folder where audio file will be stored
audio_folder_path = r'D:\newspaper_bott_deployed_Paid\audio_results'

# Perform OCR and save the text
output_text_file, ocr_text = ocr_english_image(image_path, folder_path)
print(f"Text saved at: {output_text_file}")
print(f"OCR Text: {ocr_text}")

# Convert the text to audio and save it
audio_file_path = convert_text_to_audio(ocr_text, audio_folder_path)
print(f"Audio saved at: {audio_file_path}")