import os
import pytesseract
from PIL import Image
from gtts import gTTS

def ocr_hindi_image(image_path, folder_path):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Open the image file
    image = Image.open(image_path)
    
    # Perform OCR with Hindi language support
    text = pytesseract.image_to_string(image, lang='hin')
    
    # Save the OCR result to a text file inside the folder
    output_text_path = os.path.join(folder_path, 'ocr_output.txt')
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    return output_text_path, text

def convert_text_to_audio(text, output_audio_folder):
    # Create folder if it doesn't exist
    if not os.path.exists(output_audio_folder):
        os.makedirs(output_audio_folder)
    
    # Convert text to audio using gTTS
    tts = gTTS(text=text, lang='hi')
    audio_output_path = os.path.join(output_audio_folder, 'ocr_output_audio.mp3')
    
    # Save the audio file
    tts.save(audio_output_path)
    
    return audio_output_path

# Example usage

# Path to the image for OCR
image_path = r'D:\Detection-and-recognition-of-multi-column-text-in-Hindi-newspaper--main\English1.jpg

# Folder where OCR text file will be stored
folder_path = r'D:\Detection-and-recognition-of-multi-column-text-in-Hindi-newspaper--main\ocr_results'

# Folder where audio file will be stored
audio_folder_path = r'D:\Detection-and-recognition-of-multi-column-text-in-Hindi-newspaper--main\audio_results'

# Perform OCR and save the text
output_text_file, ocr_text = ocr_hindi_image(image_path, folder_path)
print(f"Text saved at: {output_text_file}")
print(f"OCR Text: {ocr_text}")

# Convert the text to audio and save it
audio_file_path = convert_text_to_audio(ocr_text, audio_folder_path)
print(f"Audio saved at: {audio_file_path}")
