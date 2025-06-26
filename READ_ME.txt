Procedure

Run the file "Final code" in Python Supporting Software such as Studio Visio Code.

CHANGE THE MODEL_PATH, INPUT_IMAGE_PATH AND OUTPUT_DIR.


1.	Step 1: Search the Bot on Telegram
	Open the Telegram app.
	In the search bar, type the bot’s name (newspaper_bott).
	Select the bot from the search results.

2.	Step 2: Start the Bot
	Tap on the Start button in the chat window to initiate the conversation with the bot.
	The bot replies with a welcome message and instructions.

3.	Step 3: Set the Language
	Type or click the command /set language.
	The bot presents two options: English and Hindi.
	Tap the language corresponding to the newspaper article you will upload.
	Example: Tap Hindi if your newspaper is in Hindi.

4.	Step 4: Upload a Newspaper Image
	Send a clear image of the newspaper as a file or image for better OCR accuracy:
	Tap the attachment icon (📎).
	Choose File → Select the image from your phone or computer.
	The bot acknowledges that it received the file and starts processing.

5.	Step 5: Text Detection and Cropping
	The bot uses a modified YOLOv8 model to detect different article blocks in the image.
	Each detected article region is cropped and sent back to you as an image.
	The bot replies: “Processing detected news article”.

6.	Step 6: Optical Character Recognition (OCR)
	The cropped article image is passed through Tesseract OCR.
	It extracts the printed text based on your selected language.

7.	Step 7: Text-to-Speech Conversion
	The extracted text is converted into speech using Google Text-to-Speech (gTTS).
	An audio file (MP3 format) is generated.

8.	Step 8: Receive the Audio Output
	The bot sends the audio file back to your chat.
	You can play, download, or share the article’s audio directly within Telegram.
