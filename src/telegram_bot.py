import telebot
from telebot.types import Message
from PIL import Image
import io
from ocr_system import TextRecognitionSystem

# Initialize your OCR system
ocr = TextRecognitionSystem()

# Bot token
bot = telebot.TeleBot('TOKEN')

# Group chat id
GROUP_CHAT_ID = 'ID'

# Define the image handler
@bot.message_handler(content_types=['photo'])
def handle_image(message: Message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    # Convert bytes to an image
    image = Image.open(io.BytesIO(downloaded_file))
    
    # Process the image with OCR
    words, line_indices = ocr.process_image(image)

    # Group the words by line
    words_by_line = []
    current_line = []
    current_index = line_indices[0]

    for word, index in zip(words, line_indices):
        if index == current_index:
            current_line.append(word)
        else:
            words_by_line.append(current_line)
            current_line = [word]
            current_index = index

    # Append the last line
    words_by_line.append(current_line)

    # Join words by line and then join lines with newline character
    lines = [' '.join(line) for line in words_by_line]
    final_string = '\n'.join(lines)

    # Send the extracted text back to the user
    bot.reply_to(message, final_string)

# Start the bot
bot.polling()
