from ocr_system import TextRecognitionSystem

ocr = TextRecognitionSystem()

IMAGE = '../test_images/m06-091.png'

words, line_indices = ocr.process_image(IMAGE)



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

# Print the final string
print(final_string)