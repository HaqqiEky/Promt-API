import re
import emoji


def clean_non_alphanumeric(text):
    # Define a regex pattern to match @USER format
    user_pattern = re.compile(r'@USER')
    
    # Split the text by spaces to handle each word separately
    words = text.split()
    
    # Initialize a list to hold the cleaned words
    cleaned_words = []
    
    # Loop through each word in the text
    for word in words:
        # If the word matches @USER, add it as is
        if user_pattern.fullmatch(word):
            cleaned_words.append(word)
        else:
            # Define a regex pattern to match non-alphanumeric characters
            pattern = re.compile(r'[\W_]+')  # Matches any non-alphanumeric character or underscore
            # Replace non-alphanumeric characters with a space
            cleaned_word = re.sub(pattern, ' ', word)
            cleaned_words.append(cleaned_word)
    
    # Join the cleaned words back into a single string with spaces
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def clean_text(text):
    # Define regex pattern to match unwanted unicode characters
    pattern = re.compile(r'[^\x00-\x7F]+')  # Matches any non-ASCII characters

    # Use sub() method to remove unwanted characters
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def preprocess_text(text):
    # Remove "RT @username:" at the beginning of the text
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError:
            text = text.decode('latin-1')

    text = re.sub(r'^RT ', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    # Remove URLs
    

    # Remove emojis and other non-ASCII symbols
    text = emoji.demojize(text)

    # Convert to lowercase
    text = text.lower()
    text = re.sub(r'https?://\S+', 'HTTPURL', text)
    text = clean_text(text)
    # Remove hashtags along with their content (words starting with # followed by alphanumeric characters)
    text = re.sub(r'#\w+', '', text)

    # Remove mentions (words starting with @ followed by alphanumeric characters)
    text = re.sub(r'@\S+', '@USER', text)

    # Remove non-alphanumeric characters including remaining symbols
    text = clean_non_alphanumeric(text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text