import nltk

print("Attempting to download NLTK 'stopwords'...")
try:
    nltk.data.find('corpora/stopwords')
    print("'stopwords' resource already downloaded.")
except LookupError: # Changed from nltk.downloader.DownloadError for broader compatibility
    print("Downloading 'stopwords' resource...")
    nltk.download('stopwords')
    print("'stopwords' resource downloaded.")
except Exception as e:
    print(f"An error occurred with 'stopwords': {e}")


print("\nAttempting to download NLTK 'punkt'...")
try:
    nltk.data.find('tokenizers/punkt')
    print("'punkt' resource already downloaded.")
except LookupError: # Changed from nltk.downloader.DownloadError
    print("Downloading 'punkt' resource...")
    nltk.download('punkt')
    print("'punkt' resource downloaded.")
except Exception as e:
    print(f"An error occurred with 'punkt': {e}")

print("\nNLTK resource check complete.")