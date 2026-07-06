import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Cleans the input text by normalizing unicode characters, removing non-ASCII characters,
    and replacing multiple whitespace characters with a single space.
    """
    # 1. Normalize encoding
    text = unicodedata.normalize("NFC", text)
    
    # Remove invisible control characters (except for whitespace)
    text = re.sub(r"[\x00-\x1F\x7F]", "", text) # matches all ASCII control characters, including NUL (0) through US (31) and DEL (127).

    # 2. Remove boilerplate text (e.g., headers, footers, disclaimers)
    boilerplate_patterns = [
        r"(?i)copyright\s+@?\s*\d{4}",  
        r"(?i)all rights reserved.*",  
        r"(?i)previous page.*",
        r"(?i)next page.*",
        r"(?i)cookie policy.*",  
        r"(?i)privacy policy.*",
        r"(?i)subscribe now.*",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
        
    # 3. Normalize whitespace
    # replace multiple whitespace characters (including newlines) with a single space
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text) 
    
    # collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)
    
    # trim spaces around newlines
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    
    # 4. Remove junk chars
    text = re.sub(r"[^\S\r\n]+", " ", text)  # remove non-ASCII characters

    # 5. Standardize punctuation 
    text = re.sub(r"[“”]", '"', text)  # replace fancy quotes with standard quotes
    text = re.sub(r"[‘’]", "'", text)  # replace fancy apostrophes with standard ones   
    
    return text.strip()