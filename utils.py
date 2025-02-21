# utils.py
import tiktoken
import logging

def count_tokens_for_text(text, model):
    """Returns the number of tokens used by a given text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
