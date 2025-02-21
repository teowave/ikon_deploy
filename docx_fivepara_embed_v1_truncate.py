import io
import os
import string
from collections import defaultdict, OrderedDict
import json
import cv2
#import extract_msg
from tqdm import tqdm
from openai import OpenAI  # Updated import based on your example
from openai import OpenAIError
import logging
from PIL import Image
import base64
import numpy as np
import pytesseract
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from builtins import input
import config
import datetime
from striprtf.striprtf import rtf_to_text  # Added for RTF handling
import tiktoken  # Added for token counting

from main_import_files_to_db_v1 import create_tb_page_paragraphs_if_not_exist

# Initialize the OpenAI client globally
_openai_client = None

# Database Configuration
PG_DB_NAME = config.PG_DB_NAME
PG_DB_USER = config.PG_DB_USER
PG_DB_PASSWORD = config.PG_DB_PASSWORD
PG_DB_HOST = config.PG_DB_HOST
PG_DB_PORT = config.PG_DB_PORT

BASE_PATH = config.BASE_PATH  # Ensure BASE_PATH is defined in your config

# Configure logging to include the function name
logging.basicConfig(
    level=logging.WARNING,  # Consider changing to INFO in production
    format='%(asctime)s-%(levelname)s-[%(funcName)s:%(lineno)d]- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress unnecessary logging from external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("trace").setLevel(logging.WARNING)


# MARK: create_db_tables

def sanitize_string(s):
    """
    Remove NUL characters from a string.

    Args:
        s (str): The input string.

    Returns:
        str: The sanitized string without NUL characters.
    """
    if s is None:
        return None
    return s.replace('\x00', '')

def get_openai_client():
    """
    Initialize and return the OpenAI client.
    
    Returns:
        OpenAI: An instance of the OpenAI client.
    
    Raises:
        RuntimeError: If the OpenAI client fails to initialize.
    """
    global _openai_client
    if _openai_client is None:
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables.")
            _openai_client = OpenAI(api_key=api_key)
            logging.info('OpenAI client initialized from environment variable.')
        except OpenAIError as error:
            logging.error('Error initializing OpenAI client: %s', str(error))
            raise RuntimeError("Failed to initialize the OpenAI client") from error
        except Exception as e:
            logging.error('Unexpected error initializing OpenAI client: %s', str(e))
            raise RuntimeError("Failed to initialize the OpenAI client") from e
    return _openai_client

# MARK: get_embedding_of
def get_embedding_from_api(text, ai_api_model, max_tokens=8090):
    """
    Fetch the embedding for a given text using OpenAI's API.

    Args:
        text (str): The input text to embed.
        ai_api_model (str): The model to use for embedding.
        max_tokens (int): Maximum tokens allowed for the model.

    Returns:
        str: JSON-encoded embedding vector or None if failed.
    """
    if not text or text.strip() == "":
        logging.warning("Received empty or None text for embedding. Skipping.")
        return None

    # Initialize token encoder for the specified model
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Adjust encoding as per model
    except Exception as e:
        logging.error("Error initializing tiktoken encoding: %s", str(e))
        return None

    # Count tokens in the text
    token_count = len(encoding.encode(text))
    logging.debug(f"Token count for text: {token_count}")

    # Define model's maximum tokens
    model_max_tokens = 8192  # As per the error message

    # Calculate available tokens for the prompt (embedding)
    # Since embeddings don't require completion tokens, you can use the entire limit
    # However, to be safe, reserve some buffer (e.g., 100 tokens)
    buffer_tokens = 100
    available_tokens = model_max_tokens - buffer_tokens

    if token_count > available_tokens:
        logging.warning(f"Text exceeds maximum token limit ({model_max_tokens}). Truncating.")
        # Truncate the text to fit within the token limit
        truncated_text = encoding.decode(encoding.encode(text)[:available_tokens])
        logging.debug("Text has been truncated to fit the token limit.")
    else:
        truncated_text = text

    client = get_openai_client()
    if client is None:
        logging.error('Failed to get OpenAI client')
        return None

    try:
        response = client.embeddings.create(input=[truncated_text], model=ai_api_model)
        embedding = response.data[0].embedding
        return json.dumps(embedding)
    except OpenAIError as openai_error_embedd:
        logging.error('Error fetching embedding: %s', str(openai_error_embedd))
        return None
    except Exception as e:
        logging.error('Unexpected error fetching embedding: %s', str(e))
        return None

def update_paragraph_embeddings(conn, ai_api_model):
    """
    Update the embeddings of paragraphs in the database that have not been processed.

    Args:
        conn (psycopg2.connection): The database connection.
        ai_api_model (str): The OpenAI model to use for embeddings.
    """
    cursor = conn.cursor()

    try:
        cursor.execute("""
                        SELECT id, five_para_slid_win_text 
                            FROM tb_page_paragraphs 
                            WHERE five_para_text_embedding IS NULL 
                            AND NOT processed
                            ORDER BY id DESC
                            LIMIT 10000
                       """)
        rows = cursor.fetchall()
    except psycopg2.Error as e:
        logging.error("Error fetching paragraphs for embedding: %s", e)
        return

    if not rows:
        logging.info("No paragraph embeddings need updating.")
        return

    update_count = 0
    for row_id, text in tqdm(rows, desc="Updating paragraph embeddings"):
        if not text or text.strip() == "":
            logging.warning(f"Empty text for paragraph {row_id}. Marking as processed.")
            try:
                cursor.execute("""
                                UPDATE tb_page_paragraphs 
                                SET processed = TRUE 
                                WHERE id = %s
                               """, 
                               (row_id,))
            except psycopg2.Error as err:
                logging.error(f"Error marking paragraph {row_id} as processed: {err}")
            continue

        logging.debug("Processing paragraph %s ", row_id)

        embedding_json = get_embedding_from_api(text, ai_api_model)
        if embedding_json is None:
            logging.warning(f"Failed to get embedding for paragraph {row_id}. Will retry in the next run.")
            continue  # Do not mark as processed to allow retry

        try:
            cursor.execute(
                """
                    UPDATE tb_page_paragraphs 
                    SET 
                        five_para_text_embedding = %s::vector, 
                        processed = TRUE 
                    WHERE id = %s
                """,
                (embedding_json, row_id)
            )
            update_count += 1
            if update_count % 10 == 0:
                conn.commit()
        except psycopg2.Error as err:
            logging.error(f"Error updating embedding for paragraph {row_id}: {err}")
            continue

    try:
        conn.commit()
    except psycopg2.Error as err:
        logging.error(f"Error committing transaction: {err}")
    logging.info(f"{update_count} paragraph embeddings updated.")

def main():
    """
    Main function to execute the email embedding process.
    """
    # Database connection parameters
    db_params = {
        'dbname': PG_DB_NAME,
        'user': PG_DB_USER,
        'password': PG_DB_PASSWORD,
        'host': PG_DB_HOST,
        'port': PG_DB_PORT
    }
    conn = None

    try:
        # Establish database connection
        logging.info("Connecting to the database %s...", PG_DB_NAME)
        conn = psycopg2.connect(**db_params)
        logging.info("Database connection established.")

        # Optionally, set autocommit to False (default) to manage transactions manually
        conn.autocommit = False

        # Ensure database tables exist
        logging.info("Ensuring database tables exist...")
        if not create_tb_page_paragraphs_if_not_exist(conn):
            logging.error("Failed to create or verify database tables. Exiting.")
            return

        # Process emails
        logging.info("Processing paras into db...")
        update_paragraph_embeddings(conn, config.MODEL_EMB)
        logging.info("Paras processed successfully!")

    except psycopg2.Error as e:
        logging.error("Database connection error: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
    finally:
        # Close the database connection
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
