import os
import logging
from tqdm import tqdm
from docx import Document
import psycopg2
import config  # Ensure this module has the necessary configurations
from lxml import etree
import zipfile  # Needed for parsing numbering.xml
from main_import_files_to_db_v1 import create_tb_page_paragraphs_if_not_exist

# -------------------- Configuration --------------------

# Database Configuration
PG_DB_NAME = config.PG_DB_NAME
PG_DB_USER = config.PG_DB_USER
PG_DB_PASSWORD = config.PG_DB_PASSWORD
PG_DB_HOST = config.PG_DB_HOST
PG_DB_PORT = config.PG_DB_PORT

BASE_PATH = config.BASE_PATH  # Ensure BASE_PATH is defined in your config

# Configure logging to include the function name and line number
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO in production
    format='%(asctime)s-%(levelname)s-[%(funcName)s:%(lineno)d]- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress less critical logs from specific libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# -------------------- Helper Functions --------------------

def sanitize_string(s):
    """
    Remove NUL characters and strip whitespace from a string.

    Args:
        s (str): The input string.

    Returns:
        str: The sanitized string without NUL characters and leading/trailing whitespace.
    """
    if s is None:
        return None
    return s.replace('\x00', '').strip()

def is_paragraph_numbered(paragraph):
    """
    Determines if a paragraph has numbering applied.

    Args:
        paragraph (docx.text.paragraph.Paragraph): The paragraph to check.

    Returns:
        bool: True if the paragraph is numbered, False otherwise.
    """
    try:
        p = paragraph._p
        numPr = p.pPr.numPr
        return numPr is not None
    except AttributeError:
        return False

def get_numbering_text(paragraph, numbering_dict):
    """
    Retrieves the numbering text for a given paragraph based on its numbering properties.

    Args:
        paragraph (docx.text.paragraph.Paragraph): The paragraph to retrieve numbering for.
        numbering_dict (dict): Dictionary mapping numId and ilvl to numbering formats.

    Returns:
        str: The numbering text (e.g., "1.", "1.1.") or None if not numbered.
    """
    try:
        p = paragraph._p
        numPr = p.pPr.numPr
        if numPr is None:
            return None
        numId = numPr.numId.val
        ilvl = numPr.ilvl.val
        return numbering_dict.get(str(numId), {}).get(str(ilvl), None)
    except AttributeError:
        return None

def parse_numbering_definitions(docx_path):
    """
    Parses the numbering definitions from a .docx file.

    Args:
        docx_path (str): Full path to the .docx file.

    Returns:
        dict: Dictionary mapping numId and ilvl to numbering formats.
    """
    numbering_dict = {}
    try:
        with zipfile.ZipFile(docx_path) as docx_zip:
            numbering_xml = docx_zip.read('word/numbering.xml')
            tree = etree.fromstring(numbering_xml)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

            # Parse abstract numbering
            abstract_num_dict = {}
            for abstract_num in tree.findall('w:abstractNum', ns):
                abstract_num_id = abstract_num.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}abstractNumId')
                levels = {}
                for lvl in abstract_num.findall('w:lvl', ns):
                    ilvl = lvl.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl')
                    lvl_text_elem = lvl.find('w:lvlText', ns)
                    if lvl_text_elem is not None:
                        lvl_text = lvl_text_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                        levels[ilvl] = lvl_text
                abstract_num_dict[abstract_num_id] = levels

            # Map numId to abstractNumId
            for num in tree.findall('w:num', ns):
                num_id = num.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numId')
                abstract_num_id_elem = num.find('w:abstractNumId', ns)
                if abstract_num_id_elem is not None:
                    abstract_num_id = abstract_num_id_elem.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                    numbering_dict[num_id] = abstract_num_dict.get(abstract_num_id, {})
    except Exception as e:
        logging.error("Error parsing numbering definitions: %s", e)
    return numbering_dict

def extract_paragraphs_from_docx(docx_path):
    """
    Extracts paragraphs from a .docx file and prepares context for each paragraph.

    Args:
        docx_path (str): Full path to the .docx file.

    Returns:
        list of dict: List containing paragraph data with context.
    """
    doc = Document(docx_path)
    paragraphs = []
    numbering_dict = parse_numbering_definitions(docx_path)

    # First, extract all paragraphs
    for idx, para in enumerate(doc.paragraphs, start=1):
        para_text = sanitize_string(para.text)
        if not para_text:
            continue  # Skip empty paragraphs

        paragraphs.append({
            'document_id': None,  # To be set later
            'para_number': idx,  # External paragraph number (unique per document)
            'para_internal_docx_number': get_numbering_text(para, numbering_dict) if is_paragraph_numbered(para) else None,
            'para_text': para_text,
            'five_para_slid_win_text': None  # Placeholder for augmented text
        })

    # Now, augment each paragraph with two previous and two next paragraphs
    for i, para in enumerate(paragraphs):
        context_texts = []

        # Get two previous paragraphs
        if i >= 2:
            context_texts.append(paragraphs[i - 2]['para_text'])
        elif i == 1:
            context_texts.append(paragraphs[i - 1]['para_text'])

        # Get one previous paragraph
        if i >= 1:
            context_texts.append(paragraphs[i - 1]['para_text'])

        # Current paragraph
        context_texts.append(para['para_text'])

        # Get one next paragraph
        if i + 1 < len(paragraphs):
            context_texts.append(paragraphs[i + 1]['para_text'])

        # Get two next paragraphs
        if i + 2 < len(paragraphs):
            context_texts.append(paragraphs[i + 2]['para_text'])
        elif i + 1 < len(paragraphs):
            context_texts.append(paragraphs[i + 1]['para_text'])

        # Join the context texts with spaces or any desired separator
        para['five_para_slid_win_text'] = ' '.join(context_texts)

    return paragraphs

def store_paragraph(cur, para):
    """
    Inserts a single paragraph into the tb_page_paragraphs table.

    Args:
        cur (psycopg2.cursor): Active database cursor.
        para (dict): Dictionary containing paragraph data.

    Raises:
        Exception: If insertion fails.
    """
    try:
        insert_query = """
            INSERT INTO tb_page_paragraphs (
                document_id, para_number, para_internal_docx_number, para_text, five_para_slid_win_text, five_para_text_embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, para_number) DO NOTHING
        """
        data = (
            para['document_id'],
            para['para_number'],
            para['para_internal_docx_number'],
            para['para_text'],
            para['five_para_slid_win_text'],
            None  # Placeholder for five_para_text_embedding
        )
        cur.execute(insert_query, data)
        logging.debug("Inserted paragraph_number %d for document_id %s.", para['para_number'], para['document_id'])
    except Exception as e:
        logging.error("Error inserting paragraph_number %d for document_id %s: %s",
                      para['para_number'], para['document_id'], e)
        raise  # Re-raise the exception to handle it in the main function

# -------------------- Main Processing Function --------------------

def docx_process(conn):
    """
    Process .docx files and insert their paragraphs into the tb_page_paragraphs table.
    Sets is_text_read to TRUE in tb_documents upon successful processing.

    Args:
        conn (psycopg2.connection): Active database connection.

    Returns:
        bool: True if processing is successful, False otherwise.
    """
    if conn is None:
        logging.error("Database connection is not established.")
        return False

    try:
        with conn.cursor() as cur:
            # Retrieve all .docx documents from tb_documents where is_text_read is FALSE
            logging.info("Retrieving unprocessed .docx documents from tb_documents...")
            cur.execute(
                """
                SELECT id, filepath 
                FROM tb_documents 
                WHERE filetype = 'docx' AND is_text_read = FALSE
                """
            )
            docx_documents = cur.fetchall()
            logging.info("Found %d unprocessed .docx documents to process.", len(docx_documents))

            if not docx_documents:
                logging.info("No unprocessed .docx documents found. Exiting docx processing.")
                return True

            # Initialize counters
            processed_count = 0
            skipped_count = 0
            error_count = 0

            # Iterate over each .docx document with progress bar
            for doc in tqdm(docx_documents, desc="Processing .docx files"):
                document_id, filepath = doc

                # Full path to the .docx file
                full_path = os.path.join(BASE_PATH, filepath)  # Adjust base path as needed

                if not os.path.exists(full_path):
                    logging.error("File not found: %s. Skipping.", full_path)
                    error_count += 1
                    continue

                logging.debug('Processing .docx file for document_id %s, full path: %s', document_id, full_path)

                try:
                    # Extract paragraphs from .docx with context
                    paragraphs = extract_paragraphs_from_docx(full_path)
                    if not paragraphs:
                        logging.warning("No valid paragraphs found in %s. Skipping.", full_path)
                        skipped_count += 1
                        continue

                    # Assign document_id to each paragraph
                    for para in paragraphs:
                        para['document_id'] = document_id

                    # Start a transaction
                    with conn:
                        # Insert paragraphs one by one
                        for para in paragraphs:
                            store_paragraph(cur, para)

                        # After all paragraphs are inserted, update is_text_read to TRUE
                        cur.execute(
                            """
                            UPDATE tb_documents
                            SET is_text_read = TRUE
                            WHERE id = %s
                            """,
                            (document_id,)
                        )
                    processed_count += 1
                    logging.debug("Processed and inserted paragraphs for document_id %s.", document_id)

                except Exception as e:
                    logging.error("Error processing file %s: %s", full_path, e)
                    error_count += 1
                    # Do not set is_text_read to TRUE if there's an error
                    continue

            # Final commit (handled by 'with conn:' blocks)
            logging.info("Docx processing completed. Processed: %d, Skipped: %d, Errors: %d.",
                         processed_count, skipped_count, error_count)
            return True

    except Exception as e:
        logging.error("An error occurred during docx processing: %s", e)
        return False

# -------------------- Main Function --------------------

def main():
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
        logging.info("Ensuring tb_page_paragraphs table exists...")
        if not create_tb_page_paragraphs_if_not_exist(conn):
            logging.error("Failed to create or verify tb_page_paragraphs table. Exiting.")
            return

        # Process .docx files
        logging.info("Processing .docx files into tb_page_paragraphs...")
        if docx_process(conn):
            logging.info(".docx files processed successfully!")
        else:
            logging.error(".docx files were not processed correctly or incomplete")

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

# -------------------- Database Table Creation --------------------

# Ensure that the tb_page_paragraphs table includes the five_para_slid_win_text field
# If not, you can create or alter the table as follows:

# CREATE TABLE IF NOT EXISTS tb_page_paragraphs (
#     id SERIAL PRIMARY KEY,
#     document_id INTEGER REFERENCES tb_documents(id),
#     para_number INTEGER NOT NULL,
#     para_internal_docx_number TEXT,
#     para_text TEXT,
#     five_para_slid_win_text TEXT,
#     five_para_text_embedding VECTOR(1536),
#     row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#     CONSTRAINT unique_document_para UNIQUE (document_id, para_number)
# );
