import io 
import os
import string
from collections import defaultdict, OrderedDict
import json
#import extract_msg
from tqdm import tqdm
from openai import OpenAI
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

PG_DB_NAME = config.PG_DB_NAME
PG_DB_USER = config.PG_DB_USER
PG_DB_PASSWORD = config.PG_DB_PASSWORD
PG_DB_HOST = config.PG_DB_HOST
PG_DB_PORT = config.PG_DB_PORT

BASE_PATH = config.BASE_PATH

# Configure logging to include the function name
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s-%(levelname)s-[%(funcName)s:%(lineno)d]- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

#now configure the openai logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

Image.MAX_IMAGE_PIXELS = None  # Remove the limit

def create_database_if_not_exists(db_params):
    # Connect to default database to create new database
    conn = psycopg2.connect(
        dbname='postgres',
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_params['dbname'],))
        exists = cur.fetchone()
        if not exists:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_params['dbname'])))
            logging.info("Database %s created successfully.", db_params['dbname'])
        else:
            logging.info("Database %s already exists.", db_params['dbname'])

    conn.close()

#MARK: create_db_tables
def create_tb_page_paragraphs_if_not_exist(conn):
    """
    Ensures that the tb_page_paragraphs table exists with the required schema.
    Adds the pgvector extension if it's not already present.
    Adds the 'processed' column if it doesn't exist.
    
    Args:
        conn (psycopg2.connection): An active connection to the PostgreSQL database.
    
    Returns:
        bool: True if the table exists or was created successfully, False otherwise.
    """
    try:
        with conn.cursor() as cur:
            # 1. Create pgvector extension if it doesn't exist
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logging.info("pgvector extension ensured.")

            # 2. Create tb_page_paragraphs table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tb_page_paragraphs (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES tb_documents(id),
                    para_number INTEGER NOT NULL,
                    para_internal_docx_number TEXT,
                    para_text TEXT,
                    five_para_slid_win_text TEXT,
                    five_para_text_embedding VECTOR(1536),
                    processed BOOLEAN DEFAULT FALSE,
                    row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_document_para UNIQUE (document_id, para_number)
                )
            """)
            logging.info("tb_page_paragraphs table ensured.")

            # Alter tb_page_paragraphs table to add 'processed' column if it doesn't exist
            # Note: Since 'processed' is already included in the CREATE TABLE statement,
            # this step is redundant for new tables but useful for existing ones.
            cur.execute("""
                ALTER TABLE tb_page_paragraphs
                ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE;
            """)
            logging.info("'processed' column ensured in tb_page_paragraphs.")
            
            

        # Commit all changes
        conn.commit()
        return True
    except psycopg2.Error as e:
        logging.error("Database error during table creation: %s", e)
        conn.rollback()
        return False
    except Exception as e:
        logging.error("Unexpected error during table creation: %s", e)
        conn.rollback()
        return False

def create_tb_email_messages_if_not_exist(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tb_email_messages (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES tb_documents(id),
                subject TEXT,
                sender TEXT,
                recipients TEXT,
                cc TEXT,
                bcc TEXT,
                sent_date TIMESTAMP,
                received_date TIMESTAMP,
                body TEXT,
                parsed_body_json JSONB,
                is_parsed BOOLEAN DEFAULT FALSE,
                is_individual_messages_parsed BOOLEAN DEFAULT FALSE,
                processed BOOLEAN DEFAULT FALSE,
                has_attachments BOOLEAN,
                attachment_names TEXT[],
                body_text_embedding VECTOR(1536),
                row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

         # Add the new column if it doesn't exist
        cur.execute("""
            ALTER TABLE tb_email_messages
            ADD COLUMN IF NOT EXISTS parsed_body_json JSONB,
            ADD COLUMN IF NOT EXISTS is_parsed BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS is_individual_messages_parsed BOOLEAN DEFAULT FALSE;
        """)

        logging.info("Column is_individual_messages_parsed added to tb_email_messages.")
    conn.commit()

    return True

def create_tb_email_individual_messages_if_not_exist(conn):
    """
    Create the tb_email_individual_messages table if it does not exist.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tb_email_individual_messages (
                    id SERIAL PRIMARY KEY,
                    email_message_id INTEGER REFERENCES tb_email_messages(id) ON DELETE CASCADE,
                    message_sequence INTEGER NOT NULL,
                    content_sequence INTEGER NOT NULL,
                    content_type TEXT,
                    content_text TEXT,
                    content_embedding VECTOR(1536),
                    is_embedded BOOLEAN DEFAULT FALSE,
                    row_created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()
            logging.info("Table tb_email_individual_messages is ready.")
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating tb_email_individual_messages table: {e}")
        return False

def create_db_tables_if_not_exist(conn):
    with conn.cursor() as cur:
        # Create vector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
       
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tb_documents (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                filetype TEXT NOT NULL,
                file_size BIGINT,
                file_created_at TIMESTAMP,
                file_modified_at TIMESTAMP,
                file_accessed_at TIMESTAMP,
                has_duplicate BOOLEAN DEFAULT FALSE,
                has_same_extension BOOLEAN DEFAULT FALSE,
                is_local_dir BOOLEAN DEFAULT FALSE,
                is_text_read BOOLEAN DEFAULT FALSE,
                are_pg_imgs_extracted BOOLEAN DEFAULT FALSE,
                is_text_ocrd BOOLEAN DEFAULT FALSE,
                whole_doc_text_from_ocr TEXT,
                row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS tb_pages (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES tb_documents(id),
                page_number INTEGER NOT NULL,
                all_page_text TEXT,
                page_text_outside_table TEXT,
                page_text_embedding VECTOR(1536),
                has_table BOOLEAN DEFAULT FALSE,
                table_text TEXT,
                has_image BOOLEAN DEFAULT FALSE,
                row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_document_page UNIQUE (document_id, page_number)

            )
        """)
        conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS tb_page_images (
                id SERIAL PRIMARY KEY,
                page_id INTEGER REFERENCES tb_pages(id) UNIQUE,
                table_struct_p_l_old_style JSONB,
                table_struct_cnv_p_l JSONB,
                table_struct_cnv_p_l_i JSONB,
                table_struct_cnv_p_l_i_n JSONB,
                table_struct_cnv_p_l_i_n_cy JSONB,
                table_struct_cnv_p_l_i_n_cy_ce JSONB,
                table_struct_cnv_p_l_i_n_cy_ce_tb JSONB,
                table_struct_cnv_p_l_i_n_cy_ce_tb_ord JSONB,
                table_struct_cnv_p_l_i_n_cy_ce_tb_ord_nest JSONB,
                are_cell_images_extracted BOOLEAN DEFAULT FALSE,
                page_image_base64 TEXT,
                page_image_without_tables_base64 TEXT,
                row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_page_id UNIQUE (page_id)
            )
        """)
        conn.commit()

        # Create tb_table_cells table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tb_table_cells (
                id SERIAL PRIMARY KEY,
                page_image_id INTEGER REFERENCES tb_page_images(id) ON DELETE CASCADE,
                table_id TEXT NOT NULL,
                cell_id TEXT NOT NULL,
                cell_label TEXT,
                has_table BOOLEAN DEFAULT FALSE,
                contained_table_id TEXT,
                contained_table_bounding_box JSONB,
                cell_image_base64 TEXT NOT NULL,
                cell_image_no_table_base64 TEXT,
                cell_ocr_text TEXT,
                embedd_cell_ocr_text VECTOR(1536),
                row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (page_image_id, table_id, cell_id)
            )
        """)
        conn.commit()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS tb_page_sections (
                id SERIAL PRIMARY KEY,
                page_image_id INTEGER REFERENCES tb_page_images(id),
                section_number INTEGER,
                section_type TEXT,
                table_structure JSONB,
                row_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_page_section UNIQUE (page_image_id, section_number)
            );
        """)
        conn.commit()

        
    create_tb_page_paragraphs_if_not_exist(conn)
    create_tb_email_messages_if_not_exist(conn)
    create_tb_email_individual_messages_if_not_exist(conn)
    conn.commit()

    return True

#MARK: finished db work

#MARK: start with docs
def collect_documents(base_path):
    documents = defaultdict(list)
    for root, dirs, files in os.walk(base_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                logging.debug("Skipping hidden file: %s", os.path.join(root, file))
                continue
            
            try:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_path)
                file_type = os.path.splitext(file)[1].lower()[1:]
                base_name = os.path.splitext(file)[0]
                
                # Gather file metadata
                stat_info = os.stat(full_path)
                file_size = stat_info.st_size
                file_created_at = datetime.datetime.fromtimestamp(stat_info.st_ctime)
                file_modified_at = datetime.datetime.fromtimestamp(stat_info.st_mtime)
                file_accessed_at = datetime.datetime.fromtimestamp(stat_info.st_atime)
                
                documents[base_name].append((
                    file, 
                    relative_path, 
                    file_type,
                    file_size,
                    file_created_at,
                    file_modified_at,
                    file_accessed_at
                ))
            except Exception as e:
                logging.error("Error processing file %s: %s", full_path, e)
    return documents

def insert_documents(conn, documents):
    inserted_count = 0
    skipped_count = 0
    
    with conn.cursor() as cur:
        for base_name, files in documents.items():
            has_duplicate = len(files) > 1
            has_same_extension = len(set(doc[2] for doc in files)) == 1 if has_duplicate else False
            dir_groups = defaultdict(list)
            for doc in files:
                dir_path = os.path.dirname(doc[1])
                dir_groups[dir_path].append(doc)
            
            for doc in files:
                filename, filepath, filetype, file_size, file_created_at, file_modified_at, file_accessed_at = doc
                is_local_dir = len(dir_groups[os.path.dirname(filepath)]) > 1
                
                # Check if the document already exists in the database
                cur.execute(
                    "SELECT id FROM tb_documents WHERE filepath = %s",
                    (filepath,)
                )
                existing_doc = cur.fetchone()
                
                if not existing_doc:
                    # Document doesn't exist, insert it
                    cur.execute(
                        sql.SQL("""
                        INSERT INTO tb_documents (
                            filename,
                            filepath,
                            filetype,
                            file_size,
                            file_created_at,
                            file_modified_at,
                            file_accessed_at,
                            has_duplicate,
                            has_same_extension,
                            is_local_dir
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """),
                        (
                            filename, 
                            filepath, 
                            filetype, 
                            file_size, 
                            file_created_at, 
                            file_modified_at, 
                            file_accessed_at,
                            has_duplicate, 
                            has_same_extension, 
                            is_local_dir
                        )
                    )
                    inserted_count += 1
                    logging.debug("Inserted new document: %s", filepath)
                else:
                    skipped_count += 1
                    logging.debug("Document already exists, skipping: %s", filepath)
    
        conn.commit()  # Commit once after all inserts
    logging.info("Total documents inserted: %d", inserted_count)
    logging.info("Total documents skipped: %d", skipped_count)
    
    return True


def sanitize_string(s):
    if s is None:
        return None
    return s.replace('\x00', '')

# MARK: main
def main():
    base_path = BASE_PATH
    #base_path = 'data/source_pdfs/'
    
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
        # Create database if not exists
        logging.info("Creating database if not exists...")
        create_database_if_not_exists(db_params)

        # Collect documents
        logging.info("Collecting documents...")
        documents = collect_documents(base_path)
        logging.info("In dir %s found %s documents", base_path, sum(len(files) for files in documents.values()))

        # Connect to the database
        logging.info("Connecting to the database %s", PG_DB_NAME)
        conn = psycopg2.connect(**db_params)

        # Create table if not exists
        logging.info("Creating tables if not exist...")
        create_db_tables_if_not_exist(conn)

        # Insert documents
        logging.info("Inserting docs into db...")
        if insert_documents(conn, documents):
            logging.info("Docs name extraction completed successfully!")
        else:
            logging.info("Document name extraction was not performed.")

    except (psycopg2.Error, OSError) as e:
        logging.error("An error occurred: %s", e)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
