from flask import Flask, render_template, request, jsonify, session, send_from_directory,url_for,send_file
from flask_session import Session
import psycopg2
from psycopg2.extras import RealDictCursor
import config
import logging
from openai import OpenAI, OpenAIError
import os
import sys
import json
import tiktoken
import mimetypes
#from docx2pdf import convert
#import tempfile
import subprocess
from utils import count_tokens_for_text
from waitress import serve
from logging.handlers import TimedRotatingFileHandler
app = Flask(__name__)

DEBUG=False

# Define the path for  temporary files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_files')

# Ensure the temporary directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# **Set the SECRET_KEY from config.py**
app.secret_key = config.SECRET_KEY
# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'  # we can also use 'redis', 'mongodb', etc.
app.config['SESSION_FILE_DIR'] = 'User/teo/ikon_py/session/'  # Specify if using filesystem
app.config['SESSION_PERMANENT'] = False  # Sessions are not permanent by default

Session(app)

# Configure logging to include the function name and rotating log file
log_directory = r"C:\project\LogFiles"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-[%(funcName)s:%(lineno)d]- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        TimedRotatingFileHandler(
            filename=os.path.join(log_directory, 'ikon_app.log'),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
# Configure openai logging
logging.getLogger('openai').setLevel(logging.WARNING)  # Set OpenAI log level to WARNING

def get_db_connection():
    conn = psycopg2.connect(
        dbname=config.PG_DB_NAME,
        user=config.PG_DB_USER,
        password=config.PG_DB_PASSWORD,
        host=config.PG_DB_HOST,
        port=config.PG_DB_PORT
    )
    return conn

@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 8
    offset = (page - 1) * per_page
    file_types = request.args.getlist('file_type') or ['pdf', 'docx', 'xlsx', 'msg', 'docx_from_pdf']
    
    logging.info("File types for query: %s", file_types)  # Add this line

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
         SELECT DISTINCT d.id, d.filename, d.filepath, d.filetype, d.file_size,
           d.file_created_at, d.file_modified_at, d.file_accessed_at,
           d.has_duplicate, d.has_same_extension, d.is_local_dir,
           CASE WHEN e.id IS NOT NULL THEN TRUE ELSE FALSE END AS is_email
        FROM tb_documents d
        LEFT JOIN tb_pages p ON d.id = p.document_id
        LEFT JOIN tb_page_images pi ON p.id = pi.page_id
        LEFT JOIN tb_email_messages e ON d.id = e.document_id
        WHERE (
            (d.filetype = ANY(%s) AND d.filetype NOT IN ('docx', 'msg')) OR
            (d.filetype = 'docx' AND 'docx' = ANY(%s) AND d.filepath NOT LIKE 'docx_from_pdf/%%') OR
            (d.filetype = 'docx' AND 'docx_from_pdf' = ANY(%s) AND d.filepath LIKE 'docx_from_pdf/%%') OR
            (d.filetype = 'msg' AND 'msg' = ANY(%s))
        )
    """
    
    cur.execute(query + " ORDER BY d.id LIMIT %s OFFSET %s", (file_types, file_types, file_types, file_types, per_page, offset))
    
    files = cur.fetchall()

    # Updated count query to match the main query
    count_query = """
        SELECT COUNT(DISTINCT d.id)
        FROM tb_documents d
        LEFT JOIN tb_pages p ON d.id = p.document_id
        LEFT JOIN tb_page_images pi ON p.id = pi.page_id
        LEFT JOIN tb_email_messages e ON d.id = e.document_id
        WHERE (
            (d.filetype = ANY(%s) AND d.filetype NOT IN ('docx', 'msg')) OR
            (d.filetype = 'docx' AND 'docx' = ANY(%s) AND d.filepath NOT LIKE 'docx_from_pdf/%%') OR
            (d.filetype = 'docx' AND 'docx_from_pdf' = ANY(%s) AND d.filepath LIKE 'docx_from_pdf/%%') OR
            (d.filetype = 'msg' AND 'msg' = ANY(%s))
        )
    """
    cur.execute(count_query, (file_types, file_types, file_types, file_types))
    total = cur.fetchone()['count']

    logging.info(f"Total files matching criteria: {total}")
    logging.info(f"Files retrieved for current page: {len(files)}")


    cur.close()
    conn.close()

    return render_template('index.html', files=files, page=page, per_page=per_page, total=total, selected_types=file_types)

#MARK: file
@app.route('/file/<path:filepath>', methods=['GET'])
def serve_file(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)

    if filepath.endswith('.docx'):
        if filepath.endswith('_ocr.docx'):
            # 'converted_from_pdf' is True
            # Extract the filename from the filepath
            pdf_filename = os.path.basename(filepath)
            # Change the extension to .pdf to get the PDF filename
            pdf_filename = pdf_filename.replace('_ocr.docx','.pdf')

            logging.info("converted pdf filename: %s", pdf_filename)
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Query the tb_documents table for the PDF file with the same filename
            cur.execute("""
                SELECT filepath
                FROM tb_documents
                WHERE filename = %s AND filetype = 'pdf'
            """, (pdf_filename,))
            result = cur.fetchone()
            cur.close()
            conn.close()

            if result:
                # Get the full filepath of the PDF
                pdf_fullpath = os.path.join(config.BASE_PATH, result['filepath'])
                # Serve the PDF file
                return send_file(pdf_fullpath, as_attachment=False, mimetype='application/pdf')
            else:
                logging.error(f"PDF file '{pdf_filename}' not found in the database.")
                return jsonify({'status': 'error', 'message': 'PDF file not found.'}), 404
        else:
            # Handle regular .docx files by converting them to PDF
            try:
                temp_dir = TEMP_DIR

                # Use the original .docx file path
                docx_file_path = os.path.join(config.BASE_PATH, filepath)

                # Convert the .docx to .pdf using LibreOffice
                subprocess.run([
                    '/Applications/LibreOffice.app/Contents/MacOS/soffice',
                    '--headless',
                    '--convert-to', 'pdf',
                    '--outdir', temp_dir,
                    docx_file_path
                ], check=True)

                # Get the output PDF path
                pdf_filename = os.path.splitext(os.path.basename(docx_file_path))[0] + '.pdf'
                temp_pdf_path = os.path.join(temp_dir, pdf_filename)
                temp_pdf_path = temp_pdf_path.replace("\\", "/")
                logging.info(f"temp_pdf_path: {temp_pdf_path}")
                # Serve the generated PDF file
                return send_file(temp_pdf_path, as_attachment=False, mimetype='application/pdf')
            except Exception as e:
                logging.error(f"Error converting .docx to PDF: {e}")
                return jsonify({'status': 'error', 'message': 'Failed to convert document to PDF.'}), 500
    else:
        # Serve other file types directly
        if mime_type is None:
            mime_type = 'application/octet-stream'
        full_path = os.path.join(config.BASE_PATH, filepath)
        full_path = full_path.replace("\\", "/")
        logging.info(f"full_path: {full_path}")
        return send_file(full_path, as_attachment=False, mimetype=mime_type)

# MARK: email    
@app.route('/email/<int:file_id>')
def email_details(file_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT subject, sender, recipients, cc, bcc, sent_date, received_date, body, has_attachments, attachment_names
        FROM tb_email_messages
        WHERE document_id = %s
    """, (file_id,))
    email_data = cur.fetchone()

    cur.close()
    conn.close()

    return jsonify(email_data)

@app.route('/search')
def search_page():
    return render_template('search.html')

#MARK: semantic search num limit
@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    logging.debug("Received semantic search request.")
    
    search_query = request.form.get('search_query')
    file_types = request.form.getlist('file_type')
    logging.info("Received semantic search request with query: %s and file types: %s", search_query, file_types)
    
    if not search_query or not file_types:
        error_msg = "Missing search query or file types."
        logging.error(error_msg)
        return jsonify({'status': 'error', 'message': error_msg}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Generate embedding for the search query
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.embeddings.create(input=[search_query], model=config.MODEL_EMB)
        search_embedding = response.data[0].embedding
        
        # Perform semantic search with limit on number of results
        number_limit_query = """
            WITH
            page_data AS (
                SELECT 
                    d.id, 
                    d.filename, 
                    d.filepath, 
                    d.filetype, 
                    p.page_number AS paragraph_number,
                    p.all_page_text AS text_content,
                    pi.page_image_base64,
                    p.page_text_embedding AS embedding,
                    CASE WHEN d.filepath LIKE 'docx_from_pdf/%%' THEN TRUE ELSE FALSE END AS converted_from_pdf
                FROM tb_documents d
                JOIN tb_pages p ON d.id = p.document_id
                LEFT JOIN tb_page_images pi ON p.id = pi.page_id
                WHERE p.page_text_embedding IS NOT NULL
                  AND LENGTH(p.all_page_text) >= 36
                  AND (
                        (d.filetype = ANY(%s) AND d.filetype NOT IN ('docx', 'msg')) OR
                        (d.filetype = 'docx' AND 'docx' = ANY(%s) AND d.filepath NOT LIKE 'docx_from_pdf/%%') OR
                        (d.filetype = 'docx' AND 'docx_from_pdf' = ANY(%s) AND d.filepath LIKE 'docx_from_pdf/%%')
                      )
            ),
            paragraph_data AS (
                SELECT 
                    d.id, 
                    d.filename, 
                    d.filepath, 
                    d.filetype, 
                    pp.para_number AS paragraph_number,
                    pp.para_text AS text_content,
                    NULL AS page_image_base64,
                    pp.five_para_text_embedding AS embedding,
                    CASE WHEN d.filepath LIKE 'docx_from_pdf/%%' THEN TRUE ELSE FALSE END AS converted_from_pdf
                FROM tb_documents d
                JOIN tb_page_paragraphs pp ON d.id = pp.document_id
                WHERE pp.five_para_text_embedding IS NOT NULL
                  AND LENGTH(pp.para_text) >= 36
                  AND (
                        (d.filetype = ANY(%s) AND d.filetype NOT IN ('docx', 'msg')) OR
                        (d.filetype = 'docx' AND 'docx' = ANY(%s) AND d.filepath NOT LIKE 'docx_from_pdf/%%') OR
                        (d.filetype = 'docx' AND 'docx_from_pdf' = ANY(%s) AND d.filepath LIKE 'docx_from_pdf/%%')
                      )
            ),
            email_data AS (
                SELECT 
                    d.id, 
                    d.filename, 
                    d.filepath, 
                    d.filetype, 
                    em.message_sequence AS paragraph_number,
                    em.content_text AS text_content,
                    NULL AS page_image_base64,
                    em.content_embedding AS embedding,
                    FALSE AS converted_from_pdf
                FROM tb_documents d
                JOIN tb_email_messages e ON d.id = e.document_id
                JOIN tb_email_individual_messages em ON e.id = em.email_message_id
                WHERE em.content_embedding IS NOT NULL
                  AND LENGTH(em.content_text) >= 36
                  AND d.filetype = 'msg' AND 'msg' = ANY(%s)
            )
            SELECT *
            FROM (
                SELECT *, embedding <-> %s::vector AS distance,
                       ROW_NUMBER() OVER (
                           PARTITION BY text_content
                           ORDER BY embedding <-> %s::vector
                       ) AS rn
                FROM (
                    SELECT * FROM page_data
                    UNION ALL
                    SELECT * FROM paragraph_data
                    UNION ALL
                    SELECT * FROM email_data
                ) all_data
            ) sub
            WHERE rn = 1
            ORDER BY distance
            LIMIT 8
        """
    
        # Prepare the parameters for the query
        params = (
            file_types, file_types, file_types,  # For page_data
            file_types, file_types, file_types,  # For paragraph_data
            file_types,                          # For email_data
            search_embedding, search_embedding   # For embedding distances
        )
    
        cur.execute(number_limit_query, params)
        search_results = cur.fetchall()

        # Calculate and log the size of raw search results
        raw_size_bytes = sys.getsizeof(search_results)
        raw_size_kb = raw_size_bytes / 1024
        raw_size_mb = raw_size_kb / 1024
        logging.debug("Size of raw search results: %d bytes (%.2f KB, %.2f MB)", raw_size_bytes, raw_size_kb, raw_size_mb)

        # Calculate the total size of raw text content in bytes
        total_text = " ".join(row['text_content'] for row in search_results if 'text_content' in row)
        total_text_size = len(total_text)
        total_tokens = count_tokens_for_text(total_text, model = config.MODEL_COMPLETIONS)

        # Log the size of the raw text content
        logging.debug("raw size of text content returned by SQL query: %d characters", total_text_size)
        logging.debug("raw Tokens returned by vec dist SQL query: %d ", total_tokens)

        # Reduce the size of search results before saving to the session
        reduced_search_results = [
            {
                'filename': result.get('filename'),
                'paragraph_number': result.get('paragraph_number'),
                'text_content': result.get('text_content')
            } for result in search_results
        ]

        session['last_search_results'] = reduced_search_results



        # Create a new variable for HTML rendering with the required fields
        html_reduced_search_results = [
            {
                'filename': result.get('filename'),
                'file_link': url_for('serve_file', filepath=result.get('filepath')),
                'filetype': result.get('filetype'),
                'paragraph_number': result.get('paragraph_number'),
                'text_content': result.get('text_content'),
                'converted_from_pdf': result.get('converted_from_pdf'),
                'distance': result.get('distance')
            } for result in search_results
        ]

        cur.close()
        conn.close()

        if DEBUG:
            # Calculate and log the size of raw search results
            reduced_size_bytes = sys.getsizeof(reduced_search_results)
            reduced_size_kb = raw_size_bytes / 1024
            reduced_size_mb = raw_size_kb / 1024
            logging.debug("reduced search results: %d bytes (%.2f KB, %.2f MB)", reduced_size_bytes, reduced_size_kb, reduced_size_mb)

            # Convert the reduced results to JSON for token calculation
            reduced_text = " ".join(row['text_content'] for row in reduced_search_results if 'text_content' in row)
            reduced_text_size = len(reduced_text)
            reduced_tokens = count_tokens_for_text(reduced_text, model=config.MODEL_COMPLETIONS)

            # Log the size of the text content and token count
            logging.debug("reduced size of text content returned by SQL query: %d characters", reduced_text_size)
            logging.debug("reduced Tokens returned by vec dist SQL query: %d ", reduced_tokens)
                
        # Render the search results as HTML on the server side
        if not search_results:
            return jsonify({'status': 'success', 'rendered_html': '<p>No results found for your search query.</p>'})

        rendered_html = render_template('search_results.html', results=html_reduced_search_results)
        return jsonify({'status': 'success', 'rendered_html': rendered_html})
    
    except OpenAIError as oe:
        logging.error("OpenAI API error during semantic search: %s", str(oe))
        return jsonify({'status': 'error', 'message': 'OpenAI API error: ' + str(oe)}), 500
    except psycopg2.Error as pe:
        logging.error("Database error during semantic search: %s", str(pe))
        return jsonify({'status': 'error', 'message': 'Database error: ' + str(pe)}), 500
    except (ValueError, TypeError, KeyError) as e:
        logging.error("Unexpected error during semantic search: %s", str(e))
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500

#MARK: summarize
@app.route('/summarize', methods=['POST'])
def summarize():
    search_query = request.json['search_query']
    logging.info("received summarize request for query %s", search_query)
    flask_session_search_results = session.get('last_search_results', [])

    combined_text = "\n".join(result['text_content'] for result in flask_session_search_results)

    prompt = f"""
        Analyze the following content related to a commercial construction project 
        in the UK, focusing on the search query: '{search_query}', 
        results of the query: {combined_text}

        Provide a comprehensive summary addressing the following points:

        1. Key aspects of the project directly related to the search query
        2. Relevant regulations, standards, or legal requirements mentioned
        3. Specific technical details or specifications
        4. Project timeline or milestone information, if available
        5. Potential challenges or risks identified
        6. Stakeholders or parties involved in this aspect of the project

        Format your response in markdown, using numbered paragraphs. Be rigorous and factual, 
        basing your summary strictly on the information provided without inventing or 
        assuming details beyond the given documentation. 
        Include references to all of the processed documents, with paragraph numbers, 
        at the end of the response.

        Conclude with a brief section on recommended further steps or areas requiring 
        additional investigation based on the available information.

        Your response should be thorough yet concise, suitable for a professional project 
        management context.
        """
#here
    # Prepare the messages to be sent to the API
    messages = [
        {"role": "system", "content": "You are a rigorous assistant that summarizes content related to construction projects."},
        {"role": "user", "content": prompt}
    ]

    # Function to count tokens in messages
    def num_tokens_from_messages(messages, model):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logging.warning("Model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 0
        tokens_per_name = 0
        if model.startswith("gpt-3.5-turbo"):
            tokens_per_message = 4  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # If there's a name, the role is omitted
        elif model.startswith("gpt-4"):
            tokens_per_message = 3  # As per OpenAI's token counting rules
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"Token counting not implemented for model {model}.")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Every reply is primed with <|start|>assistant
        return num_tokens

    # Count tokens
    model = config.MODEL_COMPLETIONS
    token_count = num_tokens_from_messages(messages, model)
    logging.info(f"Token count for the request: {token_count}")

    try:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model=config.MODEL_COMPLETIONS,
            messages=messages,
            max_tokens=2750,
            temperature=0
        )
        summary = response.choices[0].message.content.strip()
        return jsonify({'summary': summary})
    except OpenAIError as e:
        logging.error(f"OpenAI API error during summarization: {str(e)}")
        return jsonify({'error': 'An error occurred during summarization'}), 500
    except ValueError as e:
        logging.error(f"Value error during summarization: {str(e)}")
        return jsonify({'error': 'An error occurred during summarization'}), 500
    except KeyError as e:
        logging.error(f"Key error during summarization: {str(e)}")
        return jsonify({'error': 'An error occurred during summarization'}), 500

#MARK: vec dist sem search
@app.route('/semantic_search_distance', methods=['POST'])
def semantic_search_distance():
    logging.debug("Received semantic search request w dist limit.")
    
    search_query = request.form.get('search_query')
    file_types = request.form.getlist('file_type')
    logging.info("info. rcvd semantic search with dist limit. Query: %s, File types: %s", search_query, file_types)
    
    if not search_query or not file_types:
        error_msg = "Missing search query or file types."
        logging.error(error_msg)
        return jsonify({'status': 'error', 'message': error_msg}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Generate embedding for the search query
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.embeddings.create(input=[search_query], model=config.MODEL_EMB)
        search_embedding = response.data[0].embedding
        logging.info("Embedding received")

        # Perform semantic search with vector distance limit
        vec_distance_limit_query = """
            WITH
            page_data AS (
                SELECT 
                    d.id, 
                    d.filename, 
                    d.filepath, 
                    d.filetype, 
                    p.page_number AS paragraph_number,
                    p.all_page_text AS text_content,
                    pi.page_image_base64,
                    p.page_text_embedding AS embedding,
                    CASE WHEN d.filepath LIKE 'docx_from_pdf/%%' THEN TRUE ELSE FALSE END AS converted_from_pdf
                FROM tb_documents d
                JOIN tb_pages p ON d.id = p.document_id
                LEFT JOIN tb_page_images pi ON p.id = pi.page_id
                WHERE p.page_text_embedding IS NOT NULL
                  AND LENGTH(p.all_page_text) >= 36
                  AND (
                        (d.filetype = ANY(%s) AND d.filetype NOT IN ('docx', 'msg')) OR
                        (d.filetype = 'docx' AND 'docx' = ANY(%s) AND d.filepath NOT LIKE 'docx_from_pdf/%%') OR
                        (d.filetype = 'docx' AND 'docx_from_pdf' = ANY(%s) AND d.filepath LIKE 'docx_from_pdf/%%')
                      )
            ),
            paragraph_data AS (
                SELECT 
                    d.id, 
                    d.filename, 
                    d.filepath, 
                    d.filetype, 
                    pp.para_number AS paragraph_number,
                    pp.para_text AS text_content,
                    NULL AS page_image_base64,
                    pp.five_para_text_embedding AS embedding,
                    CASE WHEN d.filepath LIKE 'docx_from_pdf/%%' THEN TRUE ELSE FALSE END AS converted_from_pdf
                FROM tb_documents d
                JOIN tb_page_paragraphs pp ON d.id = pp.document_id
                WHERE pp.five_para_text_embedding IS NOT NULL
                  AND LENGTH(pp.para_text) >= 36
                  AND (
                        (d.filetype = ANY(%s) AND d.filetype NOT IN ('docx', 'msg')) OR
                        (d.filetype = 'docx' AND 'docx' = ANY(%s) AND d.filepath NOT LIKE 'docx_from_pdf/%%') OR
                        (d.filetype = 'docx' AND 'docx_from_pdf' = ANY(%s) AND d.filepath LIKE 'docx_from_pdf/%%')
                      )
            ),
            email_data AS (
                SELECT 
                    d.id, 
                    d.filename, 
                    d.filepath, 
                    d.filetype, 
                    em.message_sequence AS paragraph_number,
                    em.content_text AS text_content,
                    NULL AS page_image_base64,
                    em.content_embedding AS embedding,
                    FALSE AS converted_from_pdf
                FROM tb_documents d
                JOIN tb_email_messages e ON d.id = e.document_id
                JOIN tb_email_individual_messages em ON e.id = em.email_message_id
                WHERE em.content_embedding IS NOT NULL
                  AND LENGTH(em.content_text) >= 36
                  AND d.filetype = 'msg' AND 'msg' = ANY(%s)
            )
            SELECT *
            FROM (
                SELECT *, embedding <-> %s::vector AS distance,
                       ROW_NUMBER() OVER (
                           PARTITION BY text_content
                           ORDER BY embedding <-> %s::vector
                       ) AS rn
                FROM (
                    SELECT * FROM page_data
                    UNION ALL
                    SELECT * FROM paragraph_data
                    UNION ALL
                    SELECT * FROM email_data
                ) all_data
            ) sub
            WHERE rn = 1 AND distance < %s
            ORDER BY distance
            LIMIT %s
        """
        #                            ORDER BY embedding <-> %s::vector 

        # Prepare the parameters for the query
        params = (
            file_types, file_types, file_types,  # For page_data
            file_types, file_types, file_types,  # For paragraph_data
            file_types,                          # For email_data
            search_embedding, search_embedding,  # For embedding distances
            1.02, 128                            # Distance limit and numbers limit
        )
    
        cur.execute(vec_distance_limit_query, params)
        search_results = cur.fetchall()

        # Reduce the size of search results before saving to the session
        session_reduced_search_results = [
            {
                'filename': result.get('filename'),
                'paragraph_number': result.get('paragraph_number'),
                'text_content': result.get('text_content'),
                'converted_from_pdf': result.get('converted_from_pdf')
            } for result in search_results
        ]

        session['last_search_results'] = session_reduced_search_results

        # Create a new variable for HTML rendering with the required fields
        html_reduced_search_results = [
            {
                'filename': result.get('filename'),
                'file_link': url_for('serve_file', filepath=result.get('filepath')),
                'filetype': result.get('filetype'),
                'paragraph_number': result.get('paragraph_number'),
                'text_content': result.get('text_content'),
                'converted_from_pdf': result.get('converted_from_pdf'),
                'distance': result.get('distance')
            } for result in search_results
        ]

        cur.close()
        conn.close()

        if DEBUG:
            # Log the column headers of the search results
            if search_results:
                column_headers = list(search_results[0].keys())
                logging.debug("Column headers in search results: %s", column_headers)

            # Calculate and log the size of raw search results
            raw_size_bytes = sys.getsizeof(search_results)
            raw_size_kb = raw_size_bytes / 1024
            raw_size_mb = raw_size_kb / 1024
            logging.debug("Size of raw search results: %d bytes (%.2f KB, %.2f MB)", raw_size_bytes, raw_size_kb, raw_size_mb)

            # Calculate the total size of raw text content in bytes
            total_text = " ".join(row['text_content'] for row in search_results if 'text_content' in row)
            total_text_size = len(total_text)
            total_tokens = count_tokens_for_text(total_text, model = config.MODEL_COMPLETIONS)

            # Log the size of the raw text content
            logging.debug("raw size of text content returned by SQL query: %d characters", total_text_size)
            logging.debug("raw Tokens returned by vec dist SQL query: %d ", total_tokens)

        if DEBUG:
            # Calculate and log the size of raw search results
            reduced_size_bytes = sys.getsizeof(session_reduced_search_results)
            reduced_size_kb = raw_size_bytes / 1024
            reduced_size_mb = raw_size_kb / 1024
            logging.debug("reduced search results: %d bytes (%.2f KB, %.2f MB)", reduced_size_bytes, reduced_size_kb, reduced_size_mb)

            # Convert the session reduced results to JSON for token calculation
            reduced_text = " ".join(row['text_content'] for row in session_reduced_search_results if 'text_content' in row)
            reduced_text_size = len(reduced_text)
            reduced_tokens = count_tokens_for_text(reduced_text, model=config.MODEL_COMPLETIONS)

            # Log the size of the text content and token count
            logging.debug("session size of text content returned by SQL query: %d characters", reduced_text_size)
            logging.debug("session Tokens returned by vec dist SQL query: %d ", reduced_tokens)
    
            # Log the column headers of the reduced search results
            if html_reduced_search_results:
                html_reduced_column_headers = list(html_reduced_search_results[0].keys())
                logging.debug("Col headers in html reduced search results: %s", html_reduced_column_headers)

        # Render the search results as HTML on the server side with  reduced search results
        if not search_results:
            return jsonify({'status': 'success', 'rendered_html': '<p></p><p>No results found for your search query. Try "Search with Item limit".</p>'})

        rendered_html = render_template('search_results.html', results=html_reduced_search_results)
        return jsonify({'status': 'success', 'rendered_html': rendered_html})
        
    except OpenAIError as oe:
        logging.error("OpenAI API error during semantic search with distance limit: %s", str(oe))
        return jsonify({'status': 'error', 'message': 'OpenAI API error: ' + str(oe)}), 500
    except psycopg2.Error as pe:
        logging.error("Database error during semantic search with distance limit: %s", str(pe))
        return jsonify({'status': 'error', 'message': 'Database error: ' + str(pe)}), 500
    except (ValueError, TypeError, KeyError) as e:
        logging.error("Unexpected error during semantic search with distance limit: %s", str(e))
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500


#MARK: summarize w context
@app.route('/summarize_with_context', methods=['POST'])
def summarize_with_context():
    search_query = request.json['search_query']
    file_types = request.form.getlist('file_type')
    logging.info("info. rcvd semantic search w relevance. Query: %s, File types: %s", search_query, file_types)
    
    flask_session_search_results = session.get('last_search_results', [])

    #convert to json
    initial_results_json = json.dumps(flask_session_search_results)
    initial_tokens = count_tokens_for_text(initial_results_json, config.MODEL_COMPLETIONS)
    logging.debug("initial tokens: %d", initial_tokens)

    # Reduce context to just filename, paragraph_number, and text_content
    # Reduce context to just filename, paragraph_number, and text_content
    reduced_results = [
        {
            'filename': result.get('filename'),
            'paragraph_number': result.get('paragraph_number'),
            'text_content': result.get('text_content')
        } for result in flask_session_search_results
    ]
    
    # Count tokens after reduction (only once)
    reduced_results_json = json.dumps(reduced_results)
    reduced_tokens = count_tokens_for_text(reduced_results_json, config.MODEL_COMPLETIONS)
    logging.debug("Reduced tokens (after reduction): %d", reduced_tokens)

    # Convert the reduced results to JSON string with indentation
    structured_data = json.dumps(reduced_results, indent=2)

    prompt = f"""
        We have this search query: "{search_query}". 
        and the data retrieved by this query via semantic search is this: 
        {structured_data}
        This data comes from the document repository of a commercial construction project based in the UK. 
        We, the organization called Ikon, are managing this project under contract. 
        Our new employees have asked the search query above about the project. 

        Please summarize the data retrieved so it is useful to the new 
        and unexperienced employee who is submitting this query. 
        Arrange the summary in a logical sequence. 
        If the retrieved data contains semantically different subject matters, 
        separate them, returning the identified clusters with their names and .
        Separate semantically different things into separate sections, if needed. 
        
        You MUST be specific in your summary. Vague statements like 
        "Additional references include various clauses related to..." 
        are not permitted, all data must be summarized explicitly. 
        We are involved in mega million dollar commercial projects here, 
        every detail counts and it has implications. 
        It is the main duty of our company to catch every little detail and this 
        is what we are being paid for. 

        You MUST include all the data retrieved by the query in your summary. All rows, all of them. 
        Including items that might seem trivial, like fabrication documents. 
        Provide references at the end of each summary paragraph or section that you send back, 
        with consolidated document name and para numbers. You must include the document name.

        Also provide consolidated references, with document name and para number for all the content 
        used, but only at the end of the summary. 
        You MUST include all the paragraphs in the references. 
        In the references document name must appear only once, with its corresponding paragraphs consolidated.
        """

    # Prepare the messages to be sent to the API
    messages = [
        {"role": "system", "content": "You are a rigorous assistant that summarizes content related to construction projects."},
        {"role": "user", "content": prompt}
    ]

    # Function to count tokens in messages
    def num_tokens_from_messages(messages, model):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logging.warning("Model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 0
        tokens_per_name = 0
        if model.startswith("gpt-3.5-turbo"):
            tokens_per_message = 4  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # If there's a name, the role is omitted
        elif model.startswith("gpt-4"):
            tokens_per_message = 3  # As per OpenAI's token counting rules
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"Token counting not implemented for model {model}.")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Every reply is primed with <|start|>assistant
        return num_tokens

    # Count tokens
    model = config.MODEL_COMPLETIONS
    token_count = num_tokens_from_messages(messages, model)
    logging.info(f"Token count for the request: {token_count}")

    #MARK: send to AI
    try:
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model=config.MODEL_COMPLETIONS,
            messages= messages,
            max_tokens=3500,
            temperature=0.0
        )

        # Extract the summary
        summary = response.choices[0].message.content.strip()

        # Log token usage from the API response
        if DEBUG:
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                logging.debug("API response token usage - Prompt tokens: %d, Completion tokens: %d, Total tokens: %d",
                            prompt_tokens, completion_tokens, total_tokens)

        logging.info("Summary received from API %s", config.MODEL_COMPLETIONS)

        return jsonify({'summary': summary})
    
    except (OpenAIError, ValueError, KeyError) as e:
        logging.error(f"Error in summarization with context: {str(e)}")
        return jsonify({'error': 'An error occurred during summarization with context'}), 500

#MARK: main
if __name__ == '__main__':
    port = int(os.environ.get("HTTP_PLATFORM_PORT", 8000))  # Default to 8000
    serve(app, host='0.0.0.0', port=port)