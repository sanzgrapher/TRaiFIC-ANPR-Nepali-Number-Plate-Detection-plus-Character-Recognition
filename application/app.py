import flask
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import logging
import tempfile
import time
import shutil


import config
from model_loader import load_models

from image_processing import process_file

from utils import to_base64


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

try:
    os.makedirs(config.UPLOAD_FOLDER_PATH, exist_ok=True)
    logging.info(f"Upload folder ready: {config.UPLOAD_FOLDER_PATH}")
except OSError as e:
    logging.error(f"Could not create upload folder '{config.UPLOAD_FOLDER_PATH}': {e}", exc_info=True)


logging.info("----- Initializing ANPR Application - Loading Models -----")
try:
    plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = load_models()
    models_loaded = all([plate_detection_model, char_seg_model, char_recog_model])
    if not models_loaded:
        logging.error("One or more models failed to load. Application might not function correctly.")
except Exception as load_err:
     logging.error(f"A critical error occurred during model loading: {load_err}", exc_info=True)
     plate_detection_model, char_seg_model, char_recog_model, device, ocr_font_path = None, None, None, "cpu", None
     models_loaded = False

logging.info("Model Load")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.secret_key = config.FLASK_SECRET_KEY

app.jinja_env.filters['to_base64'] = to_base64
app.jinja_env.globals.update(zip=zip)


@app.route('/', methods=['GET', 'POST'])
def upload_file_route():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)

        if file:
            original_filename = file.filename
            _, file_extension = os.path.splitext(original_filename)
            file_extension = file_extension.lower()

            if file_extension not in config.ALLOWED_EXTENSIONS:
                allowed_str = ", ".join(config.ALLOWED_EXTENSIONS)
                flash(f'Unsupported file type: "{file_extension}". Allowed types: {allowed_str}', 'error')
                logging.warning(f"Upload rejected: Unsupported file type '{file_extension}' from file '{original_filename}'")
                return redirect(request.url)

            temp_path = None
            fd = None
            try:
                fd, temp_path = tempfile.mkstemp(suffix=file_extension, dir=app.config['UPLOAD_FOLDER'], text=False)
                file.save(temp_path)
                logging.info(f"File '{original_filename}' saved temporarily to '{temp_path}'")

                if fd is not None:
                    os.close(fd)
                    fd = None

                if not models_loaded:
                     flash('Models are not loaded correctly. Cannot process file.', 'error')
                     logging.error("Processing aborted: Models not loaded.")
                     return redirect(url_for('upload_file_route'))

                start_process_time = time.time()
                results = process_file(
                    temp_path,
                    plate_detection_model,
                    char_seg_model,
                    char_recog_model,
                    device,
                    ocr_font_path
                 )
                end_process_time = time.time()
                logging.info(f"Processing '{original_filename}' completed in {end_process_time - start_process_time:.3f} seconds. Found {len(results)} plates.")

                return render_template('results.html', results=results, filename=original_filename)

            except Exception as e:
                logging.error(f"Error processing uploaded file '{original_filename}': {e}", exc_info=True)
                flash(f'An error occurred during processing: {str(e)}', 'error')
                return redirect(url_for('upload_file_route'))

            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError as close_err:
                         logging.warning(f"Warning: Could not close temp file descriptor {fd}: {close_err}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logging.info(f"Removed temporary file: {temp_path}")
                    except Exception as rm_err:
                        logging.warning(f"Could not remove temporary file '{temp_path}': {rm_err}")

    return render_template('upload.html')

if __name__ == '__main__':
    logging.info("----- Starting ANPR Flask Application Web Server -----")
    logging.info(f"Flask Secret Key: {'Set' if config.FLASK_SECRET_KEY != 'your_very_secret_key_change_me' else '!!! Using Default !!!'}")
    logging.info(f"Max Upload Size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f} MB")
    logging.info(f"Allowed Extensions: {', '.join(config.ALLOWED_EXTENSIONS)}")
    logging.info(f"Models Loaded: {models_loaded}")
    if not models_loaded:
        logging.warning("Running with one or more models missing!")

    app.run(host='0.0.0.0', port=5001, debug=True)