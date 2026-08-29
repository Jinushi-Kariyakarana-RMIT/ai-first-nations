from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
import os
import secrets
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from ml_model import load_or_train_model, predict_mangrove, predict_combined
from binary_detector import load_binary_model

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Secret key: read from environment in production; fall back to a random
# per-process key locally so nothing hardcoded ever ships in source control.
# Note: a random fallback means sessions/flashes won't survive a restart -
# set FLASK_SECRET_KEY explicitly for any real deployment.
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)
if not os.environ.get('FLASK_SECRET_KEY'):
    print("Warning: FLASK_SECRET_KEY not set - using a random key for this process only.")

DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = None
mangrove_type = None
gpu = None
binary_model = None


def initialize_models():
    global model, mangrove_type, gpu, binary_model
    print("Initializing ML models...")

    # Load multi-class model
    try:
        model, mangrove_type, gpu = load_or_train_model()
        print("Multi-class ML model ready.")
    except Exception as e:
        print(f"Warning: Error preparing multi-class ML model: {e}")
        model = None
        mangrove_type = None
        gpu = None

    # Load binary mangrove detector
    try:
        binary_model = load_binary_model()
        print("Binary mangrove detector ready.")
    except Exception as e:
        print(f"Warning: Error loading binary mangrove detector: {e}")
        binary_model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_image(filepath):
    """Verify the uploaded file is actually a readable image, not just a
    file with an image-like extension. Catches renamed/corrupt uploads
    before they reach the ML pipeline."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


# Prevent double initialization in Flask reloader
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not DEBUG_MODE:
    initialize_models()


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'orthomosaic' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['orthomosaic']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if not is_valid_image(filepath):
                os.remove(filepath)
                flash('That file is not a valid image (extension did not match content).')
                return redirect(request.url)

            flash(f'File uploaded successfully: {filename}')
            return redirect(url_for('analyze', name=filename))
        else:
            flash('Invalid file type. Allowed: PNG, JPG, JPEG')
            return redirect(request.url)

    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/files/<name>')
def serve_file(name):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)


@app.route('/analyze/<name>')
def analyze(name):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], name)

    if not os.path.exists(filepath):
        flash(f'File not found: {name}')
        return redirect(url_for('upload'))

    analysis_result = None
    error_message = None

    if model is not None and binary_model is not None:
        try:
            analysis_result = predict_combined(filepath, binary_model, model, mangrove_type, gpu)
        except Exception as e:
            error_message = f"Error during analysis: {str(e)}"
            print(f"Exception during analysis: {e}")
    elif model is not None:
        try:
            analysis_result, _, error = predict_mangrove(filepath, model, mangrove_type, gpu)
            if error:
                error_message = error
                print(f"Analysis error: {error}")
        except Exception as e:
            error_message = f"Error during analysis: {str(e)}"
            print(f"Exception during analysis: {e}")
    else:
        error_message = "ML models not available. Analysis cannot be performed."

    image_url = url_for('serve_file', name=name)

    return render_template(
        'analysis_results.html',
        image_url=image_url,
        filename=name,
        result=analysis_result,
        error=error_message
    )


@app.route('/uploads/<name>')
def uploads(name):
    return render_template(
        'uploaded.html',
        image_url=url_for('serve_file', name=name)
    )


@app.route('/base')
def base():
    return render_template('base.html')


if __name__ == '__main__':
    # Debug mode (and its interactive debugger) must stay off outside local
    # development - set FLASK_DEBUG=true in your local .env to enable it.
    app.run(debug=DEBUG_MODE)