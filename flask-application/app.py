from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
import os
import secrets
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from ml_model import load_or_train_model, predict_mangrove, predict_combined
from binary_detector import load_binary_model, predict_binary
from tree_counter_service import load_tree_counter_model, predict_tree_count

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
MAX_RECENT_UPLOADS = 6

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
tree_counter_model = None

# In-memory history of recent uploads for display on the upload page.
# Not persisted across restarts - fine for local/demo use, but this would
# need a real store (DB, or even a JSON file) to survive a server restart
# or run across multiple worker processes in production.
recent_uploads = []


def record_upload(filename, result, error):
    """Add an entry to the recent-uploads list (most recent first)."""
    if error:
        summary = 'Analysis error'
        status = 'error'
    elif not result:
        summary = 'No result'
        status = 'error'
    elif result.get('multi_class'):
        mc = result['multi_class']
        summary = f"{mc['predicted_class']} ({mc['confidence'] * 100:.0f}%)"
        status = 'ok'
    elif result.get('tree_count', {}).get('status') == 'available':
        tree_count = result['tree_count']['estimated_trees']
        summary = f"Estimated {tree_count} trees"
        status = 'ok'
    elif 'predicted_class' in result:
        # Fallback single-model path (predict_mangrove) returns the
        # classification dict directly rather than nested under multi_class.
        summary = f"{result['predicted_class']} ({result['confidence'] * 100:.0f}%)"
        status = 'ok'
    else:
        binary = result.get('binary') if result else None
        summary = binary['prediction'] if binary else 'No result'
        status = 'none'

    recent_uploads.insert(0, {
        'filename': filename,
        'summary': summary,
        'status': status,
        'timestamp': datetime.now().strftime('%d %b, %H:%M'),
    })
    del recent_uploads[MAX_RECENT_UPLOADS:]


def initialize_models():
    global model, mangrove_type, gpu, binary_model, tree_counter_model
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

    try:
        tree_counter_model = load_tree_counter_model()
        print("Experimental tree counter ready.")
    except Exception as e:
        print(f"Warning: Error loading tree counter: {e}")
        tree_counter_model = None


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
@app.route('/analyse', methods=['GET', 'POST'])
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

            return redirect(url_for('analyze', name=filename))
        else:
            flash('Invalid file type. Allowed: PNG, JPG, JPEG')
            return redirect(request.url)

    return render_template('home.html', recent_uploads=recent_uploads)


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

    analysis_result = {
        'binary': None,
        'multi_class': None,
        'tree_count': {
            'status': 'unavailable',
            'message': 'Tree counting is not available in this environment.',
        },
    }
    error_message = None

    if model is not None and binary_model is not None:
        try:
            mangrove_result = predict_combined(filepath, binary_model, model, mangrove_type, gpu)
            analysis_result.update(mangrove_result)
        except Exception as e:
            analysis_result['mangrove_error'] = "Mangrove analysis could not be completed."
            print(f"Exception during analysis: {e}")
    elif binary_model is not None:
        try:
            analysis_result.update({
                'binary': predict_binary(filepath, binary_model),
                'multi_class': None,
            })
        except Exception as e:
            analysis_result['mangrove_error'] = "Mangrove detection could not be completed."
            print(f"Binary analysis exception: {e}")
    elif model is not None:
        try:
            classification, _, error = predict_mangrove(filepath, model, mangrove_type, gpu)
            if error:
                analysis_result['mangrove_error'] = "Mangrove classification could not be completed."
                print(f"Analysis error: {error}")
            else:
                analysis_result['multi_class'] = classification
        except Exception as e:
            analysis_result['mangrove_error'] = "Mangrove classification could not be completed."
            print(f"Exception during analysis: {e}")
    else:
        print("Warning: Mangrove analysis models are unavailable.")

    overlay_name = f"{os.path.splitext(name)[0]}-tree-overlay.jpg"
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_name)
    try:
        tree_result = predict_tree_count(filepath, tree_counter_model, overlay_path)
        if tree_result.get('overlay_path'):
            tree_result['overlay_url'] = url_for('serve_file', name=overlay_name)
        analysis_result['tree_count'] = tree_result
    except Exception as e:
        analysis_result['tree_count'] = {
            'status': 'error',
            'message': 'Tree counting could not be completed for this image.',
        }
        print(f"Tree-counting exception: {e}")

    modules_available = any([
        analysis_result.get('binary'),
        analysis_result.get('multi_class'),
        analysis_result.get('tree_count', {}).get('status') == 'available',
    ])
    if not modules_available and error_message is None:
        error_message = "Analysis models are currently unavailable. Please contact the project team."

    image_url = url_for('serve_file', name=name)

    record_upload(name, analysis_result, error_message)

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


@app.errorhandler(413)
def file_too_large(_error):
    flash('Image is too large. Please upload an image smaller than 50 MB.')
    return redirect(url_for('upload'))


if __name__ == '__main__':
    # Debug mode (and its interactive debugger) must stay off outside local
    # development - set FLASK_DEBUG=true in your local .env to enable it.
    app.run(debug=DEBUG_MODE)
