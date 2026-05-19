from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'tiff']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',  methods=['GET', 'POST'])
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
            assert file.filename is not None
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploads', name=filename))
        
        # TODO:
        # run ML analysis of uploaded image
        # deepforest(file) or something like that (?)

        # The idea here is to upload the image and then run the analysis automatically
        # This will more than likely use a fair amount of compute to run, so optimisation will be *key*
        
    return render_template('home.html')

@app.route('/files/<name>')
def serve_file(name):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

@app.route('/uploads/<name>')
def uploads(name):
    print(os.path.join(app.config["UPLOAD_FOLDER"], name))
    return render_template(
        'uploaded.html', 
        image_url=url_for('serve_file', name=name)
        )

@app.route('/base')
def base():
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)