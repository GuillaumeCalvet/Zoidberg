import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from predict import load_model, predict_image, generate_report

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = os.path.expanduser('~/Zoidberg/uploads')
MODEL_FOLDER = os.path.expanduser('~/Zoidberg/models')
REPORT_FOLDER = os.path.expanduser('~/Zoidberg/docs')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    model_choices = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt') or f.endswith('.pth')]
    selected_model = None
    filename = None
    report_path = None

    if request.method == 'POST':
        delete_file = request.form.get('delete')
        if delete_file:
            delete_path = os.path.join(app.config['UPLOAD_FOLDER'], delete_file)
            if os.path.exists(delete_path):
                os.remove(delete_path)
            return redirect(url_for('index'))

        file = request.files.get('file')
        selected_model = request.form.get('model')

        if file and selected_model:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            model_path = os.path.join(MODEL_FOLDER, selected_model)
            model = load_model(model_path)
            prediction = predict_image(model, filepath)
            report_path = generate_report(prediction, filename, REPORT_FOLDER)

    return render_template('index.html',
                           prediction=prediction,
                           model_choices=model_choices,
                           selected_model=selected_model,
                           filename=filename,
                           report_path=report_path)

@app.route('/reports/<path:filename>')
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
