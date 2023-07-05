# import requirements needed
from flask import Flask, render_template, request
import flask
from utils import get_base_url
import os
from qa import question_answer, summerization
# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/content/')
def content():
    return render_template('content.html')

@app.route('/qa/')
def qa():

    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    context = {"files": uploaded_files}
    return render_template('qa.html', **context)

@app.route('/upload/', methods=['POST'])
def upload():
    if not request.files['file']:
        flask.abort(403)
    file = request.files['file']
    file_name = file.filename
    temp_path = 'static/upload/' + file_name
    file.save(temp_path)

    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    context = {"files": uploaded_files}
    return render_template('qa.html', **context)

@app.route('/delete/', methods=['POST'])
def delete():

    if 'filename' not in request.form:
        flask.abort(403)
    filename = request.form['filename']
    if filename not in os.listdir('static/upload'):
        flask.abort(403)
    path = 'static/upload/' + filename
    
    os.remove(path)

    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    context = {"files": uploaded_files}
    return render_template('qa.html', **context)

@app.route('/submit/', methods=['POST'])
def submit():

    if 'key' not in request.form or 'question' not in request.form:
        flask.abort(403)
    

    key = request.form['key']
    question = request.form['question']

    res_text = question_answer(key, question)

    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    context = {"files": uploaded_files, 'response': res_text}
    return render_template('qa.html', **context)

@app.route('/summerize/', methods=['POST'])
def summerize():

    if 'key' not in request.form:
        flask.abort(403)
    

    key = request.form['key']

    res_text = summerization(key)

    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    context = {"files": uploaded_files, 'response': res_text}
    return render_template('qa.html', **context)



if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'url'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)