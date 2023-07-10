# import requirements needed
from flask import Flask, render_template, request, session, redirect
import flask
from utils import get_base_url
import os
from qa import question_answer, summarization, long_question_answer, translation
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
    if 'api_key' not in session:
        return redirect('/login')
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
    if 'api_key' not in session:
        return redirect('/login')
    if 'question' not in request.form or 'option' not in request.form or 'lang' not in request.form:
        flask.abort(403)
    

    key = session['api_key']
    question = request.form['question']
    choice = request.form['option']
    lang = request.form['lang']
    if choice == 'short':
        res_text = question_answer(key, question)
        if lang != 'English':
            res_text = translation(key, request.form['lang'], res_text)
    else:
        res_text = long_question_answer(key, question)
    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    if len(uploaded_files) == 0:
        flask.abort(400)
    context = {"files": uploaded_files, 'response': res_text}
    return render_template('qa.html', **context)

@app.route('/summarize/', methods=['POST'])
def summerize():
    if 'api_key' not in session:
        return redirect('/login')
    if 'file' not in request.form or 'lang' not in request.form:
        flask.abort(403)
    key = session['api_key']

    filename = request.form['file']
    lang = request.form['lang']
    res_text = summarization(key, filename)
    if lang != 'English':
        res_text = translation(key, request.form['lang'], res_text)
    

    uploaded_files = []
    for filename in os.listdir('static/upload'):
        uploaded_files.append(filename)
    context = {"files": uploaded_files, 'response': res_text}
    return render_template('qa.html', **context)

@app.route('/login')
def login():
    if 'api_key' in session:
        return redirect('qa')
    else:
        return render_template('login.html')

@app.route('/signin', methods=['POST'])
def signin():
    if 'api_key' not in request.form:
        flask.abort(403)

    session['api_key'] = request.form['api_key']
    return redirect('/qa')
@app.route('/logout', methods=['POST'])
def logout():
    if 'api_key' not in session:
        flask.abort(403)

    session.clear()
    return redirect('/')

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'url'
    # Secret key for encrypting cookies
    app.secret_key = 'super secret key'
    # File Upload to var/uploads/
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)