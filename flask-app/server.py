from flask import Flask, Response, request
from flask import send_from_directory as send
from flask import render_template as render
import sys

app = Flask(__name__, template_folder = './static/templates/')

@app.route('/', methods = ['POST', 'GET'])
def home():
    resp = {'alert' : ''}
    if(request.method == 'POST'):
        ftype = request.files['data-in'].content_type
        if(not(ftype.split('/')[0] == 'image')):
            resp['alert'] = 'Please upload an image to classify'
        else:
            
    return render('index.html', **resp)

    

if __name__ == '__main__':
    app.run()