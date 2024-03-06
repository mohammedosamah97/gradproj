
from flask import Flask, redirect, url_for, render_template, request, jsonify

app = Flask(__name__)

# Defining the home page of our site
 # some basic inline html

@app.route("/")
def home():
    return render_template("temp.html")  

ALLOWED_EXTENSIONS = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file found'
    video = request.files['video']
    if video.filename == '':
        return 'No video selected'
    if video and allowed_file(video.filename):
        video.save('static/videos/' + video.filename)

        

        return render_template('preview.html', video_name=video.filename)
    return 'Invalid video file'  
if __name__ == "__main__":
    app.debug = True
    app.run()




