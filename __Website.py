from flask import Flask, redirect, url_for, render_template, request, jsonify, send_file
from PIL import Image
import numpy as np
import cv2  
import glob
import pickle
import pandas as pd
from tensorflow import keras
from tensorflow import image
from PIL import Image
from tensorflow.keras.preprocessing import image
#from keras.applications import ImageDataGenerator
import tensorflow 
import time 
import os 
import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm
import logging 

##GENERATION PART (feature extraction)
def get_id_emb(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

#function that faceswaps 
import traceback

def video_test(source_img_path, target_video_path, output_path, image_size, merge_result, use_gpu):
    print("Inside video_test function")

    try:
        paddle.set_device("gpu" if use_gpu else 'cpu')
        print("Device set to:", "gpu" if use_gpu else 'cpu')

        faceswap_model = FaceSwap(use_gpu)
        print("FaceSwap model initialized")

        print("Initializing ResNet")
        id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        print("ResNet initialized")
        
        print("Loading ResNet parameters")
        id_net.set_dict(paddle.load('C:/Users/moham/OneDrive/Desktop/GraduationProject-main/checkpoint/._arcface.pdparams'))
        print("ResNet parameters loaded")

        id_net.eval()

        weight = paddle.load('C:/Users/moham/OneDrive/Desktop/GraduationProject-main/checkpoint/._MobileFaceSwap_224.pdparams')
        print("Model weight loaded")

        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        print("Landmark model prepared")

        id_img = cv2.imread(source_img_path)
        landmark = landmarkModel.get(id_img)
        print("Landmarks detected")

        if landmark is None:
            print('**** No Face Detect Error ****')
            return

        aligned_id_img, _ = align_img(id_img, landmark)

        id_emb, id_feature = get_id_emb(id_net, aligned_id_img)
        print("Identity embedding generated")

        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()
        print("FaceSwap model set up")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture()
        cap.open(target_video_path)
        print("Target video opened")

        videoWriter = cv2.VideoWriter(os.path.join(output_path, os.path.basename(target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Video writer initialized")

        for i in tqdm(range(int(all_f))):
            ret, frame = cap.read()
            print(f"Processing frame {i+1}/{int(all_f)}")

            landmark = landmarkModel.get(frame)
            if landmark is not None:
                att_img, back_matrix = align_img(frame, landmark)
                att_img = cv2paddle(att_img)
                res, mask = faceswap_model(att_img)
                res = paddle2cv(res)
                mask = np.transpose(mask[0].numpy(), (1, 2, 0))
                res = dealign(res, frame, back_matrix, mask)
                frame = res
            else:
                print('**** No Face Detect Error ****')

            videoWriter.write(frame)

        cap.release()
        videoWriter.release()

    except Exception as e:
        print("Error in video_test:", e)
        traceback.print_exc()
        # Handle the exception gracefully or re-raise it as needed


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("temp.html") 
# Defining the home page of our site
 # some basic inline html
@app.route('/generate.html')
def generate():
    return render_template("generate.html") 
@app.route('/Generator1', methods=['POST'])
def Generator1(): 
    global video1 
    video1 = request.files['video1']
    if video1.filename[-3:] in ["mp4","jpg","png"]:
        video1.save('C:/Users/moham/OneDrive/Desktop/GraduationProject-main/generate/inpute/' + video1.filename)
    return render_template("generate.html")
@app.route('/Generator2', methods=['POST'])
def Generator2():  
    video2 = request.files['video2']
    if video2.filename[-3:] in ["mp4","jpg","png"]:
        video2.save('C:/Users/moham/OneDrive/Desktop/GraduationProject-main/generate/inpute/' + video2.filename)    
    source_img_path = 'C:/Users/moham/OneDrive/Desktop/GraduationProject-main/generate/inpute/' + video2.filename
    target_video_path = 'C:/Users/moham/OneDrive/Desktop/GraduationProject-main/generate/inpute/' + video1.filename

    output_path = "C:/Users/moham/OneDrive/Desktop/GraduationProject-main/result/"
    image_size = 256
    merge_result = True
    use_gpu = False
    print(video1.filename)
    print(video2.filename)

    video_test(source_img_path ,target_video_path ,output_path ,image_size ,merge_result ,use_gpu)  
      
    return render_template("generate.html")
from flask import send_file

from flask import send_file
import os

@app.route('/download-video')
def download_video():
    # Replace 'video_path' with the actual path to your video file
    video_filename = os.listdir("C:/Users/moham/OneDrive/Desktop/GraduationProject-main/result/")[0]
    video_path = os.path.join("C:/Users/moham/OneDrive/Desktop/GraduationProject-main/result/", video_filename)
    
    # Set the appropriate headers for the response
    headers = {
        'Content-Disposition': f'attachment; filename={video_filename}',
        'Content-Type': 'video/mp4',
    }
    
    # Send the file as a response with the specified headers
    return send_file(video_path, as_attachment=True, mimetype='video/mp4')





#DETECTION PART
ALLOWED_EXTENSIONS = ['mp4']
#ALLOWED_EXTENSIONS = ['jpg']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    
    print("Inside upload function")  # Add this line
    video = request.files['video']
    print("Received video file")  # Add this line


    for i in os.listdir(f"C:/Users/moham/OneDrive/Desktop/GraduationProject-main/messi/"):
        os.remove(f"C:/Users/moham/OneDrive/Desktop/GraduationProject-main/messi/{i}")

    if video.filename[-3:] == "mp4":
        video.save('C:/Users/moham/OneDrive/Desktop/GraduationProject-main/static/videos/' + video.filename)
        img_number = 0
        # Load the cascade  
        #preprocessing to extract facial features
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        #file path
        path = 'C:/Users/moham/OneDrive/Desktop/GraduationProject-main/static/videos/' + video.filename  
        all_videos = glob.glob(path)
        li = []
        counter = 0
        for video_path in all_videos:
            # To capture video from existing video.  
            cap = cv2.VideoCapture(video_path)  
            counter += 1
            while True:  
            # Read the frame 
               ret, img = cap.read()     
               if not ret:
                   break
            # Detect the faces  
               faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)
               print(f"Number of faces detected in frame: {len(faces)}")      
            # Draw the rectangle around each face  
               for (x, y, w, h) in faces: 
                   cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)  
                   crop_img = img[y:y+h, x:x+w]
                   img_number += 1
                   filename = f"face_{counter}_{img_number}.jpg"
                   filepath = os.path.join("C:/Users/moham/OneDrive/Desktop/GraduationProject-main/faceimg/", filename)
                   cv2.imwrite(filepath, crop_img)
                
            # Release the VideoCapture object  
            cap.release()
            print("preload")

    else:
        video.save(f"C:/Users/moham/OneDrive/Desktop/GraduationProject-main/messi/{video.filename}")    
    
    #load the pretrained model
    pickled_model = tensorflow.keras.models.load_model("C:/Users/moham/OneDrive/Desktop/GraduationProject-main/cnn2.hdf5", compile=False)
 
    print("precompile")

    pickled_model.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])
    print("postcompile")

    #CLASSIFICATION 
    img = image.load_img(f'C:/Users/moham/OneDrive/Desktop/GraduationProject-main/faceimg/{os.listdir("C:/Users/moham/OneDrive/Desktop/GraduationProject-main/faceimg/")[0]}', target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (96, 96, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 96, 96, 3) and return 4D tensor
    x = np.expand_dims(x, axis=0)
    x = x/255
    #classes = pickled_model.predict_classes(x)
    predict_x = pickled_model.predict(x) 
    if predict_x[0][0] < 0.5:
        print (predict_x)
        print(predict_x[0])
        prediction = "FAKE!"
    else:
        print(predict_x[0])
        prediction = "Real!"    

    #time.sleep(10) 
    return render_template('temp.html', prediction=prediction)   
if __name__ == "__main__":
    app.debug = True
    app.run()




