from flask import Flask, request
import labelMap_pb2 
from os.path import join
import numpy as np
import pickle
import PIL.Image as Image
import io
import torch
import cv2

import hashlib
import time

from inference import Muse

inc=1
app = Flask(__name__)
model = Muse()


def gen_uuid(l: int = 10) -> str:
    hash = hashlib.sha1()
    hash.update(str(time.time()))
    print(hash.hexdigest())
    print(hash.hexdigest()[:10])
    return hash.hexdigest()[:l]


def save_image(img: np.ndarray):
     fname = join('./infer_images', f'{str(time.time())}.png')
     print('saving image: ', fname)
     im = Image.fromarray(img)
     im.save(fname)

def process_pb(pb: labelMap_pb2.Labelmapbuffer):
    """ Convert Protocal buffer png
    """
    h = pb.height
    w = pb.width

    labelMap = np.array(pb.pixel).reshape(h, w)
    return labelMap

def handler(pb: labelMap_pb2.Labelmapbuffer):
    """ Convert Protocal buffer png
    """
    print('received image')
    label = process_pb(pb)
    print('got label')
    print('inference')
    pred = model.infer(label)    

    save_image(pred)
    return pred
    # return arr, im

def img_to_pb(yhat, h, w):
    yhat = cv2.resize(yhat, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    yhat = yhat[...,[2,1,0]] 
    return cv2.imencode('.png', yhat)[1]

def save_pb_pickle(pb):
    with open('./example_pickle_v2.pkl', 'wb') as f:
        pickle.dump(pb, f)
        print('saved pb pickle')

@app.route("/image", methods=["GET","POST"])
def receiveImage():
    data = request.get_data()
    print('message received')
    labelMap = labelMap_pb2.Labelmapbuffer()
    labelMap.ParseFromString(data) 
    print('parsed protobuffer', labelMap.height, labelMap.width)
    save_pb_pickle(labelMap)
    pred = handler(labelMap)
    pngData = img_to_pb(pred, labelMap.height, labelMap.width)
    #print(pngData)
    gen_img_response = labelMap_pb2.GeneratedImageResponse()
    gen_img_response.data.extend(pngData)
    
    #with open('./generated_pb.pkl', 'wb') as f:
    #    pickle.dump(gen_img_response, f)

    #print('genbuffer: ',gen_img_response)

    return gen_img_response.SerializeToString()

@app.after_request
def add_header(response):
    response.headers['Content-Type'] = 'application/protobuf'
    return response



# Get Speakers speaking at the conference. Loop through the predefinied array of contact dictionary, and create a Contact object, adding it to an array and serializing our Speakers object.
# @app.route("/speakers")
# def getSpeakers():
#     contacts = []
#     return speakers.SerializeToString()

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
