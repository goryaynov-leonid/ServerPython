import torch
import pandas as pd
from torch.utils.data import DataLoader
from http.server import BaseHTTPRequestHandler,HTTPServer
import cgi

import numpy as np
import pymysql.cursors
import facenet_pytorch as fp
from torchvision import transforms, datasets

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        self._set_headers()
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )

        if form.getvalue("type") == "image":
            img = form.getvalue("image")

            out_file = open("img.png", "wb")  # open for [w]riting as [b]inary
            out_file.write(img)
            out_file.close()
            print("image saved")

        #TODO add another request type to create new user

def run(server_class=HTTPServer, handler_class=S, port=9999):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

def getConnectionToDB():
    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='Root1234',
                                 db='pictureusers',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection



def runFacenet():
    mtcnn = fp.MTCNN()
    resnet = fp.InceptionResnetV1(pretrained='casia-webface').eval()

    return mtcnn, resnet


def recognize(userEmbeddings, names, recognitionPath = 'imageToRecognize'):
    # Define a dataset and data loader
    trans = transforms.Compose([
        transforms.Resize(1024)
    ])
    dataset = datasets.ImageFolder(recognitionPath, transform=trans)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    # Perfom MTCNN facial detection
    aligned = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)

    # Calculate image embeddings
    aligned = torch.stack(aligned)
    recognizedEmbedding = resnet(aligned)

    dists = [(e - recognizedEmbedding).norm().item() for e in userEmbeddings]
    df = pd.DataFrame(dists, index=names, columns=['Dist'])

    #returns recognized index
    return df.idxmin()

def getEmbeddings(mtcnn, resnet, imagesPath = 'userImages'):
    # Define a dataset and data loader
    trans = transforms.Compose([
        transforms.Resize(1024)
    ])
    dataset = datasets.ImageFolder(imagesPath, transform=trans)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])

    # Perfom MTCNN facial detection
    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    # Calculate image embeddings
    aligned = torch.stack(aligned)
    embeddings = resnet(aligned)

    return embeddings, names

if __name__ == "__main__":
    mtcnn, resnet = runFacenet()
    embeddings, names = getEmbeddings(mtcnn, resnet)
    print(recognize(embeddings,names)['Dist'])
    #run()