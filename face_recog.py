import uvicorn
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import numpy as np 
import cv2
import io
import base64
from cv_model import op_model
import time
import os
import shutil
# 2. Create the app object
app = FastAPI()
db = []
origins = [
    "*"
]

class DataSchema(BaseModel):
    image:str

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/send")
async def send(data: DataSchema):
    #print(data.image)
    #i = 0
    shutil.rmtree("./Frames/")
    shutil.rmtree("./Model_gen")
    os.mkdir("./Frames")
    os.mkdir("./Model_gen")
    decodeit = open("./Frames/"+"Frame_"+str(time.time())+'.jpeg', 'wb')
    decodeit.write(base64.b64decode((data.image)))
    decodeit.close()
    #i = i +1
    image = base64.b64decode(data.image)
    for i in os.listdir("./Frames/"):
        img_op = op_model("C:/Users/hamma/OneDrive/Documents/face_recog/Frames/"+i)
    
    cv2.imwrite("./Model_gen/Frame"+str(time.time())+".jpg",img_op)
    for j in os.listdir("./Model_gen/"):
        with open("./Model_gen/"+j, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
    
    return b64_string
if __name__ == '__main__':
    uvicorn.run(app, host='192.168.18.189', port=8000, debug=True)