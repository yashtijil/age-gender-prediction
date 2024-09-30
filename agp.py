import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiVOLO'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'facexformer'))

from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from facenet_pytorch import MTCNN
from MiVOLO.mivolo.predictor import Predictor
import cv2
import argparse
import numpy as np
from facexformer.network import FaceXFormer

app = FastAPI()

facexformer_weights_path = "facexformer/ckpts/model.pt"
yolov8_weights_path = "MiVOLO/models/yolov8x_person_face.pt"
mivolo_weights_path = "MiVOLO/models/mivolo_imbd.pth.tar"

# MTCNN
mtcnn = MTCNN(keep_all=True, device="cpu")

# FaceXFormer
face_xformer_model = FaceXFormer().to("cpu")
checkpoint = torch.load(facexformer_weights_path, map_location="cpu")
face_xformer_model.load_state_dict(checkpoint['state_dict_backbone'])
face_xformer_model.eval()

# MiVOLO
def setup_mivolo_predictor():
    args_dict = {
        'detector_weights': yolov8_weights_path,
        'checkpoint': mivolo_weights_path,
        'device': "cpu",
        'with_persons': True,
        'disable_faces': False,
        'draw': False
    }
    return Predictor(argparse.Namespace(**args_dict))

mivolo_predictor = setup_mivolo_predictor()

# Utility function for FaceXFormer inference
def process_image_with_face_xformer(image, model):
    transforms_image = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transforms_image(image)
    task = torch.tensor([4])
    images = image.unsqueeze(0).to("cpu")

    with torch.no_grad():
        _, _, _, _, age_output, gender_output, _, _ = model(images, {"a_g_e": torch.zeros([3]).unsqueeze(0)}, task.to("cpu"))
    age_preds = torch.argmax(age_output, dim=1)[0]
    gender_preds = torch.argmax(gender_output, dim=1)[0]
    return age_preds, gender_preds

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AGM API"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    
    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return "NOT DETECTED"

    # Crop the first detected face and run it through FaceXFormer
    x_min, y_min, x_max, y_max = boxes[0]
    img_cropped = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    age_preds_face_xformer, gender_preds_face_xformer = process_image_with_face_xformer(img_cropped, face_xformer_model)

    # Convert the original image to OpenCV format for MiVOLO
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    detected_objects, _ = mivolo_predictor.recognize(img_cv)
    ages = detected_objects.ages
    genders = detected_objects.genders
    
    # Age, gender prediction using MiVOLO
    age_preds_mivolo = next((age for age in ages if age is not None), None)
    gender_preds_mivolo = next((gender for gender in genders if gender is not None), None)
    
    if age_preds_mivolo is None:
        return "NOT DETECTED"

    age_category = 'CHILD' if age_preds_mivolo <= 15 else 'ADULT'

    # labels for FaceXFormer
    age_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    gender_labels = ["male", "female"]

    face_xformer_age_label = age_labels[age_preds_face_xformer.item()]
    face_xformer_gender_label = gender_labels[gender_preds_face_xformer.item()]
    
    return {
        "Age_Category": age_category,
        "FaceXFormer_Age": face_xformer_age_label,
        "MiVOLO_Age": age_preds_mivolo,
        "FaceXFormer_Gender": face_xformer_gender_label,
        "MiVOLO_Gender": gender_preds_mivolo
    }
    # return age_category
    