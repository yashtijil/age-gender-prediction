# Age and Gender Prediction

## Project Overview

This project utilizes deep learning pre-trained models and transformers for age and gender prediction. By uploading an image, the application will provide predictions for both age and gender based on the content of the image.


## FaceXFormer Model 

The FaceXFormer model can be downloaded either manually from [Hugging Face](https://huggingface.co/kartiknarayan/facexformer) or programmatically using the following Python code:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
```
### Directory Structure
After downloading, ensure that the directory structure is as follows:
```plaintext

facexformer/
├── ckpts/
│   └── model.pt
├── network/
└── inference.py
```

## MiVOLO Model
To set up the MiVOLO model, follow these steps:

1.	[Download](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) body + face detector model to mivolo/models/yolov8x_person_face.pt
2.	[Download](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) mivolo checkpoint to mivolo/models/mivolo_imbd.pth.tar

## Main File(agp.py)
To run the application, execute the following command:
```python
uvicorn agm:app --reload
```
Once the server is running, navigate to /docs in your browser to access Swagger UI. From there, the model can be tested.


