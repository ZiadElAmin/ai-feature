from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


hairnet_model = YOLO('runs/detect/hairnet_model/weights/best.pt')
gloves_model = YOLO('runs/detect/gloves_model/weights/best.pt')


@app.get("/")
def health():
    return {"status": "AI server running"}


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    #
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        hairnet_results = hairnet_model(temp_path, conf=0.5)
        gloves_results = gloves_model(temp_path, conf=0.5)


        wearing_hairnet = False
        for result in hairnet_results:
            for box in result.boxes:
                if int(box.cls) == 1:
                    wearing_hairnet = True


        glove_count = 0
        for result in gloves_results:
            for box in result.boxes:
                if int(box.cls) == 0:
                    glove_count += 1

        wearing_gloves = glove_count >= 2

        approved = wearing_hairnet and wearing_gloves

        if approved:
            message = "Approved! You are wearing both hairnet and gloves."
        elif wearing_hairnet and not wearing_gloves:
            message = " You are wearing a hairnet but NO gloves (or only one!)."
        elif not wearing_hairnet and wearing_gloves:
            message = "You are wearing gloves but NO hairnet."
        else:
            message = "You are wearing NEITHER hairnet nor gloves."

        return {
            "hairnet": wearing_hairnet,
            "gloves": wearing_gloves,
            "approved": approved,
            "message": message
        }
    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)