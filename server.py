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

# Unified model — classes: 0=glove, 1=hairnet, 2=no_glove, 3=no_hairnet
model = YOLO('runs/detect/runs/ppe/merged_v4/weights/best.pt')

CLASS_GLOVE = 0
CLASS_HAIRNET = 1


@app.get("/")
def health():
    return {"status": "AI server running"}


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = model(temp_path, conf=0.5)

        glove_count = 0
        wearing_hairnet = False

        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                if cls == CLASS_GLOVE:
                    glove_count += 1
                elif cls == CLASS_HAIRNET:
                    wearing_hairnet = True

        wearing_gloves = glove_count >= 2
        approved = wearing_hairnet and wearing_gloves

        if approved:
            message = "Approved! You are wearing both hairnet and gloves."
        elif wearing_hairnet and not wearing_gloves:
            message = "You are wearing a hairnet but NO gloves (or only one!)."
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)