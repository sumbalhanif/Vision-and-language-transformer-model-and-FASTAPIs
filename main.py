from fastapi import FastAPI, UploadFile # type: ignore
from PIL import Image
import io
from Model_setup import model_pipeline

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/ask")
def ask(text: str, image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))  # Correct way to open uploaded image
    result = model_pipeline(text, image)
    return {"result": result}
