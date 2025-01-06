from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# Load processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Define model_pipeline
def model_pipeline(text: str, image: Image.Image):
    encoding = processor(image, text, return_tensors="pt")

    # Forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]
