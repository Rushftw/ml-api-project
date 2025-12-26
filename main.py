from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms, models
import torch
from PIL import Image
import io

app = FastAPI()

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet labels (for classification)
import requests
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

@app.get("/")
def read_root():
    return {"message": "Image Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess and predict
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(input_batch)
    
    # Get top prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_idx = torch.max(probabilities, 0)
    
    predicted_label = labels[top_idx.item()]
    confidence = top_prob.item()
    
    return JSONResponse(content={
        "predicted_class": predicted_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)