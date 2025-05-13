import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

model_name = "google/mobilenet_v2_1.0_224"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]
    return label

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path_to_image")
    else:
        image_path = sys.argv[1]
        label = predict(image_path)
        print(f"Predicted label: {label}")
