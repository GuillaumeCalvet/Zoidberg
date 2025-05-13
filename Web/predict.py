import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from datetime import datetime

CLASSES = ['Normal', 'Pneumonie Virale', 'Pneumonie Bactérienne']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(model_path):
    """Charge un modèle MobileNetV2 avec des poids"""
    # Charge uniquement les poids (state_dict)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Reconstruit le modèle avec la même architecture
    model = models.mobilenet_v2(pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {CLASSES[i]: round(prob.item(), 2) for i, prob in enumerate(probabilities)}

def generate_report(prediction, image_name, save_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_name = f"report_{timestamp}_{image_name}.txt"
    report_path = os.path.join(save_dir, report_name)
    with open(report_path, 'w') as f:
        f.write(f"Analyse de l'image : {image_name}\n")
        f.write("Résultats de la prédiction :\n")
        for label, score in prediction.items():
            f.write(f"- {label} : {score * 100:.2f}%\n")
    return report_path
