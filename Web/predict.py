import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from datetime import datetime

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_class_labels(num_classes):
    """Retourne les étiquettes en fonction du nombre de classes"""
    if num_classes == 2:
        return ['Normal', 'Pneumonie']
    elif num_classes == 3:
        return ['Normal', 'Pneumonie Virale', 'Pneumonie Bactérienne']
    else:
        return [f"Classe {i}" for i in range(num_classes)]

def build_model_from_state(state_dict):
    """
    Détecte automatiquement l'architecture (MobileNetV2 ou ResNet)
    en fonction des clés du state_dict et la reconstruit.
    """
    if 'classifier.1.weight' in state_dict:
        num_classes = state_dict['classifier.1.weight'].size(0)
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
        return model, num_classes

    if 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].size(0)
        in_features = state_dict['fc.weight'].size(1)

        resnet_variants = [
            models.resnet18,
            models.resnet34,
            models.resnet50,
            models.resnet101,
            models.resnet152
        ]

        for variant in resnet_variants:
            try:
                tmp_model = variant(weights=None)
                if tmp_model.fc.in_features == in_features:
                    tmp_model.fc = torch.nn.Linear(in_features, num_classes)
                    return tmp_model, num_classes
            except Exception:
                continue

    raise ValueError("Architecture non reconnue : les clés du modèle ne correspondent pas.")

def load_model(model_path):
    """Charge dynamiquement un modèle (objet complet, checkpoint ou state_dict brut)"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    try:
        # Support torchxrayvision si nécessaire
        try:
            import torchxrayvision as xrv
            torch.serialization.add_safe_globals({"torchxrayvision.models.DenseNet": xrv.models.DenseNet})
        except ImportError:
            pass  # Ne rien faire si non utilisé

        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        # Cas 1 : modèle complet (pickled object)
        if isinstance(state, torch.nn.Module):
            state.eval()
            if hasattr(state, 'classifier'):
                num_classes = state.classifier[-1].out_features
            elif hasattr(state, 'fc'):
                num_classes = state.fc.out_features
            else:
                num_classes = 2
            return state, get_class_labels(num_classes)

        # Cas 2 : checkpoint contenant un dict avec model_state_dict
        if isinstance(state, dict) and 'model_state_dict' in state:
            model, num_classes = build_model_from_state(state['model_state_dict'])
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            return model, get_class_labels(num_classes)

        # Cas 3 : state_dict pur
        if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
            model, num_classes = build_model_from_state(state)
            model.load_state_dict(state)
            model.eval()
            return model, get_class_labels(num_classes)

        raise ValueError("Format de fichier non reconnu.")

    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

def predict_image(model, image_path, class_labels):
    """Effectue une prédiction sur une image et retourne un dictionnaire étiquettes/scores"""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {class_labels[i]: round(prob.item(), 2) for i, prob in enumerate(probabilities)}

def generate_report(prediction, image_name, save_dir):
    """Génère un rapport .txt avec les résultats de prédiction"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(image_name)[0]
    report_name = f"report_{timestamp}_{base_name}.txt"
    report_path = os.path.join(save_dir, report_name)

    with open(report_path, 'w') as f:
        f.write(f"Analyse de l'image : {image_name}\n")
        f.write("Résultats de la prédiction :\n")
        for label, score in prediction.items():
            f.write(f"- {label} : {score * 100:.2f}%\n")

    return report_path

def explain_prediction(prediction):
    """Renvoie une interprétation textuelle de la prédiction principale"""
    if not prediction:
        return "Aucune explication disponible."

    top_label = max(prediction, key=prediction.get)
    explanations = {
        'Normal': "✅ Aucun signe de pathologie détecté. Vos poumons semblent sains.",
        'Pneumonie': "⚠️ Signes de pneumonie détectés. Consultez un professionnel de santé.",
        'Pneumonie Virale': "⚠️ Signes d'infection virale détectés. Consultez un professionnel de santé.",
        'Pneumonie Bactérienne': "🦠 Signes d'infection bactérienne détectés. Un traitement antibiotique peut être requis."
    }
    return explanations.get(top_label, "Résultat non interprétable.")

