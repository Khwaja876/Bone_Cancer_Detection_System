import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Ensure notebooks directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from notebooks.data_preprocessing import get_dataloaders, CLAHETransform

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'bone_cancer_model.pth')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

class BoneCancerClassifier(nn.Module):
    """
    ResNet18 / Transfer Learning CNN for Bone Cancer Classification.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(BoneCancerClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class GradCAMEngine:
    """
    Gradient-Weighted Class Activation Mapping (Grad-CAM)
    for visual explainability in bone tumor localization.
    """
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        target_layer = dict(self.model.backbone.named_children())[target_layer_name]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
            
        cam = cv2.resize(cam, (224, 224))
        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        return cam, target_class, probs

def train_model(epochs=5, lr=1e-3, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    train_loader, val_loader, class_names = get_dataloaders(batch_size=batch_size)
    num_classes = len(class_names)
    
    model = BoneCancerClassifier(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n=== Starting Model Training Phase ===")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'val_acc': val_acc
            }, MODEL_SAVE_PATH)

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved model to: {MODEL_SAVE_PATH}")
    return model, class_names

if __name__ == '__main__':
    train_model(epochs=5)
