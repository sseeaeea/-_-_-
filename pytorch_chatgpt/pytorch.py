import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

# 모델 로드 (사전 학습된 ResNet18 모델)
model = resnet18(pretrained=True)
model.eval()  # 추론 모드로 전환

# 이미지 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
