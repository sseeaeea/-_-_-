from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import subprocess
import ollama
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

app = Flask(__name__)

# 모델 로드 및 설정
model = resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/chatgpt", methods=["GET"])
def chatgpt():
    return render_template("chat_gpt.html")

@app.route("/app", methods=["POST"])
def myApp():
    message = request.form.get("message")
    response = ollama.chat(model='llama3:latest', messages=[{'role': 'user', 'content': message}])
    return jsonify({'response': response['message']['content']})

@app.route("/process", methods=["GET"])
def proc():
    try:
        process = subprocess.run(['pip', 'install', 'subprocess'], capture_output=True, text=True)
        output = process.stdout
    except subprocess.CalledProcessError as e:
        output = f"오류"
    return render_template("chat_gpt.html", msg=output)

# 이미지 업로드 및 처리 엔드포인트
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        # 이미지 열기 및 RGB로 변환
        image = Image.open(file).convert('RGB')
        # 이미지 전처리
        image = preprocess(image)

        # 이미지를 배치 차원 (batch dimension) 추가
        image = image.unsqueeze(0)

        # 모델 예측
        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

        # 결과 반환
        return jsonify({'class_index': class_idx})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
