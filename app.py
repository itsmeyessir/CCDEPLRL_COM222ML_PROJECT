import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------- CNN Definition -----------------
import torch.nn as nn


# comment out if not testing
class DeepFakeCNN(nn.Module):
    def __init__(self):
        super(DeepFakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# comment out if not testing
# class DeepFakeCNN(nn.Module):
#     def __init__(self):
#         super(DeepFakeCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, 2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 28 * 28, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 2)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         return self.classifier(x)
    
# ----------------- Setup -----------------
st.title("🎭 DeepFake Detection App")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar dropdown to select a model
st.sidebar.header("🧠 Model Selection")
model_files = sorted([f for f in os.listdir("models") if f.endswith(".pth")])
selected_model_name = st.sidebar.selectbox("Choose a model:", model_files)

@st.cache_resource
def load_model(model_path):
    model = DeepFakeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model_path = os.path.join("models", selected_model_name)
model = load_model(model_path)

mtcnn = MTCNN(keep_all=False, device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # Normalize to [-1, 1]
])

# ----------------- Upload Training Log (Optional) -----------------
st.sidebar.header("📊 Training Evaluation")
log_file = st.sidebar.file_uploader("Upload training log (.csv or .json)", type=["csv", "json"])

if log_file is not None:
    st.subheader("📈 Training Accuracy and Loss")

    if log_file.name.endswith(".csv"):
        df = pd.read_csv(log_file)
        st.line_chart(df[["train_acc", "val_acc"]].rename(columns={"train_acc": "Train Accuracy", "val_acc": "Val Accuracy"}))
        st.line_chart(df[["train_loss", "val_loss"]].rename(columns={"train_loss": "Train Loss", "val_loss": "Val Loss"}))

    elif log_file.name.endswith(".json"):
        import json
        metrics = json.load(log_file)
        st.line_chart(pd.DataFrame({
            "Train Accuracy": metrics["train_accuracies"],
            "Validation Accuracy": metrics["val_accuracies"]
        }))
        st.line_chart(pd.DataFrame({
            "Train Loss": metrics["train_losses"],
            "Validation Loss": metrics["val_losses"]
        }))


# ----------------- Video Upload -----------------
st.header("🎞️ Video Inference")
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

num_frames_to_process = st.slider("📍 Select number of frames to analyze", min_value=1, max_value=30, value=5)
frame_predictions = []

if video_file is not None:

    with NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(video_file.read())
        video_path = tmpfile.name

    st.video(video_file)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num=num_frames_to_process, dtype=int)

    st.write("🔍 Running inference on selected frames...")
    progress_bar = st.progress(0)

    processed = 0
    for i in range(frame_count):
        success, frame = cap.read()
        if not success or i not in frame_indices:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for box in boxes[:1]:
                x1, y1, x2, y2 = map(int, box)
                face = rgb[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_img = Image.fromarray(face)
                face_tensor = transform(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = F.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    label = "REAL" if pred == 0 else "FAKE"
                    confidence = probs[0][pred].item()

                    frame_predictions.append((label, confidence))

                    # Show face with label
                    st.image(face_img, caption=f"{label} ({confidence*100:.2f}%)", width=150)

        processed += 1
        progress_bar.progress(processed / len(frame_indices))

    cap.release()
    
    # ----------------- Inference Summary Visualization -----------------
if frame_predictions:
    st.subheader("📊 Inference Summary")

    labels = [lbl for lbl, _ in frame_predictions]
    label_counts = {"REAL": labels.count("REAL"), "FAKE": labels.count("FAKE")}

    conf_df = pd.DataFrame({
        "Frame": list(range(1, len(frame_predictions)+1)),
        "Confidence": [conf for _, conf in frame_predictions],
        "Prediction": labels
    })

    st.markdown("**Frame-by-frame Confidence:**")
    st.bar_chart(conf_df.set_index("Frame")[["Confidence"]])

    st.markdown("**Label Distribution:**")
    st.write(pd.DataFrame.from_dict(label_counts, orient='index', columns=["Count"]))

    # ✅ Properly build and display the pie chart
    fig, ax = plt.subplots()
    ax.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    progress_bar.empty()

    # Final Verdict
    if frame_predictions:
        labels = [lbl for lbl, _ in frame_predictions]
        majority_label = max(set(labels), key=labels.count)
        st.success(f"🔎 Final Verdict: {majority_label} (via majority vote)")
    else:
        st.error("❌ No face detected in selected frames.")
        
    if majority_label == "REAL":
        st.metric("Final Prediction", "REAL", delta="👍 Trustworthy")
        st.balloons()
    else:
        st.metric("Final Prediction", "FAKE", delta="⚠️ Suspicious")
        st.snow()
