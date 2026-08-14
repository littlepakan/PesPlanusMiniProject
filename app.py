import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
import os
import glob

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ==========================================
st.set_page_config(
    page_title="Pes Planus Diagnosis AI",
    page_icon="🦶",
    layout="centered"
)

st.title("🦶 ระบบปัญญาประดิษฐ์คัดกรองภาวะเท้าแบน")
st.markdown("อัปโหลดภาพถ่ายรังสีเอกซ์ (X-ray) บริเวณเท้าเพื่อวิเคราะห์และจำแนกภาวะ Pes Planus")

# ==========================================
# 2. ฟังก์ชันโหลดโมเดล CNN & Classifier ทั้งหมด
# ==========================================
@st.cache_resource
def load_ai_models():
    # โหลด SqueezeNet (Feature Extractor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # ค้นหาไฟล์ .pkl ทั้งหมดในโฟลเดอร์ปัจจุบัน
    pkl_files = glob.glob("*.pkl")
    models_dict = {}
    
    for file_path in pkl_files:
        try:
            with open(file_path, 'rb') as f:
                model_name = file_path.replace('.pkl', '') # ลบนามสกุลไฟล์ออกเพื่อใช้เป็นชื่อแสดงผล
                models_dict[model_name] = pickle.load(f)
        except Exception as e:
            print(f"ไม่สามารถโหลดไฟล์ {file_path} ได้: {e}")
            
    return backbone, device, models_dict

# โหลดโมเดลทั้งหมดเก็บไว้ในหน่วยความจำ
feature_extractor, device, available_models = load_ai_models()

# ==========================================
# 3. กำหนดตัวประมวลผลภาพ (Image Transform)
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABEL_DECODER = {
    0: "Normal (ปกติ)",
    1: "Pes Planus (ภาวะเท้าแบน)"
}

# ==========================================
# 4. ส่วนของ Sidebar แสดงสถานะและเลือกโมเดล
# ==========================================
st.sidebar.header("⚙️ การตั้งค่า AI")

# สร้าง Dropdown ให้เลือกโมเดลถ้ามีไฟล์ .pkl อยู่ในโฟลเดอร์
clf = None
if available_models:
    st.sidebar.success(f"✅ พบโมเดลพร้อมใช้งาน {len(available_models)} ตัว")
    selected_model_name = st.sidebar.selectbox(
        "เลือกระบบจำแนก (Classifier) ที่ต้องการ:",
        list(available_models.keys())
    )
    clf = available_models[selected_model_name]
    st.sidebar.info(f"กำลังใช้งาน: {type(clf).__name__}")
else:
    st.sidebar.error("❌ ไม่พบไฟล์ `.pkl` ในระบบ กรุณานำไฟล์โมเดลมาวางไว้ในโฟลเดอร์เดียวกับโค้ด")

# ==========================================
# 5. ส่วนหลัก: อัปโหลดภาพและทำนายผล
# ==========================================
st.markdown("---")
uploaded_image = st.file_uploader("📸 อัปโหลดภาพ X-ray เท้า (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="ภาพ X-ray ที่อัปโหลด", use_container_width=True)
    
    if st.button("🚀 ประมวลผลวิเคราะห์", use_container_width=True):
        if clf is None:
            st.error("❌ ระบบไม่สามารถประมวลผลได้เนื่องจากไม่ได้เลือกโมเดล หรือไม่มีไฟล์ .pkl ในระบบ")
        else:
            with st.spinner(f"⏳ กำลังวิเคราะห์ด้วยโมเดล {selected_model_name}..."):
                try:
                    # 1. เตรียมภาพและโยนเข้า CNN เพื่อสกัด Feature
                    input_tensor = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = feature_extractor(input_tensor)
                    features_np = features.cpu().numpy()
                    
                    # 2. ให้ Machine Learning ทายผล
                    prediction = clf.predict(features_np)[0]
                    result_text = LABEL_DECODER.get(prediction, f"คลาสไม่รู้จัก ({prediction})")
                    
                    # 3. คำนวณความมั่นใจ (Confidence Score)
                    confidence_text = ""
                    if hasattr(clf, "predict_proba"):
                        probabilities = clf.predict_proba(features_np)[0]
                        confidence = np.max(probabilities) * 100
                        confidence_text = f"ความมั่นใจ (Confidence): {confidence:.2f}%"
                    
                    # แสดงผลลัพธ์
                    st.markdown("### 📊 ผลการวินิจฉัย:")
                    if prediction == 1: 
                        st.error(f"**{result_text}**")
                    else:
                        st.success(f"**{result_text}**")
                        
                    if confidence_text:
                        st.info(confidence_text)
                        
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")