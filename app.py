import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import os
import glob

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ==========================================
st.set_page_config(
    page_title="Pes Planus Diagnosis AI",
    page_icon="🦶",
    layout="wide" # ใช้ layout แบบ wide เพื่อให้แสดงตาราง/กราฟได้สวยขึ้น
)

# ==========================================
# 2. เมนูนำทาง (Sidebar Navigation)
# ==========================================
st.sidebar.title("📌 เมนูนำทาง (Navigation)")
menu = st.sidebar.radio(
    "เลือกหัวข้อที่ต้องการ:",
    (
        "1. ปัญหาและ Dataset",
        "2. Data Preprocessing",
        "3. ทฤษฎีและการสร้างโมเดล ML",
        "4. การประเมินและเปรียบเทียบโมเดล",
        "5. 🚀 ใช้งานแอปพลิเคชัน (Prediction)"
    )
)

st.sidebar.markdown("---")
st.sidebar.info("แอปพลิเคชันนี้เป็นส่วนหนึ่งของโครงงานการคัดกรองภาวะเท้าแบนด้วยปัญญาประดิษฐ์")

# ==========================================
# 3. ฟังก์ชันโหลดโมเดล (Cache ไว้จะได้ไม่โหลดซ้ำ)
# ==========================================
@st.cache_resource
def load_ai_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Backbone
    backbone = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Classifiers
    pkl_files = glob.glob("*.pkl")
    models_dict = {}
    for file_path in pkl_files:
        try:
            with open(file_path, 'rb') as f:
                model_name = file_path.replace('.pkl', '')
                models_dict[model_name] = pickle.load(f)
        except: pass
    return backbone, device, models_dict

feature_extractor, device, available_models = load_ai_models()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABEL_DECODER = {0: "Normal (ปกติ)", 1: "Pes Planus (ภาวะเท้าแบน)"}

# ==========================================
# 4. ส่วนแสดงผลตามเมนูที่เลือก (Pages)
# ==========================================

# ------------------------------------------
# หน้า 1: ปัญหาและ Dataset
# ------------------------------------------
if menu == "1. ปัญหาและ Dataset":
    st.title("🦶 การกำหนดปัญหาและ Dataset")
    st.markdown("""
    ### 🛑 ปัญหา (Problem Statement)
    **ภาวะเท้าแบน (Pes Planus)** คือภาวะที่ส่วนโค้งตามยาวด้านในของเท้า (Medial Longitudinal Arch) ลดลงหรือแบนราบไปกับพื้น 
    ซึ่งหากปล่อยทิ้งไว้อาจส่งผลให้เกิดอาการปวดข้อเท้า เข่า สะโพก และหลังได้ การวินิจฉัยในปัจจุบันมักอาศัยการตรวจทางคลินิกและการวัดมุมจากภาพถ่ายรังสีเอกซ์ (X-ray) โดยแพทย์ผู้เชี่ยวชาญ ซึ่งใช้เวลาและอาจมีความคลาดเคลื่อนได้

    ### 📊 ข้อมูลที่นำมาใช้ (Dataset)
    * **ทำไมถึงเลือกใช้ข้อมูลชุดนี้?:** เราเลือกใช้ภาพถ่าย X-ray บริเวณเท้า (Lateral Weight-bearing Radiographs) เพราะเป็นมาตรฐานทองคำ (Gold Standard) ในการวินิจฉัยโครงสร้างกระดูก เช่น การดูมุม *Calcaneal Inclusion Angle* * **รายละเอียดข้อมูล:** ประกอบด้วยภาพ X-ray เท้าของผู้ป่วยที่ถูกจัดกลุ่ม (Label) โดยรังสีแพทย์ แบ่งเป็น 2 คลาส คือ:
        1. `Class 0`: Normal (เท้าปกติ)
        2. `Class 1`: Pes Planus (เท้าแบน)
    """)

# ------------------------------------------
# หน้า 2: Data Preprocessing
# ------------------------------------------
elif menu == "2. Data Preprocessing":
    st.title("⚙️ Data Preprocessing (การเตรียมข้อมูล)")
    st.markdown("""
    ก่อนนำภาพเข้าสู่โมเดลปัญญาประดิษฐ์ เราได้ทำการปรับแต่งข้อมูล (Preprocessing) เพื่อให้โมเดลเรียนรู้ได้ดีที่สุด ดังนี้:

    1. **การปรับขนาดภาพ (Resizing):** ปรับขนาดภาพ X-ray ทุกใบให้มีขนาด `224 x 224 pixels` ซึ่งเป็นขนาดมาตรฐานสำหรับโครงข่ายประสาทเทียม
    2. **การแปลงเป็นเทนเซอร์ (ToTensor):** แปลงค่าพิกเซลของภาพ (0-255) ให้อยู่ในรูปของ Matrix ตัวเลขทศนิยม (0-1) ที่ PyTorch สามารถประมวลผลได้
    3. **การปรับมาตรฐานสี (Normalization):** ปรับค่า Mean และ Standard Deviation ของภาพให้ตรงกับมาตรฐาน ImageNet `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` เพื่อลดสัญญาณรบกวน (Noise) ของแสงและเงา
    4. **การสกัดคุณลักษณะ (Feature Extraction):** นำภาพที่ปรับแต่งแล้ว ป้อนเข้าสู่โมเดล CNN (SqueezeNet) เพื่อแปลงภาพ 2 มิติ ให้กลายเป็นชุดตัวเลขเวกเตอร์ (Deep Features)
    5. **การจัดการคลาสที่ไม่สมดุล (SMOTE):** ใช้เทคนิค Synthetic Minority Over-sampling Technique (SMOTE) เพื่อสร้างข้อมูลจำลองของคลาสที่มีจำนวนน้อยกว่า ทำให้โมเดลไม่เอนเอียง (Bias) ไปทางคลาสใดคลาสหนึ่ง
    """)

# ------------------------------------------
# หน้า 3: ทฤษฎีและการสร้างโมเดล ML
# ------------------------------------------
elif menu == "3. ทฤษฎีและการสร้างโมเดล ML":
    st.title("🧠 การสร้างโมเดล ML และทฤษฎี")
    st.markdown("""
    โปรเจกต์นี้ใช้สถาปัตยกรรมแบบ **Hybrid Approach (CNN + Traditional ML)** โดยให้ SqueezeNet ทำหน้าที่เป็น "ดวงตา" สกัดจุดเด่นของภาพ 
    และให้ Machine Learning 3 ตัวนี้ทำหน้าที่เป็น "สมอง" ในการตัดสินใจจำแนกโรค:

    ### 1. Support Vector Machine (SVM)
    * **ทฤษฎี:** SVM สร้างเส้นแบ่ง (Hyperplane) หรือระนาบในพื้นที่หลายมิติ เพื่อแบ่งกลุ่มข้อมูล (Normal กับ Pes Planus) ออกจากกันให้มีระยะห่าง (Margin) กว้างที่สุด
    * **การตั้งค่า:** ใช้ RBF Kernel เพื่อรองรับการแบ่งข้อมูลแบบไม่เป็นเส้นตรง (Non-linear)

    ### 2. Decision Tree
    * **ทฤษฎี:** โครงสร้างต้นไม้ตัดสินใจ ทำงานโดยการสร้างเงื่อนไข (If-Else) แบบลำดับชั้น จากราก (Root) ไปสู่ใบ (Leaf) เพื่อจำแนกประเภทข้อมูล
    * **จุดเด่น:** เข้าใจง่ายและสามารถอธิบายได้ว่าโมเดลใช้ฟีเจอร์ใดเป็นตัวแบ่งคลาสหลัก

    ### 3. K-Nearest Neighbors (KNN)
    * **ทฤษฎี:** เป็นโมเดลที่เรียบง่ายที่สุด โดยจะเปรียบเทียบข้อมูลใหม่กับข้อมูลเดิมทั้งหมด และตัดสินใจโดยดูจาก "เพื่อนบ้านที่ใกล้ที่สุด" จำนวน K ตัว (ในที่นี้ใช้ K=5)
    * **จุดเด่น:** ทำงานได้ดีเมื่อชุดคุณลักษณะ (Features) มีการเกาะกลุ่มกันชัดเจน
    """)

# ------------------------------------------
# หน้า 4: การประเมินและเปรียบเทียบโมเดล
# ------------------------------------------
elif menu == "4. การประเมินและเปรียบเทียบโมเดล":
    st.title("📊 การประเมินและเปรียบเทียบประสิทธิภาพ")
    st.markdown("ตารางและกราฟแสดงผลการทดลองจากการทำ **5-Fold Cross Validation** เพื่อเปรียบเทียบโมเดลทั้ง 3 อัลกอริทึม")

    # พยายามโหลดไฟล์ CSV
    csv_file = "5fold_ml_classification_results.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        # กรองเอาเฉพาะข้อมูลค่าเฉลี่ย (Average)
        df_avg = df[df['Fold'] == 'Average'].copy()
        
        if not df_avg.empty:
            st.subheader("ตารางเปรียบเทียบค่าเฉลี่ย (Average Scores)")
            # จัดรูปแบบตารางให้สวยงาม
            st.dataframe(df_avg.style.format({
                'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
                'Specificity': '{:.4f}', 'F1-Score': '{:.4f}', 'AUC': '{:.4f}', 'MCC': '{:.4f}'
            }), use_container_width=True)

            st.subheader("📈 กราฟเปรียบเทียบ (Accuracy, F1-Score, AUC)")
            # เตรียมข้อมูลสำหรับ Plot กราฟ
            chart_data = df_avg.set_index("Classifier")[["Accuracy", "F1-Score", "AUC"]]
            st.bar_chart(chart_data)
        else:
            st.write("ตารางข้อมูลดิบ (All Folds):")
            st.dataframe(df)
            
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ `{csv_file}` ในโฟลเดอร์ กรุณานำไฟล์ CSV ที่ได้จาก Colab มาวางไว้ที่เดียวกับ `app.py` เพื่อแสดงผลตารางและกราฟครับ")

# ------------------------------------------
# หน้า 5: Application (Prediction)
# ------------------------------------------
elif menu == "5. 🚀 ใช้งานแอปพลิเคชัน (Prediction)":
    st.title("🔍 ระบบปัญญาประดิษฐ์คัดกรองภาวะเท้าแบน")
    st.markdown("ทดสอบอัปโหลดภาพถ่ายรังสีเอกซ์ (X-ray) บริเวณเท้า เพื่อให้โมเดลวิเคราะห์")

    # ส่วนเลือกโมเดล (Dropdown)
    st.subheader("⚙️ 1. เลือกโมเดลที่ต้องการใช้งาน")
    clf = None
    if available_models:
        selected_model_name = st.selectbox(
            "เลือกโมเดล (Classifier):",
            list(available_models.keys())
        )
        clf = available_models[selected_model_name]
        st.success(f"✅ ระบบพร้อมวิเคราะห์ด้วยโมเดล: **{selected_model_name}**")
    else:
        st.error("❌ ไม่พบไฟล์ `.pkl` ในระบบ กรุณานำไฟล์โมเดลมาวางไว้ในโฟลเดอร์เดียวกับโค้ด")

    # ส่วนอัปโหลดรูป
    st.subheader("📸 2. อัปโหลดภาพ X-ray")
    uploaded_image = st.file_uploader("รองรับไฟล์ JPG, JPEG, PNG", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="ภาพ X-ray ที่อัปโหลด", use_container_width=True)
        
        with col2:
            st.write("") # เว้นบรรทัด
            st.write("") 
            if st.button("🚀 เริ่มวิเคราะห์ (Predict)", use_container_width=True):
                if clf is None:
                    st.error("กรุณาเลือกโมเดลก่อน")
                else:
                    with st.spinner("⏳ กำลังสกัด Feature และคำนวณผล..."):
                        try:
                            # สกัด Feature
                            input_tensor = preprocess(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                features = feature_extractor(input_tensor)
                            features_np = features.cpu().numpy()
                            
                            # ทำนายผล
                            prediction = clf.predict(features_np)[0]
                            result_text = LABEL_DECODER.get(prediction, "Unknown")
                            
                            # คำนวณความมั่นใจ
                            confidence_text = ""
                            if hasattr(clf, "predict_proba"):
                                prob = clf.predict_proba(features_np)[0]
                                confidence = np.max(prob) * 100
                                confidence_text = f"ความมั่นใจ (Confidence): {confidence:.2f}%"
                            
                            # แสดงผล
                            st.markdown("### 📊 ผลการวินิจฉัย:")
                            if prediction == 1: 
                                st.error(f"🚨 **{result_text}**")
                            else:
                                st.success(f"✅ **{result_text}**")
                                
                            if confidence_text:
                                st.info(confidence_text)
                                
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {e}")