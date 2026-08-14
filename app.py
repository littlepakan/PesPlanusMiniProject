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
import plotly.express as px

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Page Config)
# ==========================================
st.set_page_config(
    page_title="Pes Planus Diagnosis AI",
    page_icon="🦶",
    layout="wide"
)

# ==========================================
# 2. เมนูนำทาง (Sidebar Buttons)
# ==========================================
st.sidebar.title("📌 เมนูนำทาง")
st.sidebar.markdown("คลิกเลือกหัวข้อที่ต้องการอ่านหรือใช้งาน:")

# ใช้ Session State เพื่อจำว่าผู้ใช้กดปุ่มไหน
if 'current_page' not in st.session_state:
    st.session_state.current_page = "1. ปัญหาและ Dataset"

# สร้างปุ่มสำหรับเปลี่ยนหน้า
if st.sidebar.button("👨‍💻 ข้อมูลผู้พัฒนาโปรเจกต์", use_container_width=True):
    st.session_state.current_page = "ข้อมูลผู้พัฒนาโปรเจกต์"
st.sidebar.markdown("---")
if st.sidebar.button("1. ปัญหาและ Dataset", use_container_width=True):
    st.session_state.current_page = "1. ปัญหาและ Dataset"
if st.sidebar.button("2. Data Preprocessing & Validation", use_container_width=True):
    st.session_state.current_page = "2.1. Data Preprocessing & Validation"
if st.sidebar.button("3. ทฤษฎีและการสร้างโมเดล ML", use_container_width=True):
    st.session_state.current_page = "3. ทฤษฎี SqueezeNet & โมเดล ML"
if st.sidebar.button("4. การประเมินผลโมเดล (ตาราง/กราฟ)", use_container_width=True):
    st.session_state.current_page = "4. การประเมินผลโมเดล"
    

st.sidebar.markdown("---")
if st.sidebar.button("🚀 ใช้งานแอปพลิเคชัน (Prediction)", use_container_width=True, type="primary"):
    st.session_state.current_page = "ใช้งานแอปพลิเคชัน"
    
st.sidebar.markdown("---")

st.sidebar.link_button(
    "🐙 GitHub Repository", 
    "https://github.com/your-username/pes-planus-app",  # ✏️ แก้ไขลิงก์ GitHub โปรเจกต์ตรงนี้
    use_container_width=True
)

# ==========================================
# 3. ฟังก์ชันโหลดโมเดล
# ==========================================
@st.cache_resource
def load_ai_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    backbone = backbone.to(device)
    backbone.eval()
    
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
current_page = st.session_state.current_page

# ------------------------------------------
# หน้า 1: ปัญหาและ Dataset
# ------------------------------------------
if current_page == "1. ปัญหาและ Dataset":
    st.title("🦶 การกำหนดปัญหาและ Dataset")
    st.markdown("""
    ### 🛑 ปัญหา (Problem Statement)
    **ภาวะเท้าแบน (Pes Planus)** คือภาวะที่ส่วนโค้งตามยาวด้านในของเท้า (Medial Longitudinal Arch) ลดลงหรือแบนราบไปกับพื้น 
    ซึ่งหากปล่อยทิ้งไว้อาจส่งผลให้เกิดอาการปวดข้อเท้า เข่า สะโพก และหลังได้ การวินิจฉัยในปัจจุบันมักอาศัยการตรวจทางคลินิกและการวัดมุมจากภาพถ่ายรังสีเอกซ์ (X-ray) โดยแพทย์ผู้เชี่ยวชาญ ซึ่งใช้เวลาและอาจมีความคลาดเคลื่อนได้

    ### 📊 ข้อมูลที่นำมาใช้ (Dataset)
    * **ทำไมถึงเลือกใช้ข้อมูลชุดนี้?:** เราเลือกใช้ภาพถ่าย X-ray บริเวณเท้า (Lateral Weight-bearing Radiographs) เพราะเป็นมาตรฐานทองคำ (Gold Standard) ในการวินิจฉัยโครงสร้างกระดูก เช่น การดูมุม *Calcaneal Inclusion Angle*
    * **รายละเอียดข้อมูล:** ประกอบด้วยภาพ X-ray เท้าของผู้ป่วยที่ถูกจัดกลุ่ม (Label) โดยรังสีแพทย์ แบ่งเป็น 2 คลาส คือ:
        1. `Class 0`: Normal (เท้าปกติ)
        2. `Class 1`: Pes Planus (เท้าแบน)
    """)

# ------------------------------------------
# หน้า 2: Data Preprocessing & Validation Strategy
# ------------------------------------------
elif current_page == "2. Data Preprocessing & Validation":
    st.title("⚙️ Data Preprocessing & K-Fold Cross Validation")
    st.markdown("""
    ### 🛠️ 1. กระบวนการเตรียมข้อมูล (Data Preprocessing)
    ก่อนนำภาพเข้าสู่โมเดลปัญญาประดิษฐ์ เราได้ทำการปรับแต่งข้อมูลเพื่อให้โมเดลเรียนรู้ได้ดีที่สุด ดังนี้:

    1. **การปรับขนาดภาพ (Resizing):** ปรับขนาดภาพ X-ray ทุกใบให้มีขนาด `224 x 224 pixels` เพื่อให้เข้ากับโครงสร้างของ SqueezeNet
    2. **การแปลงเป็นเทนเซอร์ (ToTensor):** แปลงค่าพิกเซลของภาพ (0-255) ให้อยู่ในรูปของ Matrix ตัวเลขทศนิยม (0-1)
    3. **การปรับมาตรฐานสี (Normalization):** ปรับค่า Mean และ Standard Deviation ของภาพให้ตรงกับมาตรฐาน ImageNet `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
    4. **การสกัดคุณลักษณะ (Feature Extraction):** นำภาพป้อนเข้าสู่โมเดล CNN (SqueezeNet) เพื่อแปลงภาพ 2 มิติ ให้กลายเป็นชุดตัวเลขเวกเตอร์ (Deep Features)
    5. **การจัดการคลาสที่ไม่สมดุล (SMOTE):** ใช้เทคนิค Synthetic Minority Over-sampling Technique (SMOTE) เพื่อปรับสมดุลข้อมูลคลาสเท้าแบนและเท้าปกติก่อนเทรนโมเดล

    ---

    ### 🔄 2. กลยุทธ์การประเมินผลด้วย 5-Fold Cross Validation
    เพื่อยืนยันว่าโมเดลมีความแม่นยำและเสถียรจริง ไม่ได้เกิดจากความฟลุก เราจึงเลือกใช้กลยุทธ์ **5-Fold Cross Validation** ในการทดลอง:

    * **วิธีการแบ่งข้อมูล:** แบ่งชุดข้อมูลทั้งหมดออกเป็น **5 ส่วนเท่า ๆ กัน (Fold 1 ถึง Fold 5)**
    * **การทำงานในแต่ละรอบ:**
        * ในแต่ละรอบ จะใช้ข้อมูล **4 ส่วน (80%)** ในการเทรนโมเดล (Training Set)
        * ใช้ข้อมูลอีก **1 ส่วน (20%)** ที่เหลือเป็นชุดทดสอบ (Validation Set)
        * ทำซ้ำกระบวนการนี้จนครบทั้ง 5 รอบ โดยสลับชุด Validation ไปเรื่อย ๆ
    * **ประโยชน์ที่ได้:** ข้อมูลภาพ X-ray ทุกภาพจะได้รับการทดสอบจริง และค่าประสิทธิภาพทั้งหมดจะถูกนำมา **หาค่าเฉลี่ย (Average Scores)** เพื่อใช้เป็นตัวตัวชี้วัดความแม่นยำที่แท้จริงของโมเดล
    """)

# ------------------------------------------
# หน้า 3: ทฤษฎี SqueezeNet & โมเดล ML (✨ เพิ่มการอธิบาย SqueezeNet)
# ------------------------------------------
elif current_page == "3. ทฤษฎี SqueezeNet & โมเดล ML":
    st.title("🧠 ทฤษฎีระบบจำแนกและสถาปัตยกรรมโมเดล")
    st.markdown("""
    โครงงานนี้ใช้สถาปัตยกรรมประมวลผลแบบ **Hybrid Approach (CNN Feature Extraction + Machine Learning Classifiers)** โดยแบ่งหน้าที่การทำงานออกเป็น 2 ส่วนหลัก:
    """)

    st.markdown("""
    ### 🧩 1. สถาปัตยกรรม SqueezeNet (CNN Feature Extractor)
    * **SqueezeNet คืออะไร?:** เป็นโครงข่ายประสาทคอนโวลูชัน (Convolutional Neural Network: CNN) ขนาดเล็กและน้ำหนักเบา (Lightweight Deep Learning Model) ที่ถูกออกแบบโดย Iandola et al. เพื่อให้มีความแม่นยำระดับใกล้เคียงกับ AlexNet แต่ใช้พารามิเตอร์น้อยกว่าถึง **50 เท่า** (ขนาดไฟล์โมเดลเล็กกว่า 5 MB)
    * **กลไกการทำงานหลัก (Fire Module):**
        1. **Squeeze Layer:** ใช้ตัวกรอง Convolution ขนาด $1 \\times 1$ ในการบีบอัด (Squeeze) จำนวน Channel ของข้อมูลให้ลดลง เพื่อลดภาระการคำนวณ
        2. **Expand Layer:** ใช้ตัวกรอง Convolution ขนาด $1 \\times 1$ ผสมกับ $3 \\times 3$ ในการขยาย (Expand) และสกัดฟีเจอร์เชิงลึกจากภาพ X-ray
    * **บทบาทในโครงงานนี้ (Transfer Learning Backbone):**
        * เราใช้โมเดล SqueezeNet v1.0 ที่ผ่านการเทรนล่วงหน้า (Pre-trained on ImageNet)
        * ปลดเลเยอร์จำแนกประเภทท้ายสุด (Classifier Layer) ออก แล้วเปลี่ยนเป็น Adaptive Average Pooling ร่วมกับ Flatten Layer
        * SqueezeNet จะทำหน้าที่เป็น **"ดวงตา" สกัดคุณลักษณะเชิงลึก (Deep Visual Features)** แปลงภาพ X-ray เท้าขนาด $224 \\times 224 \\times 3$ พิกเซล ให้ออกมาเป็นเวกเตอร์ฟีเจอร์ความยาว 512 มิติ ก่อนส่งต่อให้ตัวแบบ Machine Learning ตัดสินใจ
    """)

    st.markdown("""
    ---
    ### 🎯 2. โมเดลการเรียนรู้ของเครื่อง (Machine Learning Classifiers)
    รับเวกเตอร์ฟีเจอร์ 512 มิติที่สกัดได้จาก SqueezeNet มาทำหน้าที่เป็น **"สมอง" ในการจำแนกคลาส (Normal vs Pes Planus)** โดยทดสอบเปรียบเทียบ 3 อัลกอริทึม:

    #### 2.1 Support Vector Machine (SVM)
    * **หลักการทำงาน:** สร้างระนาบหรือเส้นแบ่งระดับสูง (Hyperplane) ในพื้นที่หลายมิติ เพื่อแบ่งกลุ่มข้อมูลสองคลาสออกจากกัน ให้มีระยะห่างจากจุดข้อมูลที่ใกล้ที่สุด (Margin) กว้างที่สุด
    * **การประยุกต์ใช้:** เลือกใช้ **RBF Kernel (Radial Basis Function)** เพื่อรองรับการแบ่งข้อมูลเวกเตอร์ฟีเจอร์ของภาพ X-ray ที่มีความซับซ้อนเชิงเส้นแบบไม่เป็นเชิงตรง (Non-linear)

    #### 2.2 Decision Tree (ต้นไม้ตัดสินใจ)
    * **หลักการทำงาน:** สร้างโครงสร้างเงื่อนไขการตัดสินใจ (If-Else Rules) แบบลำดับชั้น ตั้งแต่ราก (Root Node) ไปจนถึงใบ (Leaf Node) โดยใช้เกณฑ์ทางสถิติ (เช่น Gini Impurity) ในการเลือกฟีเจอร์ที่แบ่งกลุ่มข้อมูลได้ดีที่สุด
    * **จุดเด่น:** สามารถอธิบายกระบวนการคิดของโมเดลได้ชัดเจน (Interpretable AI)

    #### 2.3 K-Nearest Neighbors (KNN)
    * **หลักการทำงาน:** จำแนกประเภทข้อมูลใหม่โดยการคำนวณระยะห่าง (Euclidean Distance) ไปยังจุดข้อมูลที่มีอยู่แล้วในระบบ และตัดสินคลาสตามเสียงส่วนใหญ่ของ **"เพื่อนบ้านที่อยู่ใกล้ที่สุด K ตัว"**
    * **การประยุกต์ใช้:** ในโครงงานนี้กำหนดค่า $K = 5$ เพื่อความสมดุลระหว่างความไวและความคงที่ในการตัดสินใจ
    """)

# ------------------------------------------
# หน้า 4: การประเมินและเปรียบเทียบโมเดล
# ------------------------------------------
elif current_page == "4. การประเมินผลโมเดล":
    st.title("📊 การประเมินและเปรียบเทียบประสิทธิภาพ")
    st.markdown("ผลการทดลองจากการทำ **5-Fold Cross Validation** เพื่อเปรียบเทียบโมเดลทั้ง 3 อัลกอริทึม")

    csv_file = "5fold_ml_classification_results.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        st.subheader("📈 กราฟเปรียบเทียบ Accuracy ทุก Folds")
        df_folds = df[df['Fold'] != 'Average'].copy()
        
        if not df_folds.empty:
            fig = px.bar(
                df_folds, 
                x="Fold", 
                y="Accuracy", 
                color="Classifier", 
                barmode="group",
                text_auto='.2%',
                title="Accuracy Performance Across 5 Folds"
            )
            fig.update_layout(
                yaxis_title="Accuracy",
                xaxis_title="Validation Fold",
                yaxis_tickformat='.0%',
                legend_title_text='Machine Learning Model'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        df_avg = df[df['Fold'] == 'Average'].copy()
        if not df_avg.empty:
            st.subheader("📋 ตารางสรุปค่าเฉลี่ย 5 Folds (Average Scores)")
            st.dataframe(df_avg.style.format({
                'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
                'Specificity': '{:.4f}', 'F1-Score': '{:.4f}', 'AUC': '{:.4f}', 'MCC': '{:.4f}'
            }), use_container_width=True)
            
        with st.expander("ดูตารางข้อมูลดิบทั้งหมด (Raw Data)"):
             st.dataframe(df)

    else:
        st.warning(f"⚠️ ไม่พบไฟล์ `{csv_file}` ในโฟลเดอร์")

# ------------------------------------------
# หน้า 5: ข้อมูลผู้พัฒนาโปรเจกต์ (✨ เพิ่มใหม่)
# ------------------------------------------
elif current_page == "5. ข้อมูลผู้พัฒนาโปรเจกต์":
    st.title("👨‍💻 ข้อมูลผู้พัฒนาโปรเจกต์")
    st.markdown("---")

    # กำหนดข้อมูลผู้พัฒนา (สามารถปรับแก้ชื่อ, รหัส, หมู่เรียน, พาร์ทรูปภาพ ได้ตรงนี้เลยครับ)
    developers = [
        {
            "name": "นายปกานต์ วงษ์ท่าเรือ",          # ✏️ เปลี่ยนเป็น ชื่อ-นามสกุล ของคุณ
            "id": "664245056",               # ✏️ เปลี่ยนเป็น รหัสนักศึกษา
            "class_group": "66/44",      # ✏️ เปลี่ยนเป็น หมู่เรียน
            "role": "นักศึกษา / ผู้พัฒนา",  # ✏️ เปลี่ยนเป็น บทบาทของคุณในโปรเจกต์
            "img": "p.png",              # ✏️ นำไฟล์รูปภาพมาวางในโฟลเดอร์เดียวกับ app.py แล้วแก้ชื่อตรงนี้
            "github": "https://github.com/littlepakan"
        },
        # ถ้ามีเพื่อนร่วมกลุ่มเพิ่ม ให้ปลดคอมเมนต์ก้อนนี้แล้วแก้ไขได้เลยครับ:
        # {
        #     "name": "นาย... ชื่อเพื่อน",
        #     "id": "65xxxxxxx-y",
        #     "class_group": "หมู่เรียน 65/xx",
        #     "role": "ผู้พัฒนาโปรเจกต์ / Data Engineer",
        #     "img": "profile2.jpg"
        # }
    ]

    # วาดการ์ดผู้พัฒนา
    for dev in developers:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if os.path.exists(dev["img"]):
                img = Image.open(dev["img"])
                st.image(img, use_container_width=True)
            else:
                # กรณีที่ยังไม่มีไฟล์รูป ระบบจะใช้รูป Placeholder แทนให้อัตโนมัติ
                st.image("https://via.placeholder.com/300x350?text=Profile+Image", caption="รออัปโหลดรูปภาพ", use_container_width=True)
                st.caption(f"💡 นำไฟล์รูปวางในโฟลเดอร์ชื่อ `{dev['img']}` เพื่อแสดงรูปจริง")
                
        with col2:
            st.subheader(dev["name"])
            st.markdown(f"**🆔 รหัสนักศึกษา:** {dev['id']}")
            st.markdown(f"**🏫 หมู่เรียน:** {dev['class_group']}")
            st.markdown(f"**💻 บทความ/หน้าที่:** {dev['role']}")
            st.markdown(f"**💻 GitHub:** {dev['github']}")
            st.markdown("---")
            st.markdown("**สาขาวิทยาการคอมพิวเตอร์**")
            st.markdown("**คณะวิทยาศาสตร์และเทคโนโลยี**")
            st.markdown("**มหาวิทยาลัยราชภัฏนครปฐม**")
            
        st.markdown("---")

# ------------------------------------------
# หน้า 6: Application (Prediction)
# ------------------------------------------
elif current_page == "6. ใช้งานแอปพลิเคชัน":
    st.title("🔍 ระบบปัญญาประดิษฐ์คัดกรองภาวะเท้าแบน")
    st.markdown("ทดสอบอัปโหลดภาพถ่ายรังสีเอกซ์ (X-ray) บริเวณเท้า เพื่อให้โมเดลวิเคราะห์")

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

    st.subheader("📸 2. อัปโหลดภาพ X-ray")
    uploaded_image = st.file_uploader("รองรับไฟล์ JPG, JPEG, PNG", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="ภาพ X-ray ที่อัปโหลด", use_container_width=True)
        
        with col2:
            st.write("") 
            st.write("") 
            if st.button("🚀 เริ่มวิเคราะห์ (Predict)", use_container_width=True, type="primary"):
                if clf is None:
                    st.error("กรุณาเลือกโมเดลก่อน")
                else:
                    with st.spinner("⏳ กำลังสกัด Feature และคำนวณผล..."):
                        try:
                            input_tensor = preprocess(image).unsqueeze(0).to(device)
                            with torch.no_grad():
                                features = feature_extractor(input_tensor)
                            features_np = features.cpu().numpy()
                            
                            prediction = clf.predict(features_np)[0]
                            result_text = LABEL_DECODER.get(prediction, "Unknown")
                            
                            confidence_text = ""
                            if hasattr(clf, "predict_proba"):
                                prob = clf.predict_proba(features_np)[0]
                                confidence = np.max(prob) * 100
                                confidence_text = f"ความมั่นใจ (Confidence): {confidence:.2f}%"
                            
                            st.markdown("### 📊 ผลการวินิจฉัย:")
                            if prediction == 1: 
                                st.error(f"🚨 **{result_text}**")
                            else:
                                st.success(f"✅ **{result_text}**")
                                
                            if confidence_text:
                                st.info(confidence_text)
                                
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {e}")