import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import processor
import metrics

# ตั้งค่าหน้าเว็บสไตล์คลินิกพรีเมียม
st.set_page_config(
    page_title="Dexterity AI Analyzer",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# สไตล์ CSS ตกแต่ง Dashboard รูปแบบการแพทย์ล้ำสมัย
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 24px;
        color: #f8fafc;
        font-weight: bold;
    }
    h1 {
        color: #0ea5e9;
        font-weight: 800;
    }
    h3 {
        color: #f1f5f9;
        margin-top: 15px;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# หัวข้อระบบภาษาไทย
st.title("Dexterity AI Analyzer: ระบบตรวจวัดภาวะละเลยการใช้งานมือ")
st.markdown("ประเมินดัชนีชี้วัดทางชีวภาพแบบดิจิทัล (ความเร็วข้อมือ, ระยะหนีบนิ้ว และความกระตุก) เพื่อตรวจวิเคราะห์ภาวะ **Learned Non-Use (LNU)** ในผู้สูงอายุและผู้ป่วยกายภาพบำบัด")
st.markdown("---")

# ส่วนแถบควบคุมด้านข้าง (Sidebar)
st.sidebar.header("⚙️ ตั้งค่าการวิเคราะห์")
video_source = st.sidebar.radio("เลือกแหล่งข้อมูลภาพ:", ["กรณีศึกษาตัวอย่าง (โฟลเดอร์ data)", "อัปโหลดไฟล์วิดีโอ (.MP4)", "กล้องเว็บแคมสด (Live Web Camera)"])

# สร้างโฟลเดอร์เก็บข้อมูลวิดีโอหากยังไม่มี
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

video_file_path = None
use_mock = False

if video_source == "อัปโหลดไฟล์วิดีโอ (.MP4)":
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์วิดีโอการเคลื่อนไหวมือ (MP4)", type=["mp4"])
    if uploaded_file is not None:
        video_file_path = os.path.join(data_dir, "temp_uploaded.mp4")
        with open(video_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("อัปโหลดไฟล์วิดีโอสำเร็จ!")
elif video_source == "กล้องเว็บแคมสด (Live Web Camera)":
    camera_index = st.sidebar.number_input("หมายเลขกล้อง (Camera Index)", min_value=0, max_value=10, value=0, help="ปกติคือ 0 (กล้องหลักของเครื่อง) หรือเลือก 1, 2 หากต่อกล้องเสริมภายนอก")
    video_file_path = str(camera_index)
    use_mock = False
    st.sidebar.info(f"เชื่อมต่อกล้องเว็บแคมตัวที่ {camera_index} เรียบร้อยแล้ว กดปุ่ม 'เริ่มวิเคราะห์ข้อมูล' เพื่อเริ่มบันทึก")
else:
    default_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
    if not default_files:
        st.sidebar.warning("ไม่พบไฟล์ MP4 ในโฟลเดอร์ data/")
        sim_choice = st.sidebar.selectbox("เลือกผู้ป่วยจำลอง:", ["ผู้ป่วย #60 (จำลองอาการละเลยมือซ้าย)", "ผู้ป่วย #67 (จำลองอาการละเลยมือขวา)"])
        use_mock = True
        st.sidebar.info("ระบบจะรันในโหมดจำลองพิกัดมือผู้ป่วยทางคลินิก")
    else:
        selected_file = st.sidebar.selectbox("เลือกวิดีโอกรณีศึกษาผู้ป่วย:", default_files)
        video_file_path = os.path.join(data_dir, selected_file)
        st.sidebar.success(f"เลือกกรณีศึกษา: {selected_file}")

st.sidebar.markdown("---")
st.sidebar.header("🔬 กำหนดเกณฑ์ทางคลินิก")
speed_threshold = st.sidebar.slider("เกณฑ์ความเร็วเฉลี่ย (พิกเซล/วินาที)", min_value=10.0, max_value=300.0, value=100.0, step=5.0, help="หากความเร็วเฉลี่ยต่ำกว่าเกณฑ์นี้ จะประเมินว่าขยับมือช้ากว่าปกติ")
jerk_threshold = st.sidebar.slider("เกณฑ์ดัชนีความกระตุก (พิกเซล/วินาที³)", min_value=500.0, max_value=20000.0, value=8000.0, step=500.0, help="หากดัชนีความกระตุกเฉลี่ยสูงกว่าเกณฑ์นี้ แสดงว่าการขยับมือสั่นและกระตุกมาก")

# ปุ่มเริ่มการทำงาน
analyze_button = st.sidebar.button("เริ่มวิเคราะห์ข้อมูล", type="primary")

# โครงสร้างแบบสองคอลัมน์หลัก
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🎥 ภาพบันทึกพิกัดข้อต่อมือแบบเรียลไทม์")
    video_placeholder = st.empty()
    video_placeholder.info("กรุณากดปุ่ม 'เริ่มวิเคราะห์ข้อมูล' เพื่อเริ่มแสดงพิกัดข้อต่อข้อมือและปลายนิ้วมือในการเคลื่อนไหว")

with col_right:
    st.subheader("📊 รายงานดัชนีชี้วัดและการวินิจฉัย")
    
    # บล็อกวินิจฉัยทางการแพทย์
    st.markdown("### 📋 สรุปผลการตรวจวินิจฉัยโรค")
    diagnostics_placeholder = st.empty()
    diagnostics_placeholder.info("รอการสั่งรันประมวลผลข้อมูล")
    
    tab_r, tab_l = st.tabs(["🔵 ดัชนีมือขวา", "🟠 ดัชนีมือซ้าย"])
    with tab_r:
        r_speed_card = st.empty()
        r_accuracy_card = st.empty()
        r_jerk_card = st.empty()
    with tab_l:
        l_speed_card = st.empty()
        l_accuracy_card = st.empty()
        l_jerk_card = st.empty()
        
    # วาดการ์ดเปล่าเริ่มต้น
    for card in [r_speed_card, r_accuracy_card, r_jerk_card, l_speed_card, l_accuracy_card, l_jerk_card]:
        card.markdown('<div class="metric-card"><div class="metric-label">ดัชนีประเมิน</div><div class="metric-value">--</div></div>', unsafe_allow_html=True)

    st.markdown("### 📈 กราฟแนวโน้มความเคลื่อนไหว")
    speed_chart_header = st.empty()
    speed_chart = st.empty()
    
    pinch_chart_header = st.empty()
    pinch_chart = st.empty()
    
    jerk_chart_header = st.empty()
    jerk_chart = st.empty()

# เริ่มขั้นตอนวิเคราะห์เมื่อผู้ใช้งานกดปุ่ม
if analyze_button:
    # เลือกเส้นทางไฟล์วิดีโอ
    active_path = video_file_path if (video_file_path and (video_source == "กล้องเว็บแคมสด (Live Web Camera)" or os.path.exists(video_file_path))) else None
    
    progress_bar = st.sidebar.progress(0.0)
    st.sidebar.text("ระบบกำลังทำการประมวลผลเฟรมวิดีโอ...")
    
    # เก็บข้อมูลพิกัดที่ดึงได้สดระหว่างรันสตรีม
    records = []
    
    fps = 30.0
    total_frames = 300
    
    try:
        # ดึงสตรีมภาพและพิกัดข้อต่อมือแบบ Generator
        stream = processor.process_video_stream(active_path, use_mock=use_mock)
        
        # ปรับการดึงเฟรมให้ตรงกับวิดีโอจริง (ยกเว้นกรณีกล้องเว็บแคมสตรีม)
        is_webcam = (video_source == "กล้องเว็บแคมสด (Live Web Camera)")
        if active_path and not use_mock and not is_webcam:
            cap = cv2.VideoCapture(active_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        elif is_webcam:
            fps = 30.0
            total_frames = 300

        frame_idx = 0
        for data in stream:
            # อัปเดตแถบเปอร์เซ็นต์ความคืบหน้า
            progress_bar.progress(data["progress"])
            
            # อัปเดตเฟรมวิดีโอข้อต่อบนหน้า UI
            video_placeholder.image(data["frame"], channels="RGB", use_container_width=True)
            
            # บันทึกค่าพิกัดดิบ Wrist, Thumb Tip และ Index Finger Tip เที่ยวเดียว (Single Pass)
            hand_data = data.get("hand_data", {})
            for hand, lm_dict in hand_data.items():
                records.append({
                    "frame": frame_idx,
                    "hand_label": hand,
                    "wrist_x": float(lm_dict["wrist"][0]),
                    "wrist_y": float(lm_dict["wrist"][1]),
                    "thumb_x": float(lm_dict["thumb_tip"][0]),
                    "thumb_y": float(lm_dict["thumb_tip"][1]),
                    "index_x": float(lm_dict["index_tip"][0]),
                    "index_y": float(lm_dict["index_tip"][1])
                })
            
            frame_idx += 1
            
        st.sidebar.success("การรับส่งข้อมูลวิดีโอเสร็จสิ้น!")
        progress_bar.progress(1.0)
        
        st.sidebar.text("กำลังคำนวณไบโอมาร์กเกอร์และพล็อตกราฟ...")
        
        # แปลงข้อมูลบันทึกพิกัดมือเป็น Pandas DataFrame
        if records:
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(columns=['frame', 'hand_label', 'wrist_x', 'wrist_y', 'thumb_x', 'thumb_y', 'index_x', 'index_y'])
            
        # แยกข้อมูลตำแหน่งพิกัดของมือแต่ละข้าง
        df_left = df[df["hand_label"] == "Left"].sort_values("frame")
        df_right = df[df["hand_label"] == "Right"].sort_values("frame")
        
        l_avg_speed, l_speeds = 0.0, np.zeros(0)
        l_avg_pinch, l_pinches = 0.0, np.zeros(0)
        l_avg_jerk, l_jerks = 0.0, np.zeros(0)
        
        r_avg_speed, r_speeds = 0.0, np.zeros(0)
        r_avg_pinch, r_pinches = 0.0, np.zeros(0)
        r_avg_jerk, r_jerks = 0.0, np.zeros(0)
        
        # คำนวณหาไบโอมาร์กเกอร์ทางกายภาพของมือซ้าย
        if len(df_left) >= 4:
            l_wrist = df_left[["wrist_x", "wrist_y"]].values
            l_thumb = df_left[["thumb_x", "thumb_y"]].values
            l_index = df_left[["index_x", "index_y"]].values
            
            l_avg_speed, l_speeds = metrics.calculate_speed(l_wrist, fps)
            l_avg_pinch, l_pinches = metrics.calculate_accuracy(l_thumb, l_index)
            l_avg_jerk, l_jerks = metrics.calculate_jerk(l_wrist, fps)
            
            l_speed_card.markdown(f'<div class="metric-card"><div class="metric-label">⚡ ความเร็วข้อมือเฉลี่ย</div><div class="metric-value">{l_avg_speed:.1f} px/s</div></div>', unsafe_allow_html=True)
            l_accuracy_card.markdown(f'<div class="metric-card"><div class="metric-label">🎯 ระยะหนีบนิ้วเฉลี่ย</div><div class="metric-value">{l_avg_pinch:.1f} px</div></div>', unsafe_allow_html=True)
            l_jerk_card.markdown(f'<div class="metric-card"><div class="metric-label">📈 ความกระตุกข้อมือเฉลี่ย</div><div class="metric-value">{l_avg_jerk:.0f} px/s³</div></div>', unsafe_allow_html=True)
        else:
            l_speed_card.warning("ไม่พบสตรีมการเคลื่อนไหวของมือซ้ายในระยะเวลาทดสอบ")
            
        # คำนวณหาไบโอมาร์กเกอร์ทางกายภาพของมือขวา
        if len(df_right) >= 4:
            r_wrist = df_right[["wrist_x", "wrist_y"]].values
            r_thumb = df_right[["thumb_x", "thumb_y"]].values
            r_index = df_right[["index_x", "index_y"]].values
            
            r_avg_speed, r_speeds = metrics.calculate_speed(r_wrist, fps)
            r_avg_pinch, r_pinches = metrics.calculate_accuracy(r_thumb, r_index)
            r_avg_jerk, r_jerks = metrics.calculate_jerk(r_wrist, fps)
            
            r_speed_card.markdown(f'<div class="metric-card"><div class="metric-label">⚡ ความเร็วข้อมือเฉลี่ย</div><div class="metric-value">{r_avg_speed:.1f} px/s</div></div>', unsafe_allow_html=True)
            r_accuracy_card.markdown(f'<div class="metric-card"><div class="metric-label">🎯 ระยะหนีบนิ้วเฉลี่ย</div><div class="metric-value">{r_avg_pinch:.1f} px</div></div>', unsafe_allow_html=True)
            r_jerk_card.markdown(f'<div class="metric-card"><div class="metric-label">📈 ความกระตุกข้อมือเฉลี่ย</div><div class="metric-value">{r_avg_jerk:.0f} px/s³</div></div>', unsafe_allow_html=True)
        else:
            r_speed_card.warning("ไม่พบสตรีมการเคลื่อนไหวของมือขวาในระยะเวลาทดสอบ")

        # จัดแนวและรวมเส้นข้อมูลพล็อตเวลา (Time Alignment)
        max_frame = int(df["frame"].max()) if not df.empty else (total_frames - 1)
        time_axis = np.arange(max_frame + 1) / fps
        
        speed_series_r = np.full(max_frame + 1, np.nan)
        speed_series_l = np.full(max_frame + 1, np.nan)
        
        pinch_series_r = np.full(max_frame + 1, np.nan)
        pinch_series_l = np.full(max_frame + 1, np.nan)
        
        jerk_series_r = np.full(max_frame + 1, np.nan)
        jerk_series_l = np.full(max_frame + 1, np.nan)
        
        # แมปค่าพิกัดให้สอดคล้องกับเฟรมเวลาจริงของวิดีโอ
        if len(df_right) >= 4:
            f_indices_r = df_right["frame"].values.astype(int)
            speed_series_r[f_indices_r] = r_speeds
            pinch_series_r[f_indices_r] = r_pinches
            jerk_series_r[f_indices_r] = r_jerks
            
        if len(df_left) >= 4:
            f_indices_l = df_left["frame"].values.astype(int)
            speed_series_l[f_indices_l] = l_speeds
            pinch_series_l[f_indices_l] = l_pinches
            jerk_series_l[f_indices_l] = l_jerks

        # 1. แสดงกราฟความเร็วในการเคลื่อนไหว
        speed_chart_header.markdown("#### ความเร็วในการขยับมือ (พิกเซล/วินาที)")
        speed_df = pd.DataFrame({
            "เวลา (วินาที)": time_axis,
            "ความเร็วมือขวา": speed_series_r,
            "ความเร็วมือซ้าย": speed_series_l
        }).set_index("เวลา (วินาที)")
        speed_chart.line_chart(speed_df)
        
        # 2. แสดงกราฟระยะห่างปลายนิ้วในการหนีบมือ
        pinch_chart_header.markdown("#### ระยะห่างการหนีบนิ้ว (พิกเซล)")
        pinch_df = pd.DataFrame({
            "เวลา (วินาที)": time_axis,
            "ระยะหนีบนิ้วมือขวา": pinch_series_r,
            "ระยะหนีบนิ้วมือซ้าย": pinch_series_l
        }).set_index("เวลา (วินาที)")
        pinch_chart.line_chart(pinch_df)
        
        # 3. แสดงกราฟวิเคราะห์ดัชนีความกระตุก
        jerk_chart_header.markdown("#### ดัชนีความกระตุกข้อมือ (พิกเซล/วินาที³)")
        jerk_df = pd.DataFrame({
            "เวลา (วินาที)": time_axis,
            "ความกระตุกมือขวา": jerk_series_r,
            "ความกระตุกมือซ้าย": jerk_series_l
        }).set_index("เวลา (วินาที)")
        jerk_chart.line_chart(jerk_df)

        # บล็อกตรรกะตัดสินใจทางคลินิก (Threshold Logic Block)
        # ตรวจสอบว่ามือข้างใดข้างหนึ่งความกระตุกเกินเกณฑ์และความเร็วต่ำหรือไม่
        lnu_suspected_r = (r_avg_jerk > jerk_threshold) and (r_avg_speed < speed_threshold) and (r_avg_speed > 0)
        lnu_suspected_l = (l_avg_jerk > jerk_threshold) and (l_avg_speed < speed_threshold) and (l_avg_speed > 0)
        
        if lnu_suspected_l or lnu_suspected_r:
            affected_side = []
            if lnu_suspected_l:
                affected_side.append(f"มือซ้าย (ความเร็ว: {l_avg_speed:.1f} px/s, ความกระตุก: {l_avg_jerk:.0f} px/s³)")
            if lnu_suspected_r:
                affected_side.append(f"มือขวา (ความเร็ว: {r_avg_speed:.1f} px/s, ความกระตุก: {r_avg_jerk:.0f} px/s³)")
                
            affected_side_str = " และ ".join(affected_side)
            diagnostics_placeholder.markdown(f"""
                <div style="background-color: #ef4444; border: 1px solid #b91c1c; padding: 15px; border-radius: 8px; color: white;">
                    <h4 style="margin: 0 0 10px 0; color: white;">⚠️ ตรวจพบแนวโน้มภาวะละเลยการใช้งานสูง (Learned Non-Use Suspected)</h4>
                    <p style="margin: 0; font-size: 14px;"><strong>ส่วนที่ทำงานบกพร่อง:</strong> {affected_side_str}</p>
                    <p style="margin: 5px 0 0 0; font-size: 13px;">ผู้ป่วยแสดงดัชนีความเร็วเฉลี่ยในการเคลื่อนไหวที่ต่ำมาก ร่วมกับดัชนีความกระตุกสั่นเทาขณะขยับนิ้วสูงผิดปกติ ซึ่งเป็นสัญญาณบ่งบอกทางคลินิกของภาวะละเลยการเคลื่อนไหวหรืออาการสั่นเกร็งในสมองสั่งการ</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            diagnostics_placeholder.markdown("""
                <div style="background-color: #10b981; border: 1px solid #047857; padding: 15px; border-radius: 8px; color: white;">
                    <h4 style="margin: 0 0 5px 0; color: white;">✅ การเคลื่อนไหวของมือสองข้างปกติ (Natural dominant movement)</h4>
                    <p style="margin: 0; font-size: 14px;">การเคลื่อนไหวของมือทั้งสองข้างอยู่ในเกณฑ์เกณฑ์ความเร็วและความสมูทที่สมดุลทางสรีรวิทยา ไม่พบข้อบ่งชี้ความเสี่ยงของภาวะละเลยการเคลื่อนไหว</p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"การวิเคราะห์วิดีโอและชีวภาพล้มเหลว: {e}")
        st.sidebar.error("การวิเคราะห์ล้มเหลว")
