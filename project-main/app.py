import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# بارگذاری مدل آموزش دیده
model = YOLO("flower_detector/yolov8_flower_model/weights/best.pt")

st.title("تشخیص گل‌ها با YOLOv8 🌸")
st.write("تصویر گل را آپلود کن تا نوع و تعداد آن مشخص شود.")

uploaded_file = st.file_uploader("تصویر را آپلود کنید", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # نمایش تصویر آپلود شده
    image = Image.open(uploaded_file)
    st.image(image, caption="تصویر ورودی", use_column_width=True)

    # ذخیره موقت برای پردازش
    temp_path = "temp.jpg"
    image.save(temp_path)

    # اجرای مدل
    results = model(temp_path, save=True, conf=0.3)

    # گرفتن مسیر تصویر با نتیجه
    result_image_path = os.path.join(results[0].save_dir, os.path.basename(temp_path))
    st.image(result_image_path, caption="نتیجه تشخیص", use_column_width=True)

    # نمایش تعداد گل‌ها
    st.success(f"✅ تعداد گل‌های شناسایی‌شده: {len(results[0].boxes)}")

    # نمایش نوع گل‌ها
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    st.write("نوع گل‌ها:", detected_classes)
