from ultralytics import YOLO
import cv2
import random
import numpy as np

# تحميل نموذج YOLOv8 المدرب مسبقًا
model = YOLO("yolov8n-pose.pt")  # يمكن استخدام yolov8m أو yolov8l لتحسين الدقة

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# ضبط دقة الكاميرا (الجودة) إلى دقة أعلى
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # عرض 1920 بكسل
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # ارتفاع 1080 بكسل

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# قائمة ألوان دافئة (BGR)
warm_colors = [
    (0, 0, 255),      # أحمر
    (0, 165, 255),    # برتقالي
    (0, 255, 255),    # أصفر
    (10, 100, 255),   # درجة أخرى من البرتقالي
    (20, 200, 255)    # درجة فاتحة من البرتقالي
]

# تهيئة قائمة لقطرات الحمم (النجوم)
num_drops = 300  # تقليل العدد قليلاً لتحسين الأداء
lava_drops = []
for i in range(num_drops):
    drop = {
        'x': random.randint(0, frame_width),
        'y': random.randint(0, frame_height),
        'speed': random.randint(1, 5),  # سرعة سقوط أبطأ
        'color': random.choice(warm_colors),
        'size': random.randint(10, 20)  # حجم النجمة
    }
    lava_drops.append(drop)

# إنشاء نافذة العرض بوضع ملء الشاشة
cv2.namedWindow("YOLO Hand Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def draw_star(img, center, size, color):
    """ ترسم نجمة خماسية الرؤوس عند الموقع المحدد """
    angles = np.linspace(0, 2 * np.pi, 11)  # حساب الزوايا لـ 5 رؤوس مع خطوط بينها
    points = [
        (int(center[0] + size * np.cos(angle)), int(center[1] + size * np.sin(angle)))
        for angle in angles
    ]
    for i in range(5):  # رسم النجمة بربط النقاط
        cv2.line(img, points[i * 2], points[(i * 2 + 4) % 10], color, 2, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # تشغيل YOLO على الإطار الحالي للحصول على تتبع الوضعيات
    results = model(frame)

    # نسخ الإطار الأصلي للعرض (دون رسم الهيكل أو الصناديق)
    annotated_frame = frame.copy()

    # استخراج صناديق الكشف عن الإنسان (إذا كانت متوفرة)
    boxes = []
    if hasattr(results[0], "boxes") and results[0].boxes is not None:
        boxes_array = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, "cpu") else results[0].boxes.xyxy
        boxes = boxes_array.tolist()

    # تحديث ورسم النجوم
    for drop in lava_drops:
        # تحديث موقع النجمة (السقوط من الأعلى إلى الأسفل)
        drop['y'] += drop['speed']
        if drop['y'] > frame_height:
            drop['y'] = 0
            drop['x'] = random.randint(0, frame_width)

        # التحقق مما إذا كانت النجمة داخل أي من صناديق الكشف عن الإنسان
        inside_human = False
        for b in boxes:
            if b[0] <= drop['x'] <= b[2] and b[1] <= drop['y'] <= b[3]:
                inside_human = True
                break

        # رسم النجمة إذا لم تكن فوق الإنسان
        if not inside_human:
            draw_star(annotated_frame, (drop['x'], drop['y']), drop['size'], drop['color'])

    # عرض الإطار المُعدل بوضع ملء الشاشة
    cv2.imshow("YOLO Hand Detection", annotated_frame)

    # الضغط على "q" للخروج
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
