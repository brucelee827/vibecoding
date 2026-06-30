"""
YOLO 实时检测自检（只看不控制）。
显示截取画面 + YOLO 框，确认 region 对不对、球/推车认不认得到。
按 q 退出。
"""
import time
import numpy as np, cv2, mss
from window_util import get_region
import play_boot as pb

region = get_region()
W, H = region["width"], region["height"]
pb.load_yolo()
sct = mss.mss()
print("绿框=ball 红框=paddle。若整个画面不是游戏 → region 错。按 q 退出")

while True:
    t0 = time.time()
    frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
    res = pb._yolo.predict(frame, conf=pb.YOLO_CONF, imgsz=pb.YOLO_IMGSZ,
                           device=pb._device, verbose=False)[0]
    n = 0
    for b in res.boxes:
        cls = int(b.cls); conf = float(b.conf)
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        name = pb._yolo.names[cls]
        color = (0, 255, 0) if name == "ball" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        n += 1

    fps = 1.0 / max(time.time() - t0, 1e-3)
    s = min(1.0, 760 / H)
    vis = cv2.resize(frame, (int(W * s), int(H * s)))
    cv2.putText(vis, f"detections={n}  FPS={fps:.1f}  imgsz={pb.YOLO_IMGSZ}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow("YOLO Test (q=quit)", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
