"""
HSV 实时调参器 —— 拖滑块直到只框住目标，按 s 保存到 hsv.json
  Tab / t 键：在 球(ball) / 推车(paddle) 之间切换
  s：保存当前目标的参数
  q：退出
"""
import json, os, sys, glob
import numpy as np, cv2, mss
from window_util import get_region

HSV_FILE = "hsv.json"
DEFAULT = {
    "ball":   {"h0":5,"h1":35,"s0":90,"s1":255,"v0":170,"v1":255,"amin":6,"amax":160},
    "paddle": {"h0":0,"h1":15,"s0":150,"s1":255,"v0":80,"v1":255,"amin":50,"amax":99999},
}

def load():
    if os.path.exists(HSV_FILE):
        try:
            return {**DEFAULT, **json.load(open(HSV_FILE))}
        except Exception:
            pass
    return {k: dict(v) for k, v in DEFAULT.items()}

cfg = load()

# 模式：有命令行图片参数 → 对着静态图片调（推荐，游戏不会暂停干扰）
#       否则 → 实时抓屏
STATIC = None
if len(sys.argv) > 1:
    STATIC = cv2.imread(sys.argv[1])
elif glob.glob("frames/*.jpg"):
    # 默认拿最近录的一张当样本，省得手动指定
    latest = sorted(glob.glob("frames/*.jpg"))[-1]
    print(f"[模式] 用录好的图片调参: {latest}（要实时调就删掉 frames/ 或传别的路径）")
    STATIC = cv2.imread(latest)

if STATIC is None:
    region = get_region()
    sct = mss.mss()

def grab_frame():
    if STATIC is not None:
        return STATIC.copy()
    return cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)

WIN = "Tune (Tab切换目标, s保存, q退出)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
target = "ball"

def make_bars():
    cv2.destroyWindow(WIN)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    c = cfg[target]
    cv2.createTrackbar("Hmin", WIN, c["h0"], 179, lambda v: None)
    cv2.createTrackbar("Hmax", WIN, c["h1"], 179, lambda v: None)
    cv2.createTrackbar("Smin", WIN, c["s0"], 255, lambda v: None)
    cv2.createTrackbar("Smax", WIN, c["s1"], 255, lambda v: None)
    cv2.createTrackbar("Vmin", WIN, c["v0"], 255, lambda v: None)
    cv2.createTrackbar("Vmax", WIN, c["v1"], 255, lambda v: None)
    cv2.createTrackbar("AreaMin", WIN, c["amin"], 500, lambda v: None)
    cv2.createTrackbar("AreaMax", WIN, min(c["amax"],5000), 5000, lambda v: None)

make_bars()
print(f"当前目标: {target}  （Tab 切换，s 保存，q 退出）")

while True:
    frame = grab_frame()
    H, W = frame.shape[:2]

    h0 = cv2.getTrackbarPos("Hmin", WIN); h1 = cv2.getTrackbarPos("Hmax", WIN)
    s0 = cv2.getTrackbarPos("Smin", WIN); s1 = cv2.getTrackbarPos("Smax", WIN)
    v0 = cv2.getTrackbarPos("Vmin", WIN); v1 = cv2.getTrackbarPos("Vmax", WIN)
    amin = cv2.getTrackbarPos("AreaMin", WIN); amax = cv2.getTrackbarPos("AreaMax", WIN)
    cfg[target].update(h0=h0,h1=h1,s0=s0,s1=s1,v0=v0,v1=v1,amin=amin,amax=max(amax,amin+1))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h0,s0,v0]), np.array([h1,s1,v1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # 在原图上画出通过 面积+圆度 过滤的候选
    out = frame.copy()
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_pass = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if not (amin < a < max(amax, amin+1)):
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
        cnt_pass += 1

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combo = np.hstack([out, mask_bgr])
    s = min(1.0, 1000/combo.shape[1])
    combo = cv2.resize(combo, (int(combo.shape[1]*s), int(combo.shape[0]*s)))
    cv2.putText(combo, f"目标={target}  通过候选数={cnt_pass}  (左:原图框 右:掩膜)",
                (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.imshow(WIN, combo)

    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        break
    elif k in (9, ord('t')):                 # Tab 或 t
        target = "paddle" if target == "ball" else "ball"
        make_bars()
        print(f"切换到: {target}")
    elif k == ord('s'):
        json.dump(cfg, open(HSV_FILE, "w"), indent=2)
        print(f"[保存] {HSV_FILE}  ({target}={cfg[target]})")

cv2.destroyAllWindows()
print(f"参数已写入 {HSV_FILE}，现在 test_detection.py / play_boot.py 会自动读取")
