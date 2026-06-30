"""
破牢之靴 AI —— 无头模式（不弹 OpenCV 窗口，避免抢占游戏焦点）
依赖: pip install mss pynput opencv-python numpy

关键: 运行后必须让 DOTA2 处于最前台。按键才能进游戏。
退出: 按 F8（全局热键）或 Ctrl+C
"""

import time, threading, sys, os, json
import numpy as np
import cv2
import mss
from pynput.keyboard import Key, Listener
from window_util import get_region
import winput   # 扫描码输入 + 窗口聚焦

# ── 颜色检测（优先读 tune.py 生成的 hsv.json）────────────────────────────────
_DEF = {
    "ball":   {"h0":5,"h1":35,"s0":90,"s1":255,"v0":170,"v1":255,"amin":6,"amax":160},
    "paddle": {"h0":0,"h1":15,"s0":150,"s1":255,"v0":80,"v1":255,"amin":50,"amax":99999},
}
def _load_hsv():
    if os.path.exists("hsv.json"):
        try:
            d = json.load(open("hsv.json"))
            print("[HSV] 已读取 hsv.json")
            return {**_DEF, **d}
        except Exception:
            pass
    print("[HSV] 用默认值（建议先跑 tune.py 调参）")
    return {k: dict(v) for k, v in _DEF.items()}

_H = _load_hsv()
_B, _P = _H["ball"], _H["paddle"]
BALL_HSV_LOW   = np.array([_B["h0"], _B["s0"], _B["v0"]])
BALL_HSV_HIGH  = np.array([_B["h1"], _B["s1"], _B["v1"]])
BALL_AMIN, BALL_AMAX = _B["amin"], _B["amax"]
PADDLE_HSV_LOW  = np.array([_P["h0"], _P["s0"], _P["v0"]])
PADDLE_HSV_HIGH = np.array([_P["h1"], _P["s1"], _P["v1"]])

# ── 控制参数（单死区开关控制）──────────────────────────────────────────────
DEAD_ZONE     = 6     # 推车中心与落点差小于此值就停（小=居中但易抖，大=不抖但偏）
LEAD_FRAMES   = 2.0   # 提前量：按推车当前速度，提前几帧到位，补偿移动延迟
LAUNCH_DELAY  = 4.0

# ── YOLO ─────────────────────────────────────────────────────────────────────
USE_YOLO   = True
YOLO_PATH  = "runs/detect/runs/boot_game/weights/best.pt"
YOLO_CONF  = 0.35
YOLO_IMGSZ = None     # None=自动(GPU用640,CPU用416)；想更快可手动设 320
_yolo = None
_device = 'cpu'
_half = False

def load_yolo():
    global _yolo, _device, _half, YOLO_IMGSZ
    from ultralytics import YOLO
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    _device = 0 if has_cuda else 'cpu'
    if YOLO_IMGSZ is None:
        YOLO_IMGSZ = 640 if has_cuda else 480  # CPU 适度降分辨率，兼顾小球检测
    _yolo = YOLO(YOLO_PATH)
    _yolo.predict(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), np.uint8),   # 预热
                  imgsz=YOLO_IMGSZ, device=_device, verbose=False)
    dev = 'GPU' if has_cuda else 'CPU'
    print(f"[YOLO] 已加载 {YOLO_PATH}  设备={dev}  imgsz={YOLO_IMGSZ}  类别={_yolo.names}")

def detect_yolo(frame, paddle_top_y):
    """用 YOLO 出 (ball, paddle_x, paddle_y)。ball=(cx,cy) 或 None。"""
    res = _yolo.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                        device=_device, verbose=False)[0]
    ball = None       # 取最低（最接近推车）的球
    paddle_x = None
    paddle_y = None   # 推车上表面 y（真实接球高度）
    paddle_best = -1.0
    for b in res.boxes:
        cls = int(b.cls)
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        conf = float(b.conf)
        name = _yolo.names[cls]
        if name == "ball":
            if cy <= paddle_top_y and (ball is None or cy > ball[1]):
                ball = (cx, cy)
        elif name == "paddle":
            if conf > paddle_best:        # 取置信度最高的车
                paddle_best = conf
                paddle_x = cx             # 推车中心 x
                paddle_y = y1             # 推车顶面 y
    return ball, (int(paddle_x) if paddle_x is not None else None), paddle_y
# ─────────────────────────────────────────────────────────────────────────────

_held = {'a': False, 'd': False}
_running = True


def hold(key):
    other = 'a' if key == 'd' else 'd'
    if _held[other]:
        winput.release(other); _held[other] = False
    if not _held[key]:
        winput.press(key); _held[key] = True


def release_all():
    for k in ('a', 'd'):
        if _held[k]:
            winput.release(k); _held[k] = False


def tap(key, dur=0.06):
    release_all()
    winput.press(key); time.sleep(dur); winput.release(key)


def detect_ball(frame, paddle_top_y):
    """
    找球：小圆点 + 高圆度 + **孤立性**（周围没有大片同色砖块）。
    孤立性是关键 —— 球在黑暗空地里飞，砖块成片聚集。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H = frame.shape[0]
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if not (BALL_AMIN < a < BALL_AMAX):   # 球的面积范围（tune.py 调）
            continue
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        if 4*np.pi*a/(per*per) < 0.6:         # 圆度，排除长条砖
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w/2, y + h/2
        if cy > paddle_top_y:                 # 别把推车当球
            continue
        # —— 孤立性检测：取 3 倍 bbox 的窗口，统计同色像素 ——
        pad = int(max(w, h) * 1.5)
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(frame.shape[1], x+w+pad), min(H, y+h+pad)
        win = mask[y0:y1, x0:x1]
        win_area = cv2.countNonZero(win)
        if win_area > a * 3.0:                # 周围还有一堆同色 → 是砖块群，丢弃
            continue
        if best is None or cy > best[1]:      # 取最低（最接近推车）的球
            best = (cx, cy)
    return best


def detect_paddle(frame):
    """
    推车在底部。红色像素来自：车顶横幅 + 左右两个红箭头。
    箭头对称地夹着车 → 用红色像素的**质心 x** 即为车中心，
    比'取最大轮廓'稳得多（最大轮廓常落到某个箭头上）。
    """
    h, w = frame.shape[:2]
    y0 = int(h * 0.80)
    roi = frame[y0:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PADDLE_HSV_LOW, PADDLE_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    M = cv2.moments(mask)
    if M["m00"] < 50:                         # 红像素太少 → 没找到
        return None
    return int(M["m10"] / M["m00"])


def predict_x(ball, prev, W, H, catch_y=None):
    """预测球落到推车顶面(catch_y)时球心的 x，含左右墙反弹。"""
    if catch_y is None:
        catch_y = H * 0.85
    if prev is None:
        return ball[0]
    vx = ball[0] - prev[0]
    vy = ball[1] - prev[1]
    if vy <= 0.5:                      # 球在上升，先跟着当前 x 预站位
        return ball[0]
    steps = (catch_y - ball[1]) / max(vy, 0.1)
    px = ball[0] + vx * steps
    while not (0 <= px <= W):          # 模拟撞墙反弹
        if px < 0: px = -px
        if px > W: px = 2*W - px
    return px


_prev_paddle = None

def move(target_x, paddle_x):
    """单死区开关控制 + 提前量：差超过 DEAD_ZONE 就全速冲向落点。"""
    global _prev_paddle
    if paddle_x is None or target_x is None:
        release_all(); return "?"

    # 提前量：用推车速度预判，避免冲过头/停太晚
    pv = 0.0 if _prev_paddle is None else (paddle_x - _prev_paddle)
    _prev_paddle = paddle_x
    eff_paddle = paddle_x + pv * LEAD_FRAMES
    diff = target_x - eff_paddle

    if abs(diff) <= DEAD_ZONE:            # 在死区内 → 停
        release_all(); return "停"
    if diff > 0:
        hold('d'); return "→D"
    else:
        hold('a'); return "←A"


def launch_sequence():
    time.sleep(0.5)
    print("[发球] 归中 → 锁定 → 调角 → 发射")
    for _ in range(4): tap('a', 0.08); time.sleep(0.03)
    for _ in range(2): tap('d', 0.08); time.sleep(0.03)
    time.sleep(0.3)
    tap('space', 0.08); time.sleep(0.5)     # 锁定
    tap('d', 0.12);     time.sleep(0.3)     # 微调弹道
    tap('space', 0.08)                      # 发射
    print("[发球] 完成")


def on_press(key):
    global _running
    if key == Key.f8:
        print("\n[F8] 停止")
        _running = False
        return False


def main():
    global _running
    region = get_region()
    W, H = region["width"], region["height"]
    paddle_top_y = H * 0.72

    if USE_YOLO:
        load_yolo()

    Listener(on_press=on_press).start()   # 全局 F8 退出

    print(f"\n[!] {LAUNCH_DELAY:.0f}s 后自动把 DOTA2 切到前台并开始...")
    for i in range(int(LAUNCH_DELAY), 0, -1):
        print(f"    {i}...", end=' ', flush=True); time.sleep(1)
    print()

    title = winput.focus_dota()
    if title:
        print(f"[聚焦] 已切到游戏窗口: {title}")
    else:
        print("[警告] 没找到 DOTA2 窗口！请手动点一下游戏窗口。")
    time.sleep(0.5)

    threading.Thread(target=launch_sequence, daemon=True).start()

    sct = mss.mss()
    prev_ball = None
    last_log = 0
    last_focus = 0

    print("[AI] 运行中（F8 退出）\n")
    try:
        while _running:
            t0 = time.time()

            # 每 0.4s 把游戏抢回前台，防止焦点漂回终端导致按键失效/游戏暂停
            if t0 - last_focus > 0.4:
                winput.focus_dota()
                last_focus = t0

            frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)

            catch_y = None
            if USE_YOLO:
                ball, paddle_x, paddle_y = detect_yolo(frame, paddle_top_y)
                catch_y = paddle_y           # 用推车真实顶面当接球线
            else:
                ball     = detect_ball(frame, paddle_top_y)
                paddle_x = detect_paddle(frame)

            action = "-"
            if ball:
                # tx = 球落到推车顶面时的 x；让推车中心(paddle_x)对准它 → 中央接球
                tx = predict_x(ball, prev_ball, W, H, catch_y)
                prev_ball = ball
                action = move(tx, paddle_x)
            else:
                release_all(); prev_ball = None

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0

            # 每 0.3s 打一行状态（含 FPS，FPS 太低=反应慢的根因）
            if time.time() - last_log > 0.3:
                bstr = f"({ball[0]:.0f},{ball[1]:.0f})" if ball else "None"
                pstr = f"{paddle_x}" if paddle_x is not None else "None"
                print(f"\r球={bstr:14s} 推车x={pstr:5s} 动作={action:4s} FPS={fps:4.1f}",
                      end='', flush=True)
                last_log = time.time()

            # 不再 sleep —— 让控制环尽量快地跟上 YOLO 出帧
    except KeyboardInterrupt:
        pass
    finally:
        release_all()
        print("\n[AI] 已停止")


if __name__ == "__main__":
    main()
