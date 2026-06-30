"""
录制器 —— 无头抓帧，游戏全程保持焦点不会暂停。
用法:
  python record.py
  回答区域提问后，切到 DOTA2 开始玩；它会自动把画面存到 frames/。
  停止: 按 F8（全局热键，不用切窗口）或在终端 Ctrl+C
"""
import os, time
import numpy as np, cv2, mss
from pynput.keyboard import Key, Listener
from window_util import get_region

OUT = "frames"
FPS = 8                      # 每秒存几张（球飞得快可调大到 12）
os.makedirs(OUT, exist_ok=True)

region = get_region()
_running = True

def on_press(key):
    global _running
    if key == Key.f8:
        _running = False
        return False
Listener(on_press=on_press).start()

# 接着已有图片编号往后存
exist = [f for f in os.listdir(OUT) if f.endswith(".jpg")]
idx = (max([int(f[6:11]) for f in exist], default=-1) + 1) if exist else 0

print("\n[!] 3 秒后开始录制，切到 DOTA2 开始玩。F8 停止。")
for i in (3, 2, 1):
    print(f"  {i}...", end=" ", flush=True); time.sleep(1)
print("\n[录制中] F8 停止\n")

sct = mss.mss()
interval = 1.0 / FPS
n = 0
try:
    while _running:
        t0 = time.time()
        frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
        cv2.imwrite(f"{OUT}/frame_{idx:05d}.jpg", frame)
        idx += 1; n += 1
        if n % FPS == 0:
            print(f"\r已存 {n} 张", end="", flush=True)
        dt = time.time() - t0
        if dt < interval:
            time.sleep(interval - dt)
except KeyboardInterrupt:
    pass
print(f"\n[完成] 本次 {n} 张，目录 {OUT}/ 共 {idx} 张。下一步: python label.py")
