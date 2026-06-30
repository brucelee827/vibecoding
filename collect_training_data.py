"""
YOLO 训练数据采集 —— 截取游戏窗口，鼠标框选标注
用法: python collect_training_data.py
  鼠标左键拖拽框选 → s 保存 → r 撤销上一个框 → q 退出
"""

import os
import mss
import cv2
import numpy as np
from window_util import get_region

SAVE_DIR  = "dataset/images"
LABEL_DIR = "dataset/labels"
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

CLASSES = ["ball", "paddle", "brick_normal", "brick_bonus"]
COLORS  = [(0,255,0), (0,0,255), (255,165,0), (255,0,255)]

# ─────────────────────────────────────────────────────────────────────────────
region   = get_region()
W, H     = region["width"], region["height"]
DISP_W, DISP_H = min(W, 600), min(H, 750)   # 显示缩放尺寸
SCALE_X  = DISP_W / W
SCALE_Y  = DISP_H / H

drawing     = False
ix, iy      = -1, -1
boxes       = []   # [(cls, x1,y1,x2,y2)] 在显示坐标系
current_cls = 0
frame_disp  = None   # 当前显示帧（带已画的框）
raw_frame   = None   # 原始帧（保存用）
frame_idx   = 0

sct = mss.mss()


def grab():
    global raw_frame, frame_disp
    shot      = sct.grab(region)
    raw       = np.array(shot)
    raw_frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    frame_disp = cv2.resize(raw_frame, (DISP_W, DISP_H))
    # 重绘已有框
    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame_disp, (x1,y1), (x2,y2), COLORS[cls], 1)
        cv2.putText(frame_disp, CLASSES[cls], (x1, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[cls], 1)


def mouse_cb(event, x, y, flags, _):
    global drawing, ix, iy, frame_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = frame_disp.copy()
        cv2.rectangle(tmp, (ix,iy), (x,y), COLORS[current_cls], 1)
        cv2.imshow(WIN, tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x-ix) > 4 and abs(y-iy) > 4:
            boxes.append((current_cls, min(ix,x), min(iy,y), max(ix,x), max(iy,y)))
        grab()          # 重绘带框的帧
        show_info()


def save_frame():
    global frame_idx
    if raw_frame is None:
        return
    fname = f"frame_{frame_idx:05d}"
    orig_h, orig_w = raw_frame.shape[:2]
    cv2.imwrite(f"{SAVE_DIR}/{fname}.jpg", raw_frame)
    with open(f"{LABEL_DIR}/{fname}.txt", "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            # 把显示坐标转回原始坐标再归一化
            rx1, rx2 = x1/SCALE_X, x2/SCALE_X
            ry1, ry2 = y1/SCALE_Y, y2/SCALE_Y
            cx = ((rx1+rx2)/2) / orig_w
            cy = ((ry1+ry2)/2) / orig_h
            bw = (rx2-rx1) / orig_w
            bh = (ry2-ry1) / orig_h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[保存] {fname}.jpg  ({len(boxes)} 个标注)")
    frame_idx += 1
    boxes.clear()


def show_info():
    if frame_disp is None:
        return
    vis = frame_disp.copy()
    info = (f"类别: {CLASSES[current_cls]}({current_cls})  "
            f"框数: {len(boxes)}  帧: {frame_idx}  "
            f"[0-3]切换类别  s保存  r撤销  空格刷新  q退出")
    cv2.putText(vis, info, (4, DISP_H-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,0), 1)
    cv2.imshow(WIN, vis)


WIN = "Collect (框选后按 s 保存)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, DISP_W, DISP_H)
cv2.setMouseCallback(WIN, mouse_cb)

print("操作说明:")
print("  0/1/2/3 = 切换类别 (ball/paddle/brick_normal/brick_bonus)")
print("  s       = 保存当前帧和标注")
print("  r       = 撤销上一个框")
print("  空格    = 刷新截图（不保存）")
print("  q       = 退出")
print(f"区域: {region}  显示: {DISP_W}x{DISP_H}\n")

grab()
show_info()

while True:
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
        current_cls = key - ord('0')
        print(f"切换到: {CLASSES[current_cls]}")
        show_info()
    elif key == ord('s'):
        save_frame()
        grab(); show_info()
    elif key == ord('r'):
        if boxes:
            removed = boxes.pop()
            print(f"撤销: {CLASSES[removed[0]]} {removed[1:]}")
            grab(); show_info()
    elif key == ord(' '):
        grab(); show_info()

cv2.destroyAllWindows()
print(f"共保存 {frame_idx} 帧，运行 train_yolo.py 开始训练")
