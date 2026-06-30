"""
离线标记器 —— 对着 frames/ 里录好的图片框 ball / paddle，导出 YOLO 标签。
游戏可以已经关掉，不影响（这就是和实时标记的区别）。

操作:
  鼠标拖拽    = 画一个框
  0 / 1       = 切换类别 (0=ball, 1=paddle)
  s 或 d / →  = 保存当前图并跳下一张
  a / ←       = 上一张
  r           = 撤销最后一个框
  x           = 跳过这张（不保存，常用于没有球的帧）
  q           = 退出
"""
import os, glob, shutil
import cv2

SRC   = "frames"
IMG_O = "dataset/images"
LBL_O = "dataset/labels"
os.makedirs(IMG_O, exist_ok=True)
os.makedirs(LBL_O, exist_ok=True)

CLASSES = ["ball", "paddle"]
COLORS  = [(0, 255, 0), (0, 0, 255)]

files = sorted(glob.glob(f"{SRC}/*.jpg"))
if not files:
    print(f"{SRC}/ 里没有图片，先运行 python record.py")
    raise SystemExit

i = 0
cur_cls = 0
boxes = []                 # 显示坐标系下的 (cls,x1,y1,x2,y2)
drawing = False
ix = iy = 0
disp = None
scale = 1.0
orig = None

WIN = "Label (拖框, 0/1类别, s保存下一张, x跳过, q退出)"
cv2.namedWindow(WIN)


def load(idx):
    global orig, disp, scale, boxes
    orig = cv2.imread(files[idx])
    h, w = orig.shape[:2]
    scale = min(1.0, 760 / h)
    disp = cv2.resize(orig, (int(w * scale), int(h * scale)))
    boxes = []
    redraw()


def redraw():
    vis = disp.copy()
    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[cls], 2)
        cv2.putText(vis, CLASSES[cls], (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[cls], 1)
    name = os.path.basename(files[i])
    cv2.putText(vis, f"[{i+1}/{len(files)}] {name}  类别={CLASSES[cur_cls]}({cur_cls})  框={len(boxes)}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow(WIN, vis)


def on_mouse(event, x, y, flags, _):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        vis = disp.copy()
        for cls, x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[cls], 2)
        cv2.rectangle(vis, (ix, iy), (x, y), COLORS[cur_cls], 1)
        cv2.imshow(WIN, vis)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - ix) > 3 and abs(y - iy) > 3:
            boxes.append((cur_cls, min(ix, x), min(iy, y), max(ix, x), max(iy, y)))
        redraw()


cv2.setMouseCallback(WIN, on_mouse)


def save_current():
    name = os.path.splitext(os.path.basename(files[i]))[0]
    H, W = orig.shape[:2]
    shutil.copy(files[i], f"{IMG_O}/{name}.jpg")
    with open(f"{LBL_O}/{name}.txt", "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            rx1, ry1, rx2, ry2 = x1/scale, y1/scale, x2/scale, y2/scale
            cx = (rx1 + rx2) / 2 / W
            cy = (ry1 + ry2) / 2 / H
            bw = (rx2 - rx1) / W
            bh = (ry2 - ry1) / H
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[保存] {name}  {len(boxes)} 个框")


load(0)
print("开始标记。建议：球(0)框小一点贴住球，推车(1)框住整辆车。")

while True:
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('0'):
        cur_cls = 0; redraw()
    elif k == ord('1'):
        cur_cls = 1; redraw()
    elif k == ord('r'):
        if boxes:
            boxes.pop(); redraw()
    elif k in (ord('s'), ord('d'), 83):        # 保存并下一张
        save_current()
        if i < len(files) - 1:
            i += 1; load(i)
        else:
            print("已是最后一张");
    elif k == ord('x'):                         # 跳过不保存
        if i < len(files) - 1:
            i += 1; load(i)
    elif k in (ord('a'), 81):                   # 上一张
        if i > 0:
            i -= 1; load(i)

cv2.destroyAllWindows()
n = len(glob.glob(f"{LBL_O}/*.txt"))
print(f"\n已标记 {n} 张到 dataset/。够 ~150 张后运行: python train_yolo.py")
