"""公共：游戏窗口区域获取 / 交互式框选 / 保存到 region.json"""

import os, json
import numpy as np
import cv2
import mss

REGION_FILE = "region.json"
DEFAULT_REGION = {"left": 490, "top": 55, "width": 495, "height": 730}


def load_region():
    if os.path.exists(REGION_FILE):
        try:
            with open(REGION_FILE) as f:
                r = json.load(f)
            print(f"[区域] 读取 {REGION_FILE}: {r}")
            return r
        except Exception:
            pass
    return None


def save_region(r):
    with open(REGION_FILE, "w") as f:
        json.dump(r, f)
    print(f"[区域] 已保存到 {REGION_FILE}: {r}")


def pick_region_interactive():
    """
    全屏截图，让用户拖一个框选游戏区域。
    返回 {left, top, width, height}（绝对屏幕坐标）。
    """
    sct = mss.mss()
    mon = sct.monitors[1]              # 主显示器
    shot = np.array(sct.grab(mon))
    full = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    H, W = full.shape[:2]

    # 缩放到适合屏幕显示
    disp_w = min(W, 1280)
    scale = disp_w / W
    disp = cv2.resize(full, (disp_w, int(H * scale)))

    print("\n>>> 用鼠标拖一个框选住【游戏砖块区域】(从顶部砖墙到底部推车)，")
    print(">>> 然后按 Enter 或 Space 确认；按 c 取消。\n")
    r = cv2.selectROI("框选游戏区域 (拖拽后回车确认)", disp,
                      showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = r
    if w == 0 or h == 0:
        print("[区域] 未选择，使用默认")
        return DEFAULT_REGION.copy()

    # 缩放坐标换算回真实屏幕坐标 + 显示器偏移
    region = {
        "left":   mon["left"] + int(x / scale),
        "top":    mon["top"]  + int(y / scale),
        "width":  int(w / scale),
        "height": int(h / scale),
    }
    return region


def get_region(force_pick=False):
    """优先读 region.json；没有或 force_pick 时让用户框选。"""
    if not force_pick:
        r = load_region()
        if r:
            ans = input("回车用上次的区域，输入 r 重新框选: ").strip().lower()
            if ans != 'r':
                return r

    region = pick_region_interactive()
    save_region(region)
    return region
