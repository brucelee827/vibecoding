"""
检测自检工具：显示截取区域 + 标出检测到的球(绿)和推车(红)。
不控制游戏，用来确认 region.json 和颜色阈值是否正确。
按 q 退出。
"""
import numpy as np, cv2, mss
from window_util import get_region
import play_boot as pb

region = get_region()
W, H = region["width"], region["height"]
sct = mss.mss()
print("显示中：绿点=球  红线=推车  黄圈=预测落点。按 q 退出")

prev = None
while True:
    frame = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
    ball     = pb.detect_ball(frame, H*0.72)
    paddle_x = pb.detect_paddle(frame)

    s = min(1.0, 700/H)
    vis = cv2.resize(frame, (int(W*s), int(H*s)))
    if ball:
        cv2.circle(vis, (int(ball[0]*s), int(ball[1]*s)), 7, (0,255,0), -1)
        if prev:
            tx = pb.predict_x(ball, prev, W, H)
            cv2.circle(vis, (int(tx*s), int(H*0.85*s)), 9, (0,255,255), 2)
        prev = ball
    else:
        prev = None
    if paddle_x:
        cv2.line(vis, (int(paddle_x*s), int(H*0.80*s)),
                       (int(paddle_x*s), int(H*s)), (0,0,255), 2)

    bstr = f"({ball[0]:.0f},{ball[1]:.0f})" if ball else "None"
    cv2.putText(vis, f"ball={bstr} paddle={paddle_x}", (5,18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.imshow("Detection Test (q=quit)", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
