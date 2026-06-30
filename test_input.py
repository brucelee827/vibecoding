"""
输入隔离测试：判断 SendInput 到底能不能控制别的程序。
用法：
  1. 打开记事本(notepad)，光标点进去
  2. 运行 python test_input.py
  3. 3 秒内别动，看记事本里有没有出现 'aaaaa' 和空格

结果判读：
  记事本出现 aaaaa  →  SendInput 正常。那么 DOTA2 不动 = 它以管理员运行，
                       解决：用【管理员身份】打开终端再跑 play_boot.py
  记事本没反应      →  SendInput 被系统挡了(极少见)，再排查
"""
import time, winput

print("3 秒后向当前焦点窗口连发 5 个 'a' 和 1 个空格，请把光标放到记事本里...")
for i in (3, 2, 1):
    print(f"  {i}...", end=" ", flush=True); time.sleep(1)
print("\n发送中...")

for _ in range(5):
    winput.press('a'); time.sleep(0.05); winput.release('a'); time.sleep(0.1)
winput.press('space'); time.sleep(0.05); winput.release('space')

print("完成。检查记事本是否出现 'aaaaa '。")
