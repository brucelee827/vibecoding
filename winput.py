"""
Windows 底层输入 + 窗口聚焦（纯 ctypes，无需 pywin32）。
- 用硬件扫描码 SendInput：游戏（DOTA2/Source2）认这种，pynput 的虚拟键码常被忽略。
- focus_dota()：把 DOTA2 窗口强制提到最前台并取得焦点。
"""
import ctypes

user32   = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# ── SendInput 扫描码 ─────────────────────────────────────────────────────────
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP    = 0x0002
KEYEVENTF_EXTENDED = 0x0001
PUL = ctypes.POINTER(ctypes.c_ulong)

class _Kbd(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class _Mouse(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
class _Hw(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class _I(ctypes.Union):
    _fields_ = [("ki", _Kbd), ("mi", _Mouse), ("hi", _Hw)]
class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _I)]

# Set-1 扫描码（方向键是扩展键）
SCAN = {'a': 0x1E, 'd': 0x20, 'space': 0x39, 'left': 0xCB, 'right': 0xCD}

def _send(scan, up):
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if scan > 0xFF:
        flags |= KEYEVENTF_EXTENDED
    extra = ctypes.c_ulong(0)
    ki = _Kbd(0, scan & 0xFF, flags, 0, ctypes.pointer(extra))
    inp = _Input(1, _I(ki=ki))
    user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

def press(name):
    _send(SCAN[name], False)

def release(name):
    _send(SCAN[name], True)

# ── 聚焦 DOTA2 窗口 ──────────────────────────────────────────────────────────
def find_window(keyword="dota"):
    found = []
    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if keyword in buf.value.lower():
                found.append((hwnd, buf.value))
        return True
    user32.EnumWindows(EnumProc(cb), 0)
    return found

def focus_dota():
    wins = find_window("dota")
    if not wins:
        return None
    hwnd, title = wins[0]
    fg = user32.GetForegroundWindow()
    cur = kernel32.GetCurrentThreadId()
    fg_thread = user32.GetWindowThreadProcessId(fg, 0)
    user32.AttachThreadInput(fg_thread, cur, True)
    user32.ShowWindow(hwnd, 9)          # SW_RESTORE
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(fg_thread, cur, False)
    return title
