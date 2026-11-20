"""Receive VRChat's local OSC microphone state without an extra dependency."""
import socket
import threading
import time

ADDRESS = b"/avatar/parameters/MuteSelf"
PREFIX = ADDRESS + b"\0" * (4 - len(ADDRESS) % 4)
HELP = "OSC 음소거 신호가 없습니다. VRChat Action Menu > Options > OSC > Enabled를 확인하세요. 신호가 오기 전에는 기존 방식으로 듣습니다."


def parse_mute(packet):
    if packet == PREFIX + b",T\0\0":
        return True
    if packet == PREFIX + b",F\0\0":
        return False
    return None


class MuteState:
    def __init__(self):
        self.snapshot = (False, 0)
        self.seen = False

    def receive(self, packet):
        muted = parse_mute(packet)
        if muted is not None:
            before, epoch = self.snapshot
            self.snapshot = (muted, epoch + (muted != before))
            self.seen = True

    def run(self, log, stop, port=9001):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(("127.0.0.1", port))
            except OSError:
                log("OSC_MUTE", f"port={port} bind_failed", say=HELP + f" UDP {port} 사용 여부도 확인하세요.")
                return
            sock.settimeout(0.25)
            deadline = time.monotonic() + 5
            while not stop.is_set():
                try:
                    packet, _ = sock.recvfrom(1024)
                except socket.timeout:
                    pass
                except OSError:
                    if not self.seen and deadline is not None:
                        log("OSC_MUTE", "receive_failed", say=HELP)
                    return
                else:
                    self.receive(packet)
                if deadline is not None and time.monotonic() >= deadline:
                    if not self.seen:
                        log("OSC_MUTE", "no_mute_signal", say=HELP)
                    deadline = None

    def start(self, log, stop):
        threading.Thread(target=self.run, args=(log, stop), daemon=True).start()
