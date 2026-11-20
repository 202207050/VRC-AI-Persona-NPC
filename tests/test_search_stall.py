"""Searching call that answers only "찾아볼게" is retried once, then reported honestly."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import persona_ai as pa

ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print("PASS", name)
    else: fail += 1; print("FAIL", name)

check("stall line detected", pa._is_stall({"lines": {"ko": "잠깐만, 찾아볼게!"}}))
check("empty detected", pa._is_stall({"lines": {"ko": ""}}))
check("real answer not stall", not pa._is_stall({"lines": {"ko": "찾아보니 서울은 오늘 맑고 12도래. 잠깐만 나가도 쌀쌀할 거야."}}))
check("normal short answer not stall", not pa._is_stall({"lines": {"ko": "오늘 서울은 맑아!"}}))
print(f"결과: PASS {ok}, FAIL {fail}")
sys.exit(1 if fail else 0)
