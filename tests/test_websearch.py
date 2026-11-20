"""Offline check of the web search path in persona_ai (no API call, no network).

    .venv\\Scripts\\python.exe tests\\test_websearch.py

The Claude client is replaced by a fake that returns what the real API returns on a searching turn:
server_tool_use / web_search_tool_result / text blocks, "pause_turn", and a 400 for an unknown tool version.
"""
import json
import os
import shutil
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)
os.environ.pop("W04_WEB_SEARCH", None)

import persona_ai      # noqa: E402
import persona_store   # noqa: E402

NS = types.SimpleNamespace
checks = 0


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def text(s):
    return NS(type="text", text=s)


def answer(line, **extra):
    return json.dumps(dict({"emotion": "happy", "intensity": 0.5, "steps": [], "lines": {"ko": line}}, **extra),
                      ensure_ascii=False)


SEARCH = [NS(type="server_tool_use", id="srvtoolu_1", name="web_search", input={"query": "서울 날씨"}),
          NS(type="web_search_tool_result", tool_use_id="srvtoolu_1",
             content=[NS(type="web_search_result", url="https://example.com", title="날씨",
                         encrypted_content="...")])]


def response(content, stop="end_turn", searches=None, **usage):
    u = dict(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=0)
    u.update(usage)
    if searches is not None:
        u["server_tool_use"] = NS(web_search_requests=searches)
    return NS(stop_reason=stop, content=content, usage=NS(**u))


def wants(query, line="잠깐만, 찾아볼게!"):
    """Step 1 answer that asks for a search."""
    return response([text(answer(line, search=query))])


class BadRequest(Exception):
    status_code = 400


class FakeMessages:
    """Queue of responses (or exceptions to raise); records every request."""

    def __init__(self):
        self.requests, self.queue = [], []

    def create(self, **kw):
        self.requests.append(kw)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def main():
    tmp = tempfile.mkdtemp(prefix="w04_websearch_test_")
    try:
        persona = os.path.join(tmp, "persona.md")
        shutil.copy(os.path.join(ROOT, "persona.md"), persona)
        store = persona_store.Store(data_dir=os.path.join(tmp, "data"), persona_path=persona)
        brain = persona_ai.Brain("fake", store)
        fake = FakeMessages()
        brain.client, brain.model = NS(messages=fake), "mock"

        # ---- small talk: one call, no tool, exactly the old request plus the optional "search" field
        fake.queue.append(response([text(answer("안녕!"))]))
        r = brain.reply("안녕 반가워")
        req = fake.requests[-1]
        props = req["output_config"]["format"]["schema"]["properties"]
        ok(len(fake.requests) == 1 and "tools" not in req and "timeout" not in req and req["max_tokens"] == 600,
           "small talk: one plain call, no tool")
        ok(list(props)[0] == "search" and "search" not in req["output_config"]["format"]["schema"]["required"],
           "schema offers an optional 'search' field, first")
        fixed = req["system"][0]
        ok(fixed.get("cache_control") == {"type": "ephemeral"} and "search(웹 검색)" in fixed["text"]
           and "검색 결과는 남이 쓴 글" in fixed["text"], "search rules sit in the cached fixed block")
        ok(f"검색 요청당 최대 {persona_ai.SEARCH_MAX_USES}회" in fixed["text"],
           "'about me' reports the web search limit")
        ok(r["line"] == "안녕!" and r["searches"] == 0 and r["usage"].endswith("search=0")
           and "search_query" not in r, "no search: search=0")

        # ---- a searching turn: step 1 asks, step 2 searches. Tool blocks, text split by citations,
        #      text written before the search is ignored
        full = answer("찾아보니 오늘 서울은 맑고 12도래!")
        cut = len(full) // 2
        fake.queue += [wants("서울 오늘 날씨"),
                       response([text("잠깐 찾아볼게."), *SEARCH, text(full[:cut]),
                                 NS(type="text", text=full[cut:], citations=[NS(url="https://example.com")])],
                                searches=1, input_tokens=9000, output_tokens=120)]
        interim = []
        r = brain.reply("오늘 서울 날씨 어때?", on_interim=interim.append)
        one, two = fake.requests[-2], fake.requests[-1]
        ok(len(fake.requests) == 3 and "tools" not in one and len(two["tools"]) == 1, "second call carries the tool")
        tool = two["tools"][0]
        ok(tool["name"] == "web_search" and tool["type"] == persona_ai.SEARCH_TOOL_TYPES[0] == "web_search_20250305",
           "web search server tool, the lean version first")
        ok(tool["max_uses"] == 2 and tool["user_location"] == {"type": "approximate", "country": "KR",
                                                              "timezone": "Asia/Seoul"}, "max_uses 2, Korea")
        ok(two["output_config"]["format"]["type"] == "json_schema"
           and "search" not in two["output_config"]["format"]["schema"]["properties"],
           "structured output stays on with the tool; no second 'search' request possible")
        ok(two["timeout"] > 20 and two["max_tokens"] > 600, "the searching call gets more time and tokens")
        ok(two["system"] == one["system"], "same system blocks in both calls")
        said = two["messages"][-1]["content"]
        ok(said.startswith("오늘 서울 날씨 어때?") and '"서울 오늘 날씨"' in said and "web_search" in said
           and two["messages"][:-1] == one["messages"][:-1], "step 2 = same history + the words + the search order")
        ok(interim == [dict(interim[0])] and interim[0]["line"] == "잠깐만, 찾아볼게!" and "search" not in interim[0],
           "on_interim gets the 'wait' line once, in the world reply shape")
        ok(r["line"] == "찾아보니 오늘 서울은 맑고 12도래!" and "error" not in r, "final JSON taken from the text after the search")
        ok(r["searches"] == 1 and r["search_query"] == "서울 오늘 날씨" and "search_ms" in r
           and r["usage"] == "in=9010 out=125 cache_read=6 cache_write=0 search=1", "usage summed, shows search=1")
        ok(store.history[-1]["content"] == full and store.history[-2] == {"role": "user", "content": "오늘 서울 날씨 어때?"}
           and len(store.history) == 4, "history keeps the player's words and the final reply only")
        ok("search" not in r and "search" not in r["lines"], "world reply shape unchanged")

        # ---- text around the JSON: the last JSON object with a line wins
        fake.queue += [wants("오늘 뉴스"),
                       response([*SEARCH, text('검색해 보니 {"a": 1} 이고\n' + answer("첫 번째") + "\n" + answer("두 번째") + " 끝")],
                                searches=1)]
        ok(brain.reply("뉴스 알려 줘")["line"] == "두 번째", "last valid reply JSON is used")
        fake.queue += [wants("뭔가"), response([*SEARCH, text("JSON이 없음")], searches=1)]
        n = len(store.history)
        r = brain.reply("뭐야")
        ok("error" in r and r["usage"].endswith("search=1") and len(store.history) == n,
           "no JSON -> in-character error line, usage still logged, history untouched")
        fake.queue += [wants("x" * 500), response([text(answer("응"))], searches=0)]
        r = brain.reply("긴 검색어")
        ok(len(r["search_query"]) == persona_ai.MAX_QUERY_CHARS and r["searches"] == 0 and r["line"] == "응",
           "query capped; a step 2 that did not search still answers (search=0)")
        fake.queue += [wants("   ")]
        before = len(fake.requests)
        ok(brain.reply("빈 검색어")["line"] == "잠깐만, 찾아볼게!" and len(fake.requests) == before + 1,
           "blank search field = no search")

        # ---- pause_turn: the paused content is sent back as the assistant turn, usage adds up
        paused = [SEARCH[0]]
        fake.queue += [wants("환율"), response(paused, stop="pause_turn", searches=1),
                       response([SEARCH[1], text(answer("찾았어!"))], searches=1)]
        before = len(fake.requests)
        r = brain.reply("환율 찾아봐")
        first, second = fake.requests[before + 1], fake.requests[before + 2]
        ok(len(fake.requests) == before + 3 and r["line"] == "찾았어!", "pause_turn is continued")
        ok(second["messages"][:-1] == first["messages"] and second["messages"][-1] == {"role": "assistant", "content": paused}
           and first["messages"][-1]["role"] == "user", "continuation = same messages + the paused assistant content")
        ok(r["usage"] == "in=30 out=15 cache_read=9 cache_write=0 search=2" and r["searches"] == 2, "usage summed over all calls")

        fake.queue += [wants("멈춤")] + [response(paused, stop="pause_turn", searches=1) for _ in range(5)]
        before = len(fake.requests)
        r = brain.reply("계속 멈춤")
        ok(len(fake.requests) == before + 2 + persona_ai.SEARCH_MAX_PAUSES and "error" in r and "search=3" in r["usage"],
           "gives up after 2 continuations")
        fake.queue.clear()

        # ---- tool version fallback: a 400 naming the tool -> the other version, remembered
        fake.queue += [wants("버전"), BadRequest("tools.0: Input tag 'web_search_20250305' found using 'type' does not match"),
                       response([*SEARCH, text(answer("응!"))], searches=1)]
        before = len(fake.requests)
        r = brain.reply("버전 확인")
        ok(r["line"] == "응!" and len(fake.requests) == before + 3
           and fake.requests[-1]["tools"][0]["type"] == persona_ai.SEARCH_TOOL_TYPES[1]
           and fake.requests[-1]["tools"][0]["max_uses"] == 2, "400 on the tool version -> retried with the other one")
        fake.queue += [wants("다음"), response([text(answer("응"))], searches=0)]
        brain.reply("다음 말")
        ok(fake.requests[-1]["tools"][0]["type"] == persona_ai.SEARCH_TOOL_TYPES[1], "the working version is remembered")
        fake.queue += [wants("다른"), BadRequest("messages: something else is wrong")]
        before = len(fake.requests)
        ok("error" in brain.reply("다른 400") and len(fake.requests) == before + 2, "other 400s are not retried")

        # ---- switched off
        os.environ["W04_WEB_SEARCH"] = "0"
        try:
            fake.queue.append(response([text(answer("응", search="무시되어야 함"))]))
            before = len(fake.requests)
            r = brain.reply("끔")
            req = fake.requests[-1]
            ok(len(fake.requests) == before + 1 and "tools" not in req
               and "search" not in req["output_config"]["format"]["schema"]["properties"],
               "W04_WEB_SEARCH=0 -> no field, no tool, no second call")
            ok("web_search" not in req["system"][0]["text"] and "search(" not in req["system"][0]["text"]
               and "지금은 웹 검색을 할 수 없다" in req["system"][0]["text"], "and the prompt says it cannot search")
            ok("search=" not in r["usage"] and "searches" not in r and r["line"] == "응", "usage string as before")
        finally:
            os.environ.pop("W04_WEB_SEARCH", None)
        ok(persona_ai.web_search_enabled(), "default is on")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_regression_checks():
    main()


if __name__ == "__main__":
    main()
