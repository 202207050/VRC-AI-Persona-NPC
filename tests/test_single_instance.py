"""A second copy of the app must not share the port (it made every reply come twice)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import persona_npc_app as app
import persona_store

store = persona_store.Store(root_dir=None) if "root_dir" in persona_store.Store.__init__.__code__.co_varnames else None
first = app.make_server(0, store, None, token="t")
port = first.server_address[1]
try:
    app.make_server(port, store, None, token="t")
    raise SystemExit("FAIL: second server bound the same port")
except OSError:
    pass
finally:
    first.server_close()
again = app.make_server(port, store, None, token="t")     # free again after the first one closed
again.server_close()
print("OK single instance")

assert "API 키가 틀렸습니다" in app.ai_error_hint("AuthenticationError: Error code: 401 - invalid x-api-key")
assert "크레딧" in app.ai_error_hint("BadRequestError: Your credit balance is too low")
assert app.ai_error_hint("APITimeoutError") == ""
print("OK error hints")
