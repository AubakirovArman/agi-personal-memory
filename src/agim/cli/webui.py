"""Simple web dashboard for AGI Personal Memory."""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from ..core.system import AGIMSystem


AGIM_SYSTEM: AGIMSystem | None = None


def get_agim() -> AGIMSystem:
    global AGIM_SYSTEM
    if AGIM_SYSTEM is None:
        workdir = os.environ.get("AGIM_HOME", str(Path.home() / ".agim"))
        AGIM_SYSTEM = AGIMSystem(workdir=workdir)
    return AGIM_SYSTEM


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        agim = get_agim()
        if self.path == "/api/stats":
            self._json(agim.stats().__dict__)
        elif self.path == "/api/history":
            self._json(agim.log.tail(100))
        elif self.path == "/api/memories":
            memories = {**agim.retrieval._data, **agim.refusals._data}
            self._json({"count": len(memories), "entries": list(memories.keys())})
        elif self.path in ("/", "/index.html"):
            self._html(self._dashboard_html(agim))
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def _dashboard_html(self, agim):
        s = agim.stats()
        return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AGI Personal Memory</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
max-width:900px;margin:0 auto;padding:2rem;background:#0d1117;color:#c9d1d9}}
h1{{color:#58a6ff}}h2{{color:#f0883e;margin-top:2rem}}
.card{{background:#161b22;border-radius:8px;padding:1rem;margin:1rem 0}}
.stat{{display:inline-block;margin:0.5rem 1rem;text-align:center}}
.stat-num{{font-size:2rem;color:#58a6ff}}.stat-label{{font-size:0.8rem;color:#8b949e}}
table{{width:100%;border-collapse:collapse}}th{{text-align:left;color:#f0883e;padding:8px}}
td{{padding:8px;border-top:1px solid #30363d}}
a{{color:#58a6ff}}code{{background:#0d1117;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>AGI Personal Memory</h1>
<div class="card">
<div class="stat"><div class="stat-num">{s.total_facts}</div><div class="stat-label">Total Facts</div></div>
<div class="stat"><div class="stat-num">{s.total_commits}</div><div class="stat-label">Total Commits</div></div>
<div class="stat"><div class="stat-num">{s.rollback_count}</div><div class="stat-label">Rollbacks</div></div>
</div>
<div class="card">
<h2>Facts by Tier</h2><table>
{"".join(f"<tr><td>{tier}</td><td>{count}</td></tr>" for tier,count in s.facts_by_tier.items())}
</table></div>
<div class="card">
<h2>Recent Events</h2><table><tr><th>Time</th><th>Event</th><th>Status</th></tr>
{"".join(f"<tr><td>{e['timestamp'][:19]}</td><td>{e['event']}</td><td>{e['status']}</td></tr>" for e in agim.log.tail(20)[::-1])}
</table></div>
<p style="margin-top:2rem;font-size:0.8rem;color:#8b949e">
<a href="https://github.com/AubakirovArman/agi-personal-memory">GitHub</a></p>
</body></html>"""


def run_server(host: str = "0.0.0.0", port: int = 8720):
    server = HTTPServer((host, port), DashboardHandler)
    print(f"AGI Personal Memory Dashboard: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        server.server_close()
