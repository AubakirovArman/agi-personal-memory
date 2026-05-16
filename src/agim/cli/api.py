"""REST API for AGI Personal Memory."""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from ..core.state import Intent, MemoryCandidate
from ..core.system import AGIMSystem
from .extractor import MemoryExtractor
from .intent_router import IntentRouter

_AGIM: AGIMSystem | None = None
_ROUTER = IntentRouter()
_EXTRACTOR = MemoryExtractor()


def get_agim() -> AGIMSystem:
    global _AGIM
    if _AGIM is None:
        workdir = os.environ.get("AGIM_HOME", str(Path.home() / ".agim"))
        _AGIM = AGIMSystem(workdir=workdir)
    return _AGIM


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        agim = get_agim()
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)

        if path == "/api/ask":
            question = qs.get("q", [""])[0]
            resp = agim.ask(question)
            self._json({"answer": resp.answer, "source": resp.source,
                       "memory_id": resp.memory_id, "confidence": resp.confidence})
        elif path == "/api/stats":
            self._json(agim.stats().__dict__)
        elif path == "/api/history":
            limit = int(qs.get("limit", ["50"])[0])
            self._json(agim.log.tail(limit))
        elif path == "/api/memories":
            memories = {**agim.retrieval._data, **agim.refusals._data}
            search = qs.get("search", [""])[0].lower()
            filtered = {k: v for k, v in memories.items()
                       if not search or search in k.lower() or search in str(v).lower()}
            self._json({"count": len(filtered), "memories": filtered})
        elif path == "/api/provenance":
            self._json({
                "length": agim.provenance.length,
                "valid": agim.provenance.verify_chain(),
                "chain": [{"commit_id": r.commit_id, "timestamp": r.timestamp}
                         for r in agim.provenance.chain],
            })
        elif path == "/api/risk":
            self._json({
                "total_risky": agim.risk_ledger.total_risky,
                "total_dangerous": agim.risk_ledger.total_dangerous,
                "entries": [{"memory_id": e.memory_id, "score": e.risk_score,
                            "auto_rollback": e.auto_rollback}
                           for e in agim.risk_ledger.entries[-20:]],
            })
        elif path in ("/", "/dashboard"):
            self._serve_dashboard(agim)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        agim = get_agim()
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}
        text = body.get("text", "")

        if path == "/api/teach":
            intent = _ROUTER.route(text) if text else Intent.FACT_TEACH
            cand = _EXTRACTOR.extract(text, intent) if text else MemoryCandidate(
                question=body.get("question", ""), answer=body.get("answer", ""),
                kind=body.get("kind", "fact_teach"))
            if not text:
                cand = MemoryCandidate(**{k: v for k, v in body.items()
                           if k in ["question", "answer", "kind", "source", "confidence"]})
            report = agim.compile(cand)
            if report.passed: agim.commit(report)
            self._json({"status": report.status, "candidate_id": cand.candidate_id,
                       "tier": report.tier.value, "reason": report.reason})

        elif path == "/api/correct":
            cand = _EXTRACTOR.extract(text, Intent.FACT_CORRECT)
            cand2 = agim.propose_memory(question=cand.question, answer=cand.answer,
                                        kind="fact_correct", confidence=0.9)
            report = agim.compile(cand2)
            if report.passed: agim.commit(report)
            self._json({"status": report.status, "answer": cand.answer})

        elif path == "/api/forget":
            ok = agim.rollback_last()
            self._json({"status": "PASS" if ok else "FAIL"})

        elif path == "/api/protect":
            agim.add_protected_fact(body.get("question", ""), body.get("answer", ""))
            self._json({"status": "PASS"})

        elif path == "/api/regression":
            results = agim.run_regression()
            self._json({"results": results, "all_pass": all(results.values())})

        else:
            self.send_response(404); self.end_headers()

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode())

    def _serve_dashboard(self, agim):
        s = agim.stats()
        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AGI Personal Memory</title>
<style>
*{{box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
max-width:1000px;margin:0 auto;padding:1rem;background:#0d1117;color:#c9d1d9}}
h1{{color:#58a6ff;margin:0}}h2{{color:#f0883e;margin-top:1.5rem}}
.card{{background:#161b22;border-radius:8px;padding:1rem;margin:0.5rem 0}}
.stat{{display:inline-block;margin:0.5rem 1.5rem 0.5rem 0;text-align:center}}
.stat-num{{font-size:2rem;color:#58a6ff}}.stat-label{{font-size:0.75rem;color:#8b949e}}
input,button{{padding:8px 12px;border-radius:6px;border:1px solid #30363d;
background:#0d1117;color:#c9d1d9;font-size:14px;margin:4px}}
button{{background:#238636;border:none;cursor:pointer}}button:hover{{background:#2ea043}}
button.danger{{background:#da3633}}button.danger:hover{{background:#f85149}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:#f0883e;padding:6px 8px}}
td{{padding:6px 8px;border-top:1px solid #30363d;word-break:break-word}}
.row{{display:flex;gap:0.5rem;flex-wrap:wrap}}
.col{{flex:1;min-width:300px}}
#search{{width:100%}}
#result{{margin-top:1rem;padding:1rem;background:#161b22;border-radius:8px;min-height:40px}}
.tab{{display:inline-block;padding:8px 16px;cursor:pointer;border-radius:8px 8px 0 0;
margin-right:4px;background:#161b22}}
.tab.active{{background:#1f2a3a;color:#58a6ff}}
</style></head><body>
<div class="row" style="align-items:center;justify-content:space-between">
<h1>AGI Personal Memory</h1>
<div><span class="stat"><span class="stat-num" id="facts">{s.total_facts}</span>
<span class="stat-label">Facts</span></span>
<span class="stat"><span class="stat-num" id="commits">{s.total_commits}</span>
<span class="stat-label">Commits</span></span></div></div>

<div class="card">
<div class="tab active" onclick="showTab('ask')">Ask</div>
<div class="tab" onclick="showTab('teach')">Teach</div>
<div class="tab" onclick="showTab('history')">History</div>
<div class="tab" onclick="showTab('search')">Search</div>
<div class="tab" onclick="showTab('governance')">Governance</div>
</div>

<div class="card" id="tab-ask">
<h2>Ask a Question</h2>
<div class="row"><input id="question" placeholder="What do you want to know?" style="flex:1">
<button onclick="apiAsk()">Ask</button></div>
<div id="result"></div>
</div>

<div class="card" id="tab-teach" style="display:none">
<h2>Teach a Fact</h2>
<div class="row"><input id="teachText" placeholder="Paris is the capital of France" style="flex:1">
<button onclick="apiTeach()">Teach</button>
<button class="danger" onclick="apiForget()">Undo Last</button></div>
<div id="teachResult"></div>

<h2 style="margin-top:1.5rem">Protect a Fact</h2>
<div class="row"><input id="protectQ" placeholder="Question" style="flex:1">
<input id="protectA" placeholder="Answer" style="flex:1">
<button onclick="apiProtect()">Protect</button>
<button onclick="apiRegression()">Run Regression</button></div>
<div id="protectResult"></div>
</div>

<div class="card" id="tab-history" style="display:none">
<h2>Recent Events</h2><div id="historyTable"></div>
</div>

<div class="card" id="tab-search" style="display:none">
<h2>Search Memories</h2>
<div class="row"><input id="search" placeholder="Type to search..." style="flex:1"
oninput="apiSearch()"></div>
<div id="searchResult"></div>
</div>

<div class="card" id="tab-governance" style="display:none">
<h2>Governance</h2><div id="govContent"></div>
</div>

<script>
function showTab(t){{document.querySelectorAll('.tab').forEach(el=>el.classList.remove('active'));
event.target.classList.add('active');
document.querySelectorAll('[id^="tab-"]').forEach(el=>el.style.display='none');
document.getElementById('tab-'+t).style.display='block';
if(t=='history')loadHistory();if(t=='governance')loadGovernance();if(t=='search')apiSearch()}}

async function apiAsk(){{
const q=document.getElementById('question').value;
const r=await fetch('/api/ask?q='+encodeURIComponent(q));const d=await r.json();
document.getElementById('result').innerHTML=
'<strong>A:</strong> '+d.answer+'<br><small>source='+d.source+' confidence='+d.confidence+'</small>';}}

async function apiTeach(){{
const t=document.getElementById('teachText').value;
const r=await fetch('/api/teach',{{method:'POST',body:JSON.stringify({{text:t}})}});
const d=await r.json();
document.getElementById('teachResult').innerHTML='<strong>'+d.status+'</strong> '+d.tier+' '+(d.reason||'');
document.getElementById('facts').textContent=parseInt(document.getElementById('facts').textContent)+1;}}

async function apiForget(){{
const r=await fetch('/api/forget',{{method:'POST'}});const d=await r.json();
document.getElementById('teachResult').innerHTML='<strong>'+d.status+'</strong> rollback';
if(d.status=='PASS')document.getElementById('facts').textContent=Math.max(0,parseInt(document.getElementById('facts').textContent)-1);}}

async function apiProtect(){{
const q=document.getElementById('protectQ').value,a=document.getElementById('protectA').value;
const r=await fetch('/api/protect',{{method:'POST',body:JSON.stringify({{question:q,answer:a}})}});
const d=await r.json();
document.getElementById('protectResult').innerHTML='<strong>'+d.status+'</strong> protected';}}

async function apiRegression(){{
const r=await fetch('/api/regression',{{method:'POST'}});const d=await r.json();
document.getElementById('protectResult').innerHTML='Regression: '+(d.all_pass?'ALL PASS':'SOME FAIL')+
'<br>'+JSON.stringify(d.results);}}

async function apiSearch(){{
const q=document.getElementById('search').value;
const r=await fetch('/api/memories?search='+encodeURIComponent(q));const d=await r.json();
let h='<table><tr><th>Question</th><th>Answer</th><th>Source</th></tr>';
for(const[k,v]of Object.entries(d.memories||{{}}))
h+='<tr><td>'+k+'</td><td>'+v.answer+'</td><td>'+v.source+'</td></tr>';
h+='</table>';document.getElementById('searchResult').innerHTML=h;}}

async function loadHistory(){{
const r=await fetch('/api/history?limit=50');const d=await r.json();
let h='<table><tr><th>Time</th><th>Event</th><th>Status</th><th>Detail</th></tr>';
for(const e of d.reverse())
h+='<tr><td>'+e.timestamp+'</td><td>'+e.event+'</td><td>'+e.status+'</td><td>'+JSON.stringify(e.data).slice(0,60)+'</td></tr>';
h+='</table>';document.getElementById('historyTable').innerHTML=h;}}

async function loadGovernance(){{
const[prov,risk]=await Promise.all([
fetch('/api/provenance').then(r=>r.json()),
fetch('/api/risk').then(r=>r.json())]);
document.getElementById('govContent').innerHTML=
'<h3>Provenance Chain</h3>Length: '+prov.length+' Valid: '+prov.valid+
'<br><br><h3>Risk Ledger</h3>Risky: '+risk.total_risky+' Dangerous: '+risk.total_dangerous+
'<br>'+risk.entries.map(e=>'<span style="color:'+(e.auto_rollback?'#f85149':'#8b949e')+'">'+e.memory_id+':'+e.score+'</span>').join(', ');}}
</script></body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_api(host: str = "0.0.0.0", port: int = 8720):
    print(f"AGI Personal Memory API: http://{host}:{port}")
    print(f"Dashboard: http://{host}:{port}/dashboard")
    HTTPServer((host, port), APIHandler).serve_forever()
