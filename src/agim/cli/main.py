"""AGIM CLI — command-line interface for AGI Personal Memory."""
import argparse
import os
import sys
from pathlib import Path

from ..core.state import Intent
from ..core.system import AGIMSystem
from .extractor import MemoryExtractor
from .intent_router import IntentRouter


def get_system() -> AGIMSystem:
    workdir = os.environ.get("AGIM_HOME", str(Path.home() / ".agim"))
    return AGIMSystem(workdir=workdir)


def cmd_teach(args):
    agim = get_system()
    router = IntentRouter()
    extractor = MemoryExtractor()
    text = " ".join(args.text)
    intent = router.route(text)
    if intent == Intent.FACT_QUESTION:
        return cmd_ask(args)
    candidate = extractor.extract(text, intent)
    report = agim.compile(candidate)
    if not report.passed:
        print(f"VERIFY FAILED: {report.reason}")
        return 1
    ok = agim.commit(report)
    if ok:
        print(f"MEMORIZED [{candidate.candidate_id}] {candidate.question} → {candidate.answer}")
        print(f"  tier={report.tier.value}")
        for gate in report.gates:
            print(f"  {gate.name}: {'PASS' if gate.passed else 'FAIL'}")
        return 0
    print("COMMIT FAILED")
    return 1


def cmd_ask(args):
    agim = get_system()
    question = " ".join(args.question)
    response = agim.ask(question)
    print(f"Q: {response.question}")
    print(f"A: {response.answer}")
    if response.memory_id:
        print(f"  source={response.source} memory_id={response.memory_id} confidence={response.confidence}")
    return 0


def cmd_correct(args):
    agim = get_system()
    extractor = MemoryExtractor()
    text = " ".join(args.text)
    candidate = extractor.extract(text, Intent.FACT_CORRECT)
    candidate = agim.propose_memory(
        question=candidate.question, answer=candidate.answer,
        kind="fact_correct", confidence=0.9)
    report = agim.compile(candidate)
    if not report.passed:
        print(f"VERIFY FAILED: {report.reason}")
        return 1
    ok = agim.commit(report)
    if ok:
        print(f"CORRECTED [{candidate.candidate_id}] {candidate.question} → {candidate.answer}")
        return 0
    print("COMMIT FAILED")
    return 1


def cmd_forget(args):
    agim = get_system()
    agim.rollback_last()
    if agim.commit_history:
        last = agim.commit_history[-1]
        print(f"ROLLED BACK: {last.question} → {last.answer}")
    else:
        print("ROLLED BACK (last commit undone)")
    return 0


def cmd_history(args):
    agim = get_system()
    entries = agim.log.tail(args.limit)
    if not entries:
        print("(no history yet)")
        return 0
    for e in reversed(entries):
        ts = e["timestamp"][:19]
        event = e["event"]
        status = e["status"]
        detail = e.get("data", {}).get("candidate_id", "") or e.get("data", {}).get("artifact_id", "")
        print(f"  {ts}  {event:<20} {status:<6} {detail}")
    return 0


def cmd_stats(args):
    agim = get_system()
    s = agim.stats()
    print(f"Total facts:      {s.total_facts}")
    print(f"Total commits:    {s.total_commits}")
    print(f"Rollbacks:        {s.rollback_count}")
    print(f"By tier:          {s.facts_by_tier}")
    print(f"By kind:          {s.facts_by_kind}")
    return 0


def cmd_export(args):
    from .export import export_memories
    agim = get_system()
    count = export_memories(agim, args.output)
    print(f"EXPORTED {count} memories to {args.output}")
    return 0


def cmd_import(args):
    from .export import import_memories
    agim = get_system()
    count = import_memories(agim, args.input)
    print(f"IMPORTED {count} memories from {args.input}")
    return 0


def cmd_shell(args):
    agim = get_system()
    router = IntentRouter()
    extractor = MemoryExtractor()
    print("AGI Personal Memory shell. Type /help for commands, /quit to exit.")
    print(f"Workdir: {agim.workdir}")
    while True:
        try:
            line = input("agim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line in ("/help", "/h"):
            print("Commands: /teach <fact> | /ask <question> | /correct <fix> | /forget | /history | /stats | /quit")
            continue
        if line.startswith("/teach "):
            text = line[7:]
            candidate = extractor.extract(text, router.route(text))
            report = agim.compile(candidate)
            if report.passed:
                agim.commit(report)
                print(f"  MEMORIZED [{candidate.candidate_id}] {candidate.answer[:50]}")
            else:
                print(f"  FAILED: {report.reason}")
        elif line.startswith("/ask "):
            resp = agim.ask(line[5:])
            print(f"  {resp.answer}")
        elif line.startswith("/correct "):
            text = line[9:]
            c = extractor.extract(text, Intent.FACT_CORRECT)
            c = agim.propose_memory(question=c.question, answer=c.answer, kind="fact_correct", confidence=0.9)
            report = agim.compile(c)
            if report.passed:
                agim.commit(report)
                print(f"  CORRECTED → {c.answer[:50]}")
            else:
                print(f"  FAILED: {report.reason}")
        elif line == "/forget":
            ok = agim.rollback_last()
            print("  ROLLED BACK" if ok else "  Nothing to roll back")
        elif line == "/history":
            for e in agim.log.tail(10):
                print(f"  {e['timestamp'][:19]} {e['event']}")
        elif line == "/stats":
            s = agim.stats()
            print(f"  facts={s.total_facts} commits={s.total_commits} rollbacks={s.rollback_count}")
        else:
            print("  Unknown. Try /help")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="agim", description="AGI Personal Memory")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("teach", help="Teach a new fact")
    p.add_argument("text", nargs="+")

    p = sub.add_parser("ask", help="Ask a question")
    p.add_argument("question", nargs="+")

    p = sub.add_parser("correct", help="Correct a mistake")
    p.add_argument("text", nargs="+")

    p = sub.add_parser("forget", help="Rollback last commit")

    p = sub.add_parser("history", help="Show memory timeline")
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("export", help="Export all memories to JSON")
    p.add_argument("output", help="Output file path")

    p = sub.add_parser("import", help="Import memories from JSON")
    p.add_argument("input", help="Input file path")

    sub.add_parser("stats", help="Memory statistics")
    p = sub.add_parser("webui", help="Start web dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8720)

    p = sub.add_parser("api", help="Start REST API server (with dashboard)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8720)

    sub.add_parser("demo", help="Run full demo on Llama 3.1 8B")

    sub.add_parser("shell", help="Interactive shell")

    args = parser.parse_args()
    if args.command == "teach": return cmd_teach(args)
    if args.command == "ask": return cmd_ask(args)
    if args.command == "correct": return cmd_correct(args)
    if args.command == "forget": return cmd_forget(args)
    if args.command == "history": return cmd_history(args)
    if args.command == "export": return cmd_export(args)
    if args.command == "import": return cmd_import(args)
    if args.command == "stats": return cmd_stats(args)
    if args.command == "shell": return cmd_shell(args)
    if args.command == "webui":
        from .webui import run_server
        run_server(args.host, args.port)
        return 0
    if args.command == "demo":
        from .demo import main as demo_main
        return demo_main()
    if args.command == "api":
        from .api import run_api
        run_api(args.host, args.port)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
