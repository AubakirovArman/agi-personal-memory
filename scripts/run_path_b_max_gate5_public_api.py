#!/usr/bin/env python3
"""Expose PATH_B_MAX Gate 5 public artifacts via a minimal HTTP contract."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _storage_provider_is_disallowed(provider: Any) -> bool:
    normalized = str(provider or "").strip().lower()
    return normalized in {"", "filesystem", "file_system", "filesystem_storage", "local", "local_filesystem", "mock-object-store"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve PATH_B_MAX Gate 5 public index + release packets",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("results/easyedit_official/governance/path_b_max_gate5_public_index.json"),
        help="Path to public index JSON",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8010, help="HTTP port")
    parser.add_argument(
        "--tenant",
        default=None,
        help="Optional tenant filter for index response",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Optional channel filter for index response",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help="Optional immutable receipt file to expose at /gate5/receipt.json",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=None,
        help="Optional bundle manifest to expose at /gate5/bundle.json",
    )
    parser.add_argument(
        "--transport-manifest-path",
        type=Path,
        default=None,
        help="Optional transport manifest to expose at /gate5/transport-manifest.json",
    )
    parser.add_argument(
        "--require-production-external",
        action="store_true",
        help=(
            "Require production-oriented transport guarantees before serving: external URIs, "
            "non-local immutability, and storage provider metadata"
        ),
    )
    return parser.parse_args()


def filter_records(records: Iterable[Dict[str, Any]], tenant: str | None, channel: str | None) -> list[Dict[str, Any]]:
    filtered = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if tenant is not None and record.get("tenant_scope") != tenant:
            continue
        if channel is not None and record.get("channel") != channel:
            continue
        filtered.append(record)
    return filtered


def main() -> int:
    args = parse_args()
    if not args.index_path.exists():
        print(f"Missing index: {args.index_path}")
        return 1

    payload = read_json(args.index_path)
    if payload.get("index_schema_version") != "path_b_max_gate5_index.v1":
        print("Unexpected index schema version")
        return 1
    if args.require_production_external:
        if args.transport_manifest_path is None or not args.transport_manifest_path.exists():
            print("Transport manifest required for --require-production-external")
            return 1
        transport_payload = read_json(args.transport_manifest_path)
        if transport_payload.get("transport_schema_version") != "path_b_max_gate5_transport_manifest.v1":
            print("Unexpected transport manifest schema version")
            return 1
        coverage = transport_payload.get("coverage")
        if not isinstance(coverage, dict):
            print("Transport manifest coverage missing for production mode")
            return 1
        transport_metadata = transport_payload.get("transport_metadata")
        if not isinstance(transport_metadata, dict):
            print("Transport manifest metadata missing for production mode")
            return 1
        if not coverage.get("has_all_schema_versions"):
            print("Production transport mode requires schema version for all artifacts")
            return 1
        if not coverage.get("has_external_uris"):
            print("Production transport mode requires external URIs")
            return 1
        if not coverage.get("production_ready_candidate"):
            print("Production transport mode requires production_ready_candidate == true")
            return 1
        immutability_mode = str(
            transport_metadata.get("immutability_mode", "")
        ).strip().lower()
        if immutability_mode in {"", "none", "local_only"}:
            print("Production transport mode requires non-local immutability mode")
            return 1
        if not transport_metadata.get("storage_provider"):
            print("Production transport mode requires storage_provider metadata")
            return 1
        if _storage_provider_is_disallowed(transport_metadata.get("storage_provider")):
            print("Production transport mode requires external storage provider, got local/mock provider")
            return 1
        public_base_url = transport_metadata.get("public_base_url", "")
        if not public_base_url:
            print("Production transport mode requires public_base_url metadata")
            return 1
        if isinstance(public_base_url, str):
            base = public_base_url.strip().lower()
            if (
                base.startswith("http://localhost")
                or base.startswith("https://localhost")
                or base.startswith("http://127.0.0.1")
                or base.startswith("https://127.0.0.1")
                or base.startswith("http://0.0.0.0")
                or base.startswith("https://0.0.0.0")
                or base.startswith("http://[::1]")
                or base.startswith("https://[::1]")
            ):
                print("Production transport mode requires non-local public_base_url")
                return 1
        artifacts = transport_payload.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            print("Production transport mode requires transport manifest artifacts")
            return 1
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                print("Invalid transport manifest artifact")
                return 1
            if not artifact.get("schema_version"):
                print(f"Missing schema_version for transport artifact {artifact.get('kind', 'unknown')}")
                return 1
            if not artifact.get("external_uri"):
                print(f"Missing external_uri for transport artifact {artifact.get('kind', 'unknown')}")
                return 1

    records = filter_records(payload.get("records", []), args.tenant, args.channel)
    release_by_sha = {record.get("release_sha256"): record for record in records}

    class PublicGate5Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, body_obj: Any) -> None:
            body = json.dumps(body_obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, content: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            if self.path in {"/gate5/index", "/gate5/index.json"}:
                response = {
                    "index_schema_version": payload.get("index_schema_version"),
                    "updated_utc": payload.get("updated_utc"),
                    "records": records,
                    "count": len(records),
                    "tenant_filter": args.tenant,
                    "channel_filter": args.channel,
                }
                self._send_json(200, response)
                return

            if self.path.startswith("/gate5/releases/") and self.path.endswith(".json"):
                sha = self.path.split("/gate5/releases/")[-1].replace(".json", "")
                record = release_by_sha.get(sha)
                if not record:
                    self._send_json(404, {"error": "release not found"})
                    return
                release_path = Path(record.get("release_path", ""))
                if not release_path.exists():
                    self._send_json(404, {"error": "release file missing"})
                    return
                try:
                    raw = release_path.read_bytes()
                    json.loads(raw.decode("utf-8"))
                except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._send_json(500, {"error": f"failed to load release payload: {e}"})
                    return
                self._send_bytes(200, raw)
                return

            if self.path in {"/gate5/receipt", "/gate5/receipt.json"}:
                if args.receipt_path is None:
                    self._send_json(404, {"error": "receipt unavailable"})
                    return
                if not args.receipt_path.exists():
                    self._send_json(404, {"error": "receipt file missing"})
                    return
                try:
                    raw_receipt = args.receipt_path.read_bytes()
                    json.loads(raw_receipt.decode("utf-8"))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._send_json(500, {"error": f"failed to load receipt payload: {e}"})
                    return
                self._send_bytes(200, raw_receipt)
                return
            if self.path in {"/gate5/bundle", "/gate5/bundle.json"}:
                if args.bundle_path is None:
                    self._send_json(404, {"error": "bundle unavailable"})
                    return
                if not args.bundle_path.exists():
                    self._send_json(404, {"error": "bundle file missing"})
                    return
                try:
                    raw_bundle = args.bundle_path.read_bytes()
                    json.loads(raw_bundle.decode("utf-8"))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._send_json(500, {"error": f"failed to load bundle payload: {e}"})
                    return
                self._send_bytes(200, raw_bundle)
                return
            if self.path in {"/gate5/transport-manifest", "/gate5/transport-manifest.json"}:
                if args.transport_manifest_path is None:
                    self._send_json(404, {"error": "transport manifest unavailable"})
                    return
                if not args.transport_manifest_path.exists():
                    self._send_json(404, {"error": "transport manifest file missing"})
                    return
                try:
                    raw_manifest = args.transport_manifest_path.read_bytes()
                    json.loads(raw_manifest.decode("utf-8"))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._send_json(
                        500,
                        {"error": f"failed to load transport manifest payload: {e}"},
                    )
                    return
                self._send_bytes(200, raw_manifest)
                return

            self._send_json(404, {"error": "not found"})

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer((args.host, args.port), PublicGate5Handler)
    print(f"Serving PATH_B_MAX Gate5 public API on http://{args.host}:{args.port}")
    print("Available endpoints:")
    print("  /gate5/index.json")
    print("  /gate5/releases/<release_sha>.json")
    print("  /gate5/receipt.json")
    print("  /gate5/bundle.json")
    print("  /gate5/transport-manifest.json")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
