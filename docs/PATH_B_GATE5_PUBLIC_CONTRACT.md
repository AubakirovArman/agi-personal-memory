# PATH_B_MAX Gate 5 Public Contract

This contract defines how PATH_B_MAX Gate 5 release artifacts are made discoverable outside the local repo.

## Scope and artifact types

1. `results/easyedit_official/governance/path_b_max_gate5_public_release.json`
2. `results/easyedit_official/governance/path_b_max_gate5_public_index.json`

## Release packet contract

`run_path_b_max_gate5_release_packet.py` must produce a packet with:

- `release_schema_version`: exactly `path_b_max_gate5_release.v1`
- `tenant_scope`
- `source_path`
- `source_hash`
- `proof` object with:
  - `payload` and `proof` (as produced by `run_path_b_max_patch_service_governance_proof.py`)
  - `proof_sha256`
- `claim_lock` object with:
  - `audit_chain_len`
  - `audit_chain_head`
  - `audit_chain_sha256`
- `evidence_paths`: list of supporting evidence files
- `signature` (nullable, string payload if present in source proof)
- `verifier`: metadata used by verifier CLI

## Index contract

`run_path_b_max_publish_gate5_release.py` appends records to:

- `results/easyedit_official/governance/path_b_max_gate5_public_index.json`

Index schema:

- `index_schema_version`: `path_b_max_gate5_index.v1`
- `updated_utc`
- `records`: array of:
  - `tenant_scope`
  - `channel`
  - `release_path`
  - `release_sha256`
  - `proof_sha256`
  - `claim_chain_sha256`
  - `claim_chain_len`
  - `signature`
  - `created_utc`
  - `source_path`

## Verifier contract

An external consumer that trusts a record for `(tenant_scope, channel)` must:

1. read the matching record from the index
2. verify the release file exists at `release_path`
3. hash file at `release_path` and compare with `release_sha256`
4. load release packet and ensure it is well-formed and self-consistent (`proof_sha256`, claim fields)
5. optionally fetch `source_path` and verify any transport-appropriate integrity signature

## CLI entry points

- `run_path_b_max_gate5_release_packet.py --proof-path ...`
- `run_path_b_max_verify_gate5_release.py --tenant <tenant> --release-path ...`
- `run_path_b_max_publish_gate5_release.py --release-path ... --tenant <tenant> --channel <channel>`
- `run_path_b_max_verify_gate5_index.py [--tenant <tenant>] [--channel <channel>]`
- `run_path_b_max_gate5_public_api.py --index-path ... [--tenant ...] [--channel ...] [--receipt-path ...] [--bundle-path ...] [--transport-manifest-path ...] --port ...`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel>`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-receipt --expected-receipt-sha256 <sha256>`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-transport-manifest --expected-transport-manifest-sha256 <sha256>`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --expected-release-schema-version path_b_max_gate5_release.v1`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --expected-release-sha256 <sha256>`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --expected-release-schema-version path_b_max_gate5_release.v1`
- `run_path_b_max_gate5_create_receipt.py --index-path ... --tenant <tenant> --channel <channel>`
- `run_path_b_max_verify_gate5_receipt.py --receipt-path ...`
- `run_path_b_max_gate5_create_bundle.py --index-path ... --receipt-path ...`
- `run_path_b_max_verify_gate5_bundle.py --bundle-path ...`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-bundle --expected-bundle-sha256 <sha256>`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-receipt --expected-receipt-schema-version path_b_max_gate5_receipt.v1`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-transport-manifest --expected-transport-manifest-schema-version path_b_max_gate5_transport_manifest.v1`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-transport-manifest --require-production-external`
- `run_path_b_max_gate5_audit_consumer.py --api-base ... --tenant <tenant> --channel <channel> --check-bundle --expected-bundle-schema-version path_b_max_gate5_bundle.v1`
- `run_path_b_max_gate5_verify_publication.py --api-base ... --expected-release-sha256 <release-sha256> --expected-receipt-sha256 <receipt-sha256> --expected-bundle-sha256 <bundle-sha256> --expected-transport-manifest-sha256 <transport-manifest-sha256> [--transport-manifest-path <path>] [--expected-release-schema-version path_b_max_gate5_release.v1] [--expected-receipt-schema-version path_b_max_gate5_receipt.v1] [--expected-bundle-schema-version path_b_max_gate5_bundle.v1] [--expected-transport-manifest-schema-version path_b_max_gate5_transport_manifest.v1] [--check-transport-manifest] [--require-production-external]`
- `run_path_b_max_gate5_create_transport_manifest.py --release-path ... --index-path ... --receipt-path ... --bundle-path ... --manifest-path ...`
- `run_path_b_max_gate5_create_transport_manifest.py --release-path ... --index-path ... --receipt-path ... --bundle-path ... --manifest-path ... [--tenant <tenant>] [--channel <channel>] [--public-base-url <url>] [--storage-provider <provider>] [--immutability-mode object_lock|versioned_bucket|signed_pointers] [--external-index-uri <uri>] [--external-release-uri <uri>] [--external-receipt-uri <uri>] [--external-bundle-uri <uri>] [--require-production-external]`
- `run_path_b_max_verify_gate5_transport_manifest.py --manifest-path ...`
- `run_path_b_max_verify_gate5_transport_manifest.py --manifest-path ... --require-production-external`
- production smoke run pattern:
  `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1 AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER=<PRODUCTION_IMMUTABLE_PROVIDER> AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE=object_lock AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api AGIM_GATE5_PUBLIC_API_SMOKE=1 ./scripts/run_path_b_max_hard_gates.sh 5`

In production mode (`--require-production-external` / `AGIM_GATE5_REQUIRE_PRODUCTION_EXTERNAL=1`), external transport creation additionally requires all of:
- `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER`
- `AGIM_GATE5_TRANSPORT_IMMUTABILITY_MODE`
- `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL`

## API endpoints (local consumer contract)

- `GET /gate5/index`
- `GET /gate5/index.json`
- `GET /gate5/releases/<release_sha256>.json` (raw JSON blob bytes, no re-serialization)
- `GET /gate5/receipt.json` (raw receipt bytes)
- `GET /gate5/bundle.json` (raw immutable bundle bytes)
- `GET /gate5/transport-manifest.json` (raw transport manifest bytes)

The release endpoint uses `release_sha256` from index records to resolve and return matching `public_release.json`.

## Current status

The current implementation provides:

- deterministic local packet generation
- local proof verification
- tenant+channel index publication
- index verification script that enforces matching `tenant_scope` and `channel`
- local consumer API endpoint for read access to index and release blobs
- release SHA checks are stable because release endpoint returns raw on-disk bytes for immutable hash verification
- release schema validation in public API consumer (`run_path_b_max_gate5_audit_consumer.py`)
- immutable receipt bundle generation and verification for public index/release assets
- bundle manifest generation and verification for immutable end-to-end public transport
- local transport manifest generation and verification for URI + hash + schema metadata of release/index/receipt/bundle
- production-mode transport manifest creation guard (`--require-production-external`) that requires all schemas, all external URIs, immutable transport mode metadata, and storage metadata
- Gate 5 hard-gates invoke `run_path_b_max_gate5_verify_publication.py` with explicit digest and schema arguments in both local/offline and API smoke mode for release/receipt/bundle/transport manifest integrity checks.
- API consumer now validates release consistency against index metadata (`claim_chain_sha256`, `claim_chain_len`, `proof_sha256`) before accepting release responses.
- Local publication verifier (`run_path_b_max_gate5_verify_publication.py`) additionally cross-checks index record fields (`claim_chain_sha256`, `claim_chain_len`, `proof_sha256`) against loaded release packet before API/receipt/bundle/manifest checks.

Outstanding for full external rollout:

 - production-grade immutable transport/storage guarantees for `path_b_max_gate5_public_index.json` and release packets (API smoke and `--require-production-external` now enforce manifest coverage and transport-mode checks; full immutable external storage guarantees still depend on actual external URI-backed artifact delivery and signed/persistent immutability)
- in production mode, `AGIM_GATE5_TRANSPORT_PUBLIC_BASE_URL` must be non-local (not localhost/127.0.0.1/0.0.0.0/[::1] loopback) and `AGIM_GATE5_TRANSPORT_STORAGE_PROVIDER` must be a true external immutable backend (not `mock-object-store` or filesystem-style provider)
