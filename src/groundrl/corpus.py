"""Corpus acquisition: a curated EIP list, download helpers and an integrity manifest.

The EIPs are CC0. Files are fetched from ``raw.githubusercontent.com`` only. Documents that were
moved out of the EIPs repository (the ERC category) are tiny redirect stubs there; for those the
ERCs repository is tried next. Anything under :data:`MIN_BYTES` after both attempts is dropped.
Network access is injected (``opener``) so the logic is testable offline.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

MIN_BYTES = 300
EIPS_URL = "https://raw.githubusercontent.com/ethereum/EIPs/master/EIPS/eip-{n}.md"
ERCS_URL = "https://raw.githubusercontent.com/ethereum/ERCs/master/ERCS/erc-{n}.md"
MANIFEST_NAME = "MANIFEST.json"

# Curated by hand: protocol-defining and widely cited proposals, plus recent hard-fork content
# that is rich in checkable constants (gas costs, opcode bytes, transaction type bytes, caps).
# Entries that do not resolve or are smaller than MIN_BYTES are dropped and recorded as skipped.
CURATED_EIPS: tuple[int, ...] = (
    # process / early protocol
    1, 2, 5, 6, 7, 8, 100, 140, 145, 150, 152, 155, 158, 160, 161, 170,
    196, 197, 198, 211, 214, 649, 658, 684, 706,
    # Constantinople .. London
    1014, 1052, 1108, 1153, 1283, 1344, 1559, 1884, 1967, 2028, 2124, 2200, 2315, 2384,
    2481, 2537, 2565, 2718, 2929, 2930, 2935, 3074, 3198, 3529, 3541, 3554, 3607, 3651,
    3675, 3855, 3860, 4345, 4399, 4750, 4788, 4844, 4895, 5133, 5656, 5793, 6049, 6110,
    6780, 6800, 6963,
    # Prague / Electra / Fusaka and later hard-fork content
    7002, 7044, 7045, 7251, 7514, 7516, 7523, 7549, 7594, 7623, 7642, 7685, 7691, 7692,
    7702, 7732, 7745, 7805, 7823, 7825, 7840, 7883, 7903, 7907, 7918, 7928, 7934, 7939,
    7951, 7954,
    # application-layer standards (ERC category; may live in the ERCs repository)
    20, 55, 137, 165, 173, 191, 712, 721, 777, 1155, 1167, 1191, 1271, 1820, 2098, 2309,
    2535, 2612, 2981, 3156, 3525, 4337, 4626, 4907, 5267, 6372, 6551, 6909, 7201, 7861,
)

_FRONT_MATTER = re.compile(r"^---\r?\n(.*?)\r?\n---", re.DOTALL)


@dataclass(frozen=True)
class FetchResult:
    number: int
    ok: bool
    source: str  # url the bytes came from, or "" when skipped
    content: bytes = b""
    reason: str = ""


Opener = Callable[[str], bytes]


def doc_id(number: int) -> str:
    return f"eip-{number}"


def urllib_opener(timeout: float = 30.0, retries: int = 3, pause: float = 0.2) -> Opener:
    """A stdlib downloader restricted to raw.githubusercontent.com with simple retry/backoff."""
    import urllib.error
    import urllib.request

    def open_url(url: str) -> bytes:
        if not url.startswith("https://raw.githubusercontent.com/"):
            raise ValueError(f"refusing to fetch outside raw.githubusercontent.com: {url}")
        last: Exception | None = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                    data = resp.read()
                time.sleep(pause)
                return data
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise FileNotFoundError(url) from exc
                last = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            time.sleep(0.5 * (attempt + 1))
        raise ConnectionError(f"could not fetch {url}: {last}")

    return open_url


def fetch_one(number: int, opener: Opener, min_bytes: int = MIN_BYTES) -> FetchResult:
    """Fetch EIP ``number``; fall back to the ERCs repository for moved/stub files."""
    reasons: list[str] = []
    for template in (EIPS_URL, ERCS_URL):
        url = template.format(n=number)
        try:
            data = opener(url)
        except FileNotFoundError:
            reasons.append(f"404 {url}")
            continue
        if len(data) < min_bytes:
            reasons.append(f"{len(data)} bytes < {min_bytes} at {url}")
            continue
        return FetchResult(number, True, url, data)
    return FetchResult(number, False, "", b"", "; ".join(reasons))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(
    numbers: Iterable[int],
    out_dir: Path,
    opener: Opener,
    min_bytes: int = MIN_BYTES,
) -> dict:
    """Download ``numbers`` into ``out_dir`` and return the manifest dict (also written to disk)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    skipped: list[dict] = []
    for n in sorted(set(numbers)):
        res = fetch_one(n, opener, min_bytes)
        if not res.ok:
            skipped.append({"number": n, "reason": res.reason})
            continue
        path = out_dir / f"{doc_id(n)}.md"
        path.write_bytes(res.content)
        kept.append({
            "id": doc_id(n),
            "number": n,
            "file": path.name,
            "bytes": len(res.content),
            "sha256": sha256_hex(res.content),
            "source": res.source,
        })
    manifest = {
        "license": "CC0-1.0 (Ethereum Improvement Proposals)",
        "min_bytes": min_bytes,
        "documents": kept,
        "skipped": skipped,
    }
    write_manifest(out_dir.parent / MANIFEST_NAME, manifest)
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(data_dir: Path) -> list[str]:
    """Return a list of integrity problems (empty means the raw files match the manifest)."""
    data_dir = Path(data_dir)
    manifest = read_manifest(data_dir / MANIFEST_NAME)
    problems: list[str] = []
    listed = set()
    for entry in manifest["documents"]:
        listed.add(entry["file"])
        path = data_dir / "raw" / entry["file"]
        if not path.exists():
            problems.append(f"missing {entry['file']}")
            continue
        blob = path.read_bytes()
        if sha256_hex(blob) != entry["sha256"]:
            problems.append(f"sha256 mismatch for {entry['file']}")
        if len(blob) != entry["bytes"]:
            problems.append(f"size mismatch for {entry['file']}")
        if len(blob) < manifest.get("min_bytes", MIN_BYTES):
            problems.append(f"{entry['file']} is below the minimum size")
    for path in sorted((data_dir / "raw").glob("*.md")):
        if path.name not in listed:
            problems.append(f"unlisted file {path.name}")
    return problems


def front_matter(raw: str) -> dict[str, str]:
    """Parse the flat ``key: value`` YAML-ish header of an EIP; keys are lower-cased."""
    m = _FRONT_MATTER.match(raw)
    meta: dict[str, str] = {}
    if not m:
        return meta
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta


def load_documents(raw_dir: Path) -> dict[str, str]:
    """``{"eip-1559": raw markdown}`` for every ``eip-N.md`` file under ``raw_dir``."""
    docs: dict[str, str] = {}
    for path in sorted(Path(raw_dir).glob("eip-*.md"), key=lambda p: _number_of(p.stem)):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def _number_of(stem: str) -> int:
    return int(stem.split("-", 1)[1])
