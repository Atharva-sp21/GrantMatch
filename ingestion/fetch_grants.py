import html
import json
import os
import re
from typing import Any, Dict, Iterable, List

import requests

NIH_URL = "https://api.reporter.nih.gov/v2/projects/search"
NIH_PAGE_SIZE = 100
REQUEST_TIMEOUT = 30
DEFAULT_LIMIT = 2000


def _ensure_agency(agencies_dict: Dict[str, int], agency_name: str) -> int:
    if agency_name not in agencies_dict:
        agencies_dict[agency_name] = len(agencies_dict)
    return agencies_dict[agency_name]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def _collect_names(value: Any) -> List[str]:
    names: List[str] = []
    if not value:
        return names

    if isinstance(value, str):
        pieces = [piece.strip() for piece in re.split(r"[;|]", value)]
        return [piece for piece in pieces if piece]

    if isinstance(value, dict):
        for key in (
            "full_name",
            "name",
            "display_name",
            "pi_name",
            "principal_investigator_name",
            "contact_pi_name",
        ):
            if value.get(key):
                return _collect_names(value.get(key))
        first_name = value.get("first_name")
        last_name = value.get("last_name")
        combined = " ".join(part for part in [first_name, last_name] if part)
        return [combined] if combined else []

    if isinstance(value, list):
        for item in value:
            names.extend(_collect_names(item))

    return names


def _extract_investigators(record: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    investigators: List[str] = []
    seen = set()
    for key in keys:
        for name in _collect_names(record.get(key)):
            normalized = " ".join(name.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                investigators.append(normalized)
    return investigators


def _extract_text(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    return ""


def _request_json(session: requests.Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = session.post(NIH_URL, json=payload, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        raise RuntimeError(
            f"NIH Reporter request failed with {response.status_code}: {response.text[:1000]}"
        )
    response.raise_for_status()
    return response.json()


def fetch_nih_grants(limit: int = 500):
    """Fetch real grant records from NIH Reporter."""
    grants: List[Dict[str, Any]] = []
    agency_ids: Dict[str, int] = {}
    agency_id = _ensure_agency(agency_ids, "NIH")

    print("[INFO] Fetching NIH grants from Reporter...")

    offset = 0
    session = requests.Session()
    while len(grants) < limit:
        page_size = min(NIH_PAGE_SIZE, limit - len(grants))
        payload = {
            "criteria": {
                "include_active_projects": True,
                "fiscal_years": [],
                "use_relevance": False,
            },
            "offset": offset,
            "limit": page_size,
            "sort_field": "project_end_date",
            "sort_order": "desc",
        }

        data = _request_json(session, payload)
        results = data.get("results", [])
        if not results:
            break

        for result in results:
            if len(grants) >= limit:
                break

            amount = _to_float(
                result.get("total_cost")
                or result.get("award_amount")
                or result.get("project_total_cost")
                or result.get("direct_cost")
            )

            title = _clean_text(result.get("project_title") or result.get("title") or "")
            abstract = _extract_text(
                result,
                ("abstract_text", "project_abstract", "project_abstract_text", "abstract"),
            )

            grant = {
                "id": len(grants),
                "source": "NIH Reporter",
                "nih_project_number": _clean_text(result.get("project_num") or result.get("project_number") or ""),
                "title": title,
                "agency_id": agency_id,
                "amount": amount,
                "start_date": _clean_text(result.get("project_start_date") or result.get("start_date") or ""),
                "end_date": _clean_text(result.get("project_end_date") or result.get("end_date") or ""),
                "investigators": _extract_investigators(
                    result,
                    (
                        "principal_investigators",
                        "project_investigators",
                        "pi_names",
                        "contact_pi_name",
                        "investigator",
                    ),
                ),
                "abstract": abstract,
            }
            grants.append(grant)

        offset += len(results)
        if len(results) < page_size:
            break

    if not grants:
        raise RuntimeError("NIH Reporter returned no grant records.")

    print(f"[OK] Fetched {len(grants)} NIH grant records")
    return grants, agency_ids


if __name__ == "__main__":
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    os.makedirs(output_dir, exist_ok=True)

    grants, agencies = fetch_nih_grants(limit=DEFAULT_LIMIT)

    grants_path = os.path.join(output_dir, "grants_raw.json")
    agencies_path = os.path.join(output_dir, "agency_mapping.json")

    with open(grants_path, "w", encoding="utf-8") as handle:
        json.dump(grants, handle, indent=2)

    with open(agencies_path, "w", encoding="utf-8") as handle:
        json.dump(agencies, handle, indent=2)

    print(f"[OK] Saved {grants_path}")
    print(f"[OK] Saved {agencies_path}")
