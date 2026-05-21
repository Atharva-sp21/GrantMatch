import requests
import json
import time
from collections import defaultdict

def fetch_researchers(limit=1000):
    """
    Fetch researchers from OpenAlex API.
    Returns list of researchers with institution affiliations.
    No API key required - all requests free.
    """
    researchers = []
    institution_ids = {}  # Map external IDs to our internal IDs
    institution_counter = 0

    url = "https://api.openalex.org/authors"
    params = {
        "per_page": 100,
        "page": 1,
        "sort": "works_count:desc"  # Get most prolific researchers first
    }

    print("[INFO] Fetching researchers from OpenAlex...")

    while len(researchers) < limit:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get('results'):
                print(f"[OK] Fetched {len(researchers)} researchers total")
                break

            for result in data['results']:
                if len(researchers) >= limit:
                    break

                # Get primary affiliation
                affiliation_id = None
                if result.get('last_known_institution'):
                    ext_id = result['last_known_institution']['id']
                    if ext_id not in institution_ids:
                        institution_ids[ext_id] = institution_counter
                        institution_counter += 1
                    affiliation_id = institution_ids[ext_id]

                researcher = {
                    'id': len(researchers),  # Must match array index
                    'openalex_id': result['id'],
                    'name': result['display_name'],
                    'institution_id': affiliation_id,
                    'works_count': result['works_count'],
                    'cited_by_count': result['cited_by_count'],
                    'h_index': result.get('summary_stats', {}).get('h_index', 0),
                    'i10_index': result.get('summary_stats', {}).get('i10_index', 0),
                    'last_known_institution': result.get('last_known_institution', {}).get('display_name')
                }
                researchers.append(researcher)

            params['page'] += 1
            time.sleep(0.1)  # Respectful rate limiting

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error fetching page {params['page']}: {e}")
            break

    print(f"[OK] Fetched {len(researchers)} researchers")
    print(f"[OK] Found {len(institution_ids)} institutions")

    return researchers, institution_ids


if __name__ == "__main__":
    researchers, institutions = fetch_researchers(limit=1000)

    # Save for next step
    with open('../data/raw/researchers_raw.json', 'w') as f:
        json.dump(researchers, f, indent=2)

    with open('../data/raw/institution_mapping.json', 'w') as f:
        json.dump(institutions, f, indent=2)

    print("[OK] Saved researchers_raw.json and institution_mapping.json")
