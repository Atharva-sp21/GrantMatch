import json
import numpy as np
from pathlib import Path

def load_raw_data():
    """Load intermediate data from fetch scripts."""
    with open('../data/raw/researchers_raw.json', 'r') as f:
        researchers_raw = json.load(f)

    with open('../data/raw/grants_raw.json', 'r') as f:
        grants_raw = json.load(f)

    with open('../data/raw/institution_mapping.json', 'r') as f:
        institution_mapping = json.load(f)

    with open('../data/raw/agency_mapping.json', 'r') as f:
        agency_mapping = json.load(f)

    return researchers_raw, grants_raw, institution_mapping, agency_mapping


def build_researchers_json(researchers_raw):
    """
    Convert raw researchers to final format.
    Must have integer IDs matching array indices.
    """
    researchers = []
    for i, r in enumerate(researchers_raw):
        researcher = {
            'id': i,  # CRITICAL: must match array index
            'name': r['name'],
            'institution_id': r['institution_id'],
            'h_index': r['h_index'],
            'i10_index': r['i10_index'],
            'works_count': r['works_count'],
            'cited_by_count': r['cited_by_count'],
            'last_known_institution': r['last_known_institution'],
        }
        researchers.append(researcher)

    print(f"✓ Built {len(researchers)} researcher records")
    return researchers


def build_institutions_json(institution_mapping, researchers_raw):
    """
    Build institutions from unique affiliations in researcher data.
    """
    institutions = {}

    # Reverse mapping: internal ID -> external ID
    ext_to_int = {int(v): k for k, v in institution_mapping.items()}

    # Collect unique institution info
    for r in researchers_raw:
        if r['institution_id'] is not None:
            int_id = r['institution_id']
            if int_id not in institutions:
                institutions[int_id] = {
                    'id': int_id,
                    'name': r['last_known_institution'],
                    'openalex_id': ext_to_int.get(int_id),
                    'researcher_count': 0,
                    'total_works': 0,
                }
            institutions[int_id]['researcher_count'] += 1
            institutions[int_id]['total_works'] += r['works_count']

    # Convert to list sorted by ID
    institutions_list = sorted(institutions.values(), key=lambda x: x['id'])

    print(f"✓ Built {len(institutions_list)} institution records")
    return institutions_list


def build_agencies_json(agency_mapping):
    """
    Build agencies from mapping.
    """
    agencies = []
    for agency_name, agency_id in sorted(agency_mapping.items(), key=lambda x: x[1]):
        agency = {
            'id': agency_id,
            'name': agency_name,
            'grant_count': 0,
            'total_funding': 0,
        }
        agencies.append(agency)

    print(f"✓ Built {len(agencies)} agency records")
    return agencies


def build_grants_json(grants_raw, agencies):
    """
    Convert raw grants to final format with proper IDs.
    """
    grants = []
    agency_grant_counts = {a['id']: 0 for a in agencies}
    agency_funding = {a['id']: 0 for a in agencies}

    for i, g in enumerate(grants_raw):
        grant = {
            'id': i,  # CRITICAL: must match array index
            'title': g['title'],
            'agency_id': g['agency_id'],
            'amount': g['amount'],
            'start_date': g['start_date'],
            'end_date': g['end_date'],
        }
        grants.append(grant)

        # Update agency stats
        agency_grant_counts[g['agency_id']] += 1
        agency_funding[g['agency_id']] += g['amount']

    # Update agencies with counts
    for a in agencies:
        a['grant_count'] = agency_grant_counts[a['id']]
        a['total_funding'] = agency_funding[a['id']]

    print(f"✓ Built {len(grants)} grant records")
    return grants


def build_topics_json(researchers_raw, grants_raw):
    """
    Derive topics from researcher and grant keywords.
    In practice, you'd extract from abstracts using NLP.
    For now, creating synthetic topics.
    """
    topics_set = set()

    # Extract broad topics from grant titles
    keywords = ['AI', 'Machine Learning', 'Biology', 'Chemistry',
                'Physics', 'Engineering', 'Medicine', 'Climate',
                'Energy', 'Quantum']

    for g in grants_raw:
        title_upper = g['title'].upper()
        for kw in keywords:
            if kw.upper() in title_upper:
                topics_set.add(kw)

    # If we got few topics, add default ones
    if len(topics_set) < 5:
        topics_set.update(keywords[:5])

    topics = [
        {
            'id': i,
            'name': name,
            'researcher_count': 0,
            'grant_count': 0,
        }
        for i, name in enumerate(sorted(topics_set))
    ]

    print(f"✓ Built {len(topics)} topic records")
    return topics


def save_json_files(researchers, institutions, agencies, grants, topics):
    """
    Save all JSON files to data/raw/ with proper integer IDs.
    """
    output_dir = Path('../data/raw/')

    files = {
        'researchers.json': researchers,
        'institutions.json': institutions,
        'agencies.json': agencies,
        'grants.json': grants,
        'topics.json': topics,
    }

    for filename, data in files.items():
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {filepath}")


if __name__ == "__main__":
    print("🔄 Building JSON files with integer IDs...")

    researchers_raw, grants_raw, institution_mapping, agency_mapping = load_raw_data()

    researchers = build_researchers_json(researchers_raw)
    institutions = build_institutions_json(institution_mapping, researchers_raw)
    agencies = build_agencies_json(agency_mapping)
    grants = build_grants_json(grants_raw, agencies)
    topics = build_topics_json(researchers_raw, grants_raw)

    save_json_files(researchers, institutions, agencies, grants, topics)

    print("\n✓ All JSON files created successfully!")
    print(f"  - Researchers: {len(researchers)}")
    print(f"  - Institutions: {len(institutions)}")
    print(f"  - Agencies: {len(agencies)}")
    print(f"  - Grants: {len(grants)}")
    print(f"  - Topics: {len(topics)}")
