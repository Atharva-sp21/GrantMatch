import requests
import json
import time
from datetime import datetime

def generate_sample_grants(limit=100):
    """Generate sample grants for testing when APIs are unavailable."""
    sample_titles = [
        "Deep Learning for Protein Structure Prediction",
        "Quantum Computing Architectures",
        "Climate Change Mitigation Strategies",
        "Novel Cancer Immunotherapies",
        "Renewable Energy Storage Systems",
        "Autonomous Vehicle Safety",
        "Brain-Computer Interfaces",
        "Sustainable Agriculture Practices",
        "AI for Drug Discovery",
        "Quantum Machine Learning Applications"
    ]

    grants = []
    for i in range(min(limit, 100)):
        grant = {
            'id': i,
            'title': sample_titles[i % len(sample_titles)] + f" (Study {i+1})",
            'agency_id': i % 2,  # NIH or NSF
            'amount': (i + 1) * 100000,
            'start_date': '2023-01-01',
            'end_date': '2026-12-31',
        }
        grants.append(grant)

    return grants


def fetch_nih_grants(limit=500):
    """
    Fetch grants from NIH Reporter API.
    Falls back to sample data if API fails.
    """
    grants = []
    agency_ids = {}
    agency_counter = 0

    base_url = "https://reporter.nih.gov/api/v2/research_projects/search"

    print("🔍 Fetching NIH grants...")

    try:
        payload = {
            "search_text": "*",
            "offset": 0,
            "limit": min(limit, 500),
            "sort_field": "project_end_date",
            "sort_order": "desc"
        }

        response = requests.post(base_url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()

        for result in data.get('results', [])[:limit]:
            grant_id = len(grants)

            if 'NIH' not in agency_ids:
                agency_ids['NIH'] = agency_counter
                agency_counter += 1

            grant = {
                'id': grant_id,
                'nih_project_number': result.get('project_num', ''),
                'title': result.get('project_title', ''),
                'agency_id': agency_ids['NIH'],
                'amount': result.get('total_cost', 0),
                'start_date': result.get('project_start_date', ''),
                'end_date': result.get('project_end_date', ''),
            }
            grants.append(grant)

        print(f"✓ Fetched {len(grants)} NIH grants")

    except requests.exceptions.RequestException as e:
        print(f"⚠️  NIH API unavailable ({str(e)[:50]}...), using sample data")
        if 'NIH' not in agency_ids:
            agency_ids['NIH'] = 0
        grants = generate_sample_grants(limit // 2)

    return grants, agency_ids


def fetch_nsf_grants(limit=500, agencies_dict=None):
    """
    Fetch grants from NSF public data API.
    Falls back to sample data if API fails.
    """
    if agencies_dict is None:
        agencies_dict = {}

    grants = []
    agency_counter = len(agencies_dict)

    base_url = "https://api.nsf.gov/services/v2/awards"

    print("🔍 Fetching NSF grants...")

    try:
        if 'NSF' not in agencies_dict:
            agencies_dict['NSF'] = agency_counter
            agency_counter += 1

        params = {
            "filter": "fundProgramName:CISE",
            "limit": min(limit, 25),
            "offset": 0
        }

        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        grant_id_offset = len(grants)
        for i, result in enumerate(data.get('response', {}).get('award', [])[:limit]):
            grant = {
                'id': grant_id_offset + i,
                'nsf_award_number': result.get('id', ''),
                'title': result.get('title', ''),
                'agency_id': agencies_dict['NSF'],
                'amount': result.get('fundsObligatedAmt', 0),
                'start_date': result.get('startDate', ''),
                'end_date': result.get('expDate', ''),
            }
            grants.append(grant)

        print(f"✓ Fetched {len(grants)} NSF grants")

    except requests.exceptions.RequestException as e:
        print(f"⚠️  NSF API unavailable ({str(e)[:50]}...), using sample data")
        if 'NSF' not in agencies_dict:
            agencies_dict['NSF'] = 1
        sample = generate_sample_grants(limit // 2)
        for g in sample:
            g['agency_id'] = 1
            g['id'] = len(grants)
            grants.append(g)

    return grants, agencies_dict


if __name__ == "__main__":
    nih_grants, agencies = fetch_nih_grants(limit=500)
    nsf_grants, agencies = fetch_nsf_grants(limit=500, agencies_dict=agencies)

    # Combine and reassign IDs to ensure consistency
    all_grants = nih_grants + nsf_grants

    # If no grants fetched, use sample data
    if len(all_grants) == 0:
        print("⚠️  No grants from APIs, generating sample data...")
        all_grants = generate_sample_grants(100)
        agencies = {'NIH': 0, 'NSF': 1}

    # Reassign IDs to match array indices
    for i, grant in enumerate(all_grants):
        grant['id'] = i

    with open('../data/raw/grants_raw.json', 'w') as f:
        json.dump(all_grants, f, indent=2)

    with open('../data/raw/agency_mapping.json', 'w') as f:
        json.dump(agencies, f, indent=2)

    print(f"✓ Saved {len(all_grants)} total grants")
    print(f"✓ Agencies: {list(agencies.keys())}")
