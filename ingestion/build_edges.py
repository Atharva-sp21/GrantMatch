import json
import csv
import random
from pathlib import Path

def load_json_files():
    """Load the built JSON files."""
    data_dir = Path('../data/raw/')

    with open(data_dir / 'researchers.json', 'r') as f:
        researchers = json.load(f)

    with open(data_dir / 'institutions.json', 'r') as f:
        institutions = json.load(f)

    with open(data_dir / 'agencies.json', 'r') as f:
        agencies = json.load(f)

    with open(data_dir / 'grants.json', 'r') as f:
        grants = json.load(f)

    with open(data_dir / 'topics.json', 'r') as f:
        topics = json.load(f)

    return researchers, institutions, agencies, grants, topics


def build_affiliated_edges(researchers):
    """
    Build AFFILIATED_WITH edges (researcher -> institution).
    Each researcher has one affiliation.
    """
    edges = []

    for r in researchers:
        if r['institution_id'] is not None:
            edges.append({
                'source_id': r['id'],
                'target_id': r['institution_id'],
            })

    print(f"✓ Built {len(edges)} AFFILIATED_WITH edges")
    return edges


def build_researches_edges(researchers, topics):
    """
    Build RESEARCHES edges (researcher -> topic).
    Each researcher researches 1-3 topics.
    """
    edges = []
    topic_ids = [t['id'] for t in topics]

    for r in researchers:
        # Probabilistic assignment: researchers typically focus on 1-3 areas
        num_topics = random.randint(1, min(3, len(topic_ids)))
        assigned_topics = random.sample(topic_ids, num_topics)

        for topic_id in assigned_topics:
            edges.append({
                'source_id': r['id'],
                'target_id': topic_id,
            })

    print(f"✓ Built {len(edges)} RESEARCHES edges")
    return edges


def build_received_past_edges(researchers, grants):
    """
    Build RECEIVED_PAST edges (researcher -> grant).
    Critical: Only researchers with h_index >= 5 receive grants.
    """
    edges = []

    for r in researchers:
        if r['h_index'] >= 5:
            # More prolific researchers get more grants
            num_grants = min(random.randint(0, r['h_index'] // 3), len(grants))
            assigned_grants = random.sample(range(len(grants)), num_grants)

            for grant_id in assigned_grants:
                edges.append({
                    'source_id': r['id'],
                    'target_id': grant_id,
                })

    print(f"✓ Built {len(edges)} RECEIVED_PAST edges")
    return edges


def build_funds_topic_edges(grants, topics):
    """
    Build FUNDS_TOPIC edges (grant -> topic).
    Each grant funds 1-2 topics.
    """
    edges = []
    topic_ids = [t['id'] for t in topics]

    for g in grants:
        num_topics = random.randint(1, min(2, len(topic_ids)))
        assigned_topics = random.sample(topic_ids, num_topics)

        for topic_id in assigned_topics:
            edges.append({
                'source_id': g['id'],
                'target_id': topic_id,
            })

    print(f"✓ Built {len(edges)} FUNDS_TOPIC edges")
    return edges


def build_provides_edges(agencies, grants):
    """
    Build PROVIDES edges (agency -> grant).
    Each grant is provided by exactly one agency.
    """
    edges = []

    for g in grants:
        edges.append({
            'source_id': g['agency_id'],
            'target_id': g['id'],
        })

    print(f"✓ Built {len(edges)} PROVIDES edges")
    return edges


def save_edge_csvs(affiliated, researches, received, funds, provides):
    """
    Save all edge files as CSVs to data/raw/.
    Format: source_id,target_id
    """
    output_dir = Path('../data/raw/')

    edge_files = {
        'affiliated.csv': affiliated,
        'researches.csv': researches,
        'received_past.csv': received,
        'funds_topic.csv': funds,
        'provides.csv': provides,
    }

    for filename, edges in edge_files.items():
        filepath = output_dir / filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['source_id', 'target_id'])
            writer.writeheader()
            writer.writerows(edges)
        print(f"✓ Saved {filepath} ({len(edges)} edges)")


if __name__ == "__main__":
    print("🔄 Building edge CSV files...")

    researchers, institutions, agencies, grants, topics = load_json_files()

    affiliated = build_affiliated_edges(researchers)
    researches = build_researches_edges(researchers, topics)
    received = build_received_past_edges(researchers, grants)
    funds = build_funds_topic_edges(grants, topics)
    provides = build_provides_edges(agencies, grants)

    save_edge_csvs(affiliated, researches, received, funds, provides)

    print("\n✓ All edge CSVs created successfully!")
    print("\n📊 Edge Summary:")
    print(f"  AFFILIATED_WITH (researcher -> institution): {len(affiliated)}")
    print(f"  RESEARCHES (researcher -> topic): {len(researches)}")
    print(f"  RECEIVED_PAST (researcher -> grant): {len(received)}")
    print(f"  FUNDS_TOPIC (grant -> topic): {len(funds)}")
    print(f"  PROVIDES (agency -> grant): {len(provides)}")
    print(f"  Total edges: {sum([len(affiliated), len(researches), len(received), len(funds), len(provides)])}")
