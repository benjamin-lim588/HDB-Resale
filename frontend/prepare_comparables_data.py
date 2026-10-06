"""Export comparable-sale fields for the frontend, without changing the source."""
import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    'dates': 'transaction_date', 'towns': 'town', 'streets': 'street_name',
    'blocks': 'block', 'flatTypes': 'flat_type', 'flatModels': 'flat_model',
    'storeys': 'storey_range',
}
NUMERIC = ['floor_area_sqm', 'remaining_lease_months', 'storey_mid',
           'resale_price', 'price_per_sqm']


def prepare(source, destination):
    records = []
    duplicates = 0
    seen = set()
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = set(CATEGORIES.values()) | set(NUMERIC)
        missing -= set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            try:
                labels = [row[field].strip() for field in CATEGORIES.values()]
                labels[0] = date.fromisoformat(labels[0][:10]).isoformat()
                if not all(labels):
                    raise ValueError('missing category')
                values = [float(row[field]) for field in NUMERIC]
                if not all(math.isfinite(value) and value > 0 for value in values):
                    raise ValueError('invalid numeric value')
                if values[1] > 1188 or values[2] > 60:
                    raise ValueError('lease or storey exceeds supported HDB range')
                if not math.isclose(values[3] / values[0], values[4], rel_tol=1e-6):
                    raise ValueError('price per sqm does not match resale price / area')
                record = tuple(labels + values)
                # Identical exported rows provide no extra comparable evidence.
                # Deduplication is confined to this export; the input is untouched.
                if record in seen:
                    duplicates += 1
                    continue
                seen.add(record)
                records.append(record)
            except (ValueError, KeyError) as exc:
                raise ValueError(f'Invalid comparable data on CSV line {line}: {exc}') from exc
    if not records:
        raise ValueError('The cleaned CSV has no comparable transactions.')
    dictionaries = {key: sorted({row[i] for row in records})
                    for i, key in enumerate(CATEGORIES)}
    lookups = [{label: i for i, label in enumerate(values)}
               for values in dictionaries.values()]
    columns = [[] for _ in range(len(CATEGORIES) + len(NUMERIC))]
    for row in records:
        for i, value in enumerate(row):
            columns[i].append(lookups[i][value] if i < len(CATEGORIES) else value)
    payload = {
        'version': 1, 'source': 'data/processed/hdb_clean.csv',
        'count': len(records), 'duplicatesExcluded': duplicates,
        'fields': list(CATEGORIES.values()) + NUMERIC,
        **dictionaries, 'columns': columns,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(destination)
    print(f'Prepared {len(records):,} comparable transactions; {duplicates:,} identical exported rows excluded.')
    print(f'Wrote {destination} ({destination.stat().st_size / 1_000_000:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed/hdb_clean.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'frontend/data/comparables.json')
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (ValueError, OSError) as exc:
        print(f'Cannot prepare comparables: {exc}', file=sys.stderr)
        sys.exit(1)
