"""Build compact, exact-median dashboard data without altering the pipeline."""
import argparse
import csv
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {'transaction_date', 'transaction_year', 'transaction_month', 'town',
          'flat_type', 'storey_range', 'resale_price', 'price_per_sqm'}


def prepare(source, destination):
    records = []
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            try:
                day = date.fromisoformat(row['transaction_date'][:10])
                if (day.year, day.month) != (int(row['transaction_year']), int(row['transaction_month'])):
                    raise ValueError('inconsistent date fields')
                price, psm = float(row['resale_price']), float(row['price_per_sqm'])
                labels = [row[c].strip() for c in ('town', 'flat_type', 'storey_range')]
                if not all(labels) or not all(math.isfinite(v) and v > 0 for v in (price, psm)):
                    raise ValueError('missing category or invalid price')
                records.append([day.strftime('%Y-%m'), *labels, price, psm])
            except (ValueError, KeyError) as exc:
                raise ValueError(f'Invalid data on CSV line {line}: {exc}') from exc
    if not records:
        raise ValueError('The cleaned CSV has no transactions.')
    dictionaries = [sorted({r[i] for r in records}) for i in range(4)]
    lookup = [{v: i for i, v in enumerate(values)} for values in dictionaries]
    columns = [[] for _ in range(6)]
    for record in records:
        for i in range(6):
            columns[i].append(lookup[i][record[i]] if i < 4 else record[i])
    payload = {
        'version': 1, 'generatedAt': datetime.now(timezone.utc).isoformat(),
        'source': 'data/processed/hdb_clean.csv', 'count': len(records),
        'months': dictionaries[0], 'towns': dictionaries[1],
        'flatTypes': dictionaries[2], 'storeys': dictionaries[3],
        'columns': columns,
        'priceOrder': sorted(range(len(records)), key=lambda i: columns[4][i]),
        'psmOrder': sorted(range(len(records)), key=lambda i: columns[5][i]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(destination)
    print(f'Prepared {len(records):,} transactions ({dictionaries[0][0]} to {dictionaries[0][-1]})')
    print(f'Wrote {destination} ({destination.stat().st_size / 1_000_000:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed/hdb_clean.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'frontend/data/transactions.json')
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (OSError, ValueError) as exc:
        print(f'Cannot prepare dashboard data: {exc}', file=sys.stderr)
        sys.exit(1)
