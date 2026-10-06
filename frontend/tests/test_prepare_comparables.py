"""Export validation and source preservation using a small independent fixture."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_comparables', Path(__file__).parents[1] / 'prepare_comparables_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def fixture(self, folder, rows):
        source = Path(folder) / 'source.csv'
        fields = list(module.CATEGORIES.values()) + module.NUMERIC
        with source.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return source

    def row(self, **changes):
        return {'transaction_date': '2026-09-01', 'town': 'TAMPINES', 'street_name': 'TAMPINES ST 11',
                'block': '001A', 'flat_type': '4 ROOM', 'flat_model': 'Model A', 'storey_range': '07 TO 09',
                'floor_area_sqm': 100, 'remaining_lease_months': 781, 'storey_mid': 8,
                'resale_price': 600000, 'price_per_sqm': 6000, **changes}

    def test_compact_export_deduplicates_only_locally_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.fixture(folder, [self.row(), self.row(), self.row(resale_price=650000, price_per_sqm=6500)])
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            output = Path(folder) / 'export.json'
            module.prepare(source, output)
            data = json.loads(output.read_text())
            self.assertEqual(data['count'], 2)
            self.assertEqual(data['duplicatesExcluded'], 1)
            self.assertEqual(data['blocks'], ['001A'])
            self.assertEqual(len(data['columns']), 12)
            self.assertEqual(data['columns'][10], [600000, 650000])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_invalid_values_fail_without_overwriting_existing_export(self):
        for changes in [{'transaction_date': '2026-02-30'}, {'floor_area_sqm': 0},
                        {'remaining_lease_months': 1200}, {'price_per_sqm': 6500}, {'resale_price': float('nan')}]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as folder:
                source = self.fixture(folder, [self.row(**changes)])
                output = Path(folder) / 'export.json'
                output.write_text('original')
                with self.assertRaises(ValueError):
                    module.prepare(source, output)
                self.assertEqual(output.read_text(), 'original')

    def test_empty_and_missing_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.fixture(folder, [])
            with self.assertRaisesRegex(ValueError, 'no comparable'):
                module.prepare(source, Path(folder) / 'export.json')
            source.write_text('town\nTAMPINES\n')
            with self.assertRaisesRegex(ValueError, 'Missing required columns'):
                module.prepare(source, Path(folder) / 'export.json')


if __name__ == '__main__':
    unittest.main()
