#!/usr/bin/env python
"""Aggregate evaluation results from TOPO and APLS metrics."""

import json
import re
from pathlib import Path
import numpy as np
import sys

def parse_topo(file_path):
    """Parse TOPO metric file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Last line format: precision=X overall-recall=Y
    last_line = lines[-1].strip()

    # Parse precision and recall from last line
    precision_match = re.search(r'precision=([\d.]+)', last_line)
    recall_match = re.search(r'overall-recall=([\d.]+)', last_line)

    if not precision_match or not recall_match:
        raise ValueError(f'Could not parse TOPO metrics from: {last_line}')

    precision = float(precision_match.group(1))
    recall = float(recall_match.group(1))

    # Compute F1 score from precision and recall
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def parse_apls(file_path):
    """Parse APLS metric file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Format can be either:
    # 1. Simple: "0.663891 0.715446 0.689668"
    # 2. With line number: "     1→0.663891 0.715446 0.689668"
    if lines:
        first_line = lines[0].strip()

        # Check if it has the arrow separator
        if '→' in first_line:
            parts = first_line.split('→')
            if len(parts) > 1:
                scores = parts[1].split()
                if len(scores) >= 3:
                    return float(scores[2])  # Third score is the final APLS
        else:
            # Simple format - just space-separated numbers
            scores = first_line.split()
            if len(scores) >= 3:
                return float(scores[2])  # Third score is the final APLS

    return None


def main():
    OUTPUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else 'evaluation_results'
    SPLIT_FILE = sys.argv[2] if len(sys.argv) > 2 else 'data/Sat2Graph/data/data_split.json'

    # Load split information
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    # Try to load config from summary if it exists (from previous partial run)
    config = {}
    summary_file = Path(OUTPUT_ROOT) / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            existing = json.load(f)
            config = existing.get('config', {})

    # Collect metrics for each split
    results = {
        'train': {'topo': [], 'apls': []},
        'valid': {'topo': [], 'apls': []},
        'test': {'topo': [], 'apls': []}
    }

    for split_name, region_ids in splits.items():
        for region_id in region_ids:
            # TOPO metrics
            topo_file = Path(f'{OUTPUT_ROOT}/metrics/topo/topo_{region_id}.txt')
            if topo_file.exists():
                try:
                    metrics = parse_topo(topo_file)
                    results[split_name]['topo'].append(metrics)
                except Exception as e:
                    print(f'Warning: Failed to parse TOPO for region {region_id}: {e}')

            # APLS metrics
            apls_file = Path(f'{OUTPUT_ROOT}/metrics/apls/apls_{region_id}.txt')
            if apls_file.exists():
                try:
                    apls = parse_apls(apls_file)
                    if apls is not None:
                        results[split_name]['apls'].append(apls)
                except Exception as e:
                    print(f'Warning: Failed to parse APLS for region {region_id}: {e}')

    # Print summary
    print('')
    print('='*60)
    print('EVALUATION RESULTS SUMMARY')
    print('='*60)

    for split_name in ['train', 'valid', 'test']:
        print(f'\n{split_name.upper()} SET ({len(splits[split_name])} regions):')
        print('-'*60)

        # TOPO metrics
        topo_metrics = results[split_name]['topo']
        if topo_metrics:
            avg_precision = np.mean([m['precision'] for m in topo_metrics])
            avg_recall = np.mean([m['recall'] for m in topo_metrics])
            avg_f1 = np.mean([m['f1'] for m in topo_metrics])

            print(f'  TOPO Metrics (n={len(topo_metrics)}):')
            print(f'    Precision:     {avg_precision:.4f} ({avg_precision*100:.2f}%)')
            print(f'    Recall:        {avg_recall:.4f} ({avg_recall*100:.2f}%)')
            print(f'    F1 Score:      {avg_f1:.4f} ({avg_f1*100:.2f}%)')
        else:
            print(f'  TOPO Metrics: No data available')

        # APLS metrics
        apls_scores = results[split_name]['apls']
        if apls_scores:
            avg_apls = np.mean(apls_scores)
            print(f'  APLS Score (n={len(apls_scores)}):')
            print(f'    Average:       {avg_apls:.4f} ({avg_apls*100:.2f}%)')
        else:
            print(f'  APLS Score: No data available')

    print('')
    print('='*60)

    # Save detailed results to JSON
    output_data = {
        'config': config,  # Include config if it was loaded
        'results': {}
    }

    for split_name in ['train', 'valid', 'test']:
        topo_metrics = results[split_name]['topo']
        apls_scores = results[split_name]['apls']

        output_data['results'][split_name] = {
            'num_regions': len(splits[split_name]),
            'topo': {
                'num_evaluated': len(topo_metrics),
                'precision': float(np.mean([m['precision'] for m in topo_metrics])) if topo_metrics else None,
                'recall': float(np.mean([m['recall'] for m in topo_metrics])) if topo_metrics else None,
                'f1': float(np.mean([m['f1'] for m in topo_metrics])) if topo_metrics else None,
            },
            'apls': {
                'num_evaluated': len(apls_scores),
                'score': float(np.mean(apls_scores)) if apls_scores else None,
            }
        }

    summary_file = Path(OUTPUT_ROOT) / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'Detailed results saved to {summary_file}')
    print('')


if __name__ == '__main__':
    main()
