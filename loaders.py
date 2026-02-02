"""
Data Loaders
=============
Classes for loading modularity analysis results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union


class NetNeurotoolsModularityLoader:
    """Load and process netneurotools modularity results."""

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.comprehensive_dir = self.base_dir / 'comprehensive_analysis'
        self.csv_dir = self.base_dir / 'csv_files'
        self.results_dir = self.base_dir / 'results'
        self.plots_dir = self.base_dir / 'plots'
        self.reports_dir = self.base_dir / 'reports'

        if not self.base_dir.exists():
            raise ValueError(f"Base directory not found: {self.base_dir}")

        print(f"NetNeurotools Loader initialized with base: {self.base_dir}")

    def load_summary_statistics(self) -> pd.DataFrame:
        """Load summary statistics CSV with Q and Z values for all states."""
        summary_file = self.csv_dir / 'all_k_summary_statistics.csv'
        if summary_file.exists():
            print(f"   Loading summary statistics from: {summary_file}")
            return pd.read_csv(summary_file)
        else:
            print(f"   Warning: Summary statistics not found at {summary_file}")
        return None

    def load_comprehensive_results(self, k: int) -> Dict:
        """Load comprehensive analysis results with proper Q and Z values from CSV."""
        k_dir = self.comprehensive_dir / f'k{k}_detailed'
        if not k_dir.exists():
            raise ValueError(f"Results for k={k} not found at {k_dir}")

        results = {'k': k, 'states': []}

        # First try to load summary statistics from CSV for accurate Q and Z values
        summary_df = self.load_summary_statistics()
        Q_values = {}
        Q_z_scores = {}
        module_significance = {}

        if summary_df is not None:
            # Filter for this k value
            k_data = summary_df[summary_df['k'] == k]
            for _, row in k_data.iterrows():
                state_idx = int(row['state'])
                Q_values[state_idx] = float(row['Q_total'])
                Q_z_scores[state_idx] = float(row['Q_z_score'])
                print(f"   Loaded from CSV - State {state_idx}: Q={Q_values[state_idx]:.3f}, Z={Q_z_scores[state_idx]:.2f}")

        # Fallback to loading from summary.npz if CSV not available
        if not Q_values:
            summary_file = self.comprehensive_dir / f'k{k}_summary.npz'
            if summary_file.exists():
                print(f"   Loading summary from: {summary_file}")
                summary_data = np.load(summary_file, allow_pickle=True)

                if 'Q_values' in summary_data:
                    Q_array = summary_data['Q_values']
                    for i, q in enumerate(Q_array):
                        Q_values[i] = float(q)

                if 'Q_z_scores' in summary_data:
                    Z_array = summary_data['Q_z_scores']
                    for i, z in enumerate(Z_array):
                        Q_z_scores[i] = float(z)

                if 'module_significance' in summary_data:
                    module_significance = summary_data['module_significance']

        # Load each state's detailed data
        for state_file in sorted(k_dir.glob('state*_arrays.npz')):
            state_idx = int(state_file.stem.split('state')[1].split('_')[0])

            # Load arrays
            arrays = np.load(state_file, allow_pickle=True)

            # Load node metrics CSV
            csv_file = k_dir / f'state{state_idx}_node_metrics.csv'
            if csv_file.exists():
                metrics_df = pd.read_csv(csv_file)
            else:
                print(f"   Warning: Missing metrics CSV for state {state_idx}")
                continue

            # Get module assignments
            if 'module_assignments' in arrays:
                consensus = arrays['module_assignments']
            else:
                consensus = metrics_df['module'].values

            # Get module significance for this state
            if isinstance(module_significance, np.ndarray) and state_idx < len(module_significance):
                state_module_sig = module_significance[state_idx]
            else:
                state_module_sig = np.ones(len(np.unique(consensus)), dtype=bool)

            state_data = {
                'state_idx': state_idx,
                'consensus': consensus,
                'Q_total': Q_values.get(state_idx, 0.0),
                'Q_z_score': Q_z_scores.get(state_idx, 0.0),
                'module_significance': state_module_sig,
                'Q_per_module': arrays.get('Q_per_module', np.zeros(len(np.unique(consensus)))),
                'modules_df': metrics_df,
                'participation_coef': metrics_df['participation_coef'].values if 'participation_coef' in metrics_df else np.zeros(len(metrics_df)),
                'within_module_zscore': metrics_df['within_module_zscore'].values if 'within_module_zscore' in metrics_df else np.zeros(len(metrics_df))
            }

            results['states'].append(state_data)

            if state_data['Q_total'] != 0:
                print(f"   State {state_idx}: Q={state_data['Q_total']:.3f}, Z={state_data['Q_z_score']:.2f}")

        return results
