"""
Test Script for Mesh Loading with Multiple Formats
===================================================
Tests loading mesh files in different formats (GIFTI, OBJ, MZ3, PLY)
and validates that vertices and faces are correctly extracted.

Output directory: HarrisLabPlotting/test_results/mesh_test
"""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
# Parent of HarrisLabPlotting (Research directory) for "from HarrisLabPlotting import ..."
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# HarrisLabPlotting directory itself for direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh import load_mesh_file


def test_mesh_loading():
    """
    Test loading mesh files in all supported formats.
    Validates that each format returns proper vertices and faces arrays.
    """
    # Test mesh files
    test_dir = Path(__file__).parent
    output_dir = Path(r"C:\Users\Azad Azargushasb\Research\HarrisLabPlotting\test_results\mesh_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = {
        'mz3': test_dir / "brain_filled_2.mz3",
        'obj': test_dir / "brain_filled_2.obj",
        'ply': test_dir / "brain_filled_2.ply",
        'gii': test_dir / "brain_filled_2.gii",
    }

    print("=" * 80)
    print("MESH LOADING TEST - MULTIPLE FORMATS")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nSupported formats: .gii, .obj, .mz3, .ply")
    print("=" * 80)

    results = {}
    all_passed = True

    for format_name, mesh_path in mesh_files.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {format_name.upper()} format: {mesh_path.name}")
        print(f"{'=' * 60}")

        if not mesh_path.exists():
            print(f"  ERROR: File not found: {mesh_path}")
            results[format_name] = {
                'status': 'FAILED',
                'error': 'File not found',
                'vertices': None,
                'faces': None
            }
            all_passed = False
            continue

        try:
            vertices, faces = load_mesh_file(mesh_path)

            # Validate results
            is_valid = True
            errors = []

            # Check vertices
            if vertices is None:
                errors.append("Vertices is None")
                is_valid = False
            elif not isinstance(vertices, np.ndarray):
                errors.append(f"Vertices is not ndarray: {type(vertices)}")
                is_valid = False
            elif len(vertices.shape) != 2 or vertices.shape[1] != 3:
                errors.append(f"Invalid vertices shape: {vertices.shape}")
                is_valid = False

            # Check faces
            if faces is None:
                errors.append("Faces is None")
                is_valid = False
            elif not isinstance(faces, np.ndarray):
                errors.append(f"Faces is not ndarray: {type(faces)}")
                is_valid = False
            elif len(faces.shape) != 2 or faces.shape[1] != 3:
                errors.append(f"Invalid faces shape: {faces.shape}")
                is_valid = False

            # Check face indices are valid
            if is_valid and faces is not None and vertices is not None:
                max_idx = faces.max()
                if max_idx >= len(vertices):
                    errors.append(f"Face index {max_idx} exceeds vertex count {len(vertices)}")
                    is_valid = False
                if faces.min() < 0:
                    errors.append(f"Negative face index found: {faces.min()}")
                    is_valid = False

            if is_valid:
                results[format_name] = {
                    'status': 'PASSED',
                    'vertices': vertices,
                    'faces': faces,
                    'n_vertices': vertices.shape[0],
                    'n_faces': faces.shape[0],
                    'vertex_bounds': {
                        'min': vertices.min(axis=0).tolist(),
                        'max': vertices.max(axis=0).tolist()
                    }
                }
                print(f"\n  RESULT: PASSED")
                print(f"  Vertices: {vertices.shape[0]} (shape: {vertices.shape})")
                print(f"  Faces: {faces.shape[0]} (shape: {faces.shape})")
                print(f"  Vertex dtype: {vertices.dtype}")
                print(f"  Face dtype: {faces.dtype}")
                print(f"  X range: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
                print(f"  Y range: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
                print(f"  Z range: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
            else:
                results[format_name] = {
                    'status': 'FAILED',
                    'errors': errors,
                    'vertices': vertices,
                    'faces': faces
                }
                print(f"\n  RESULT: FAILED")
                for err in errors:
                    print(f"    - {err}")
                all_passed = False

        except Exception as e:
            results[format_name] = {
                'status': 'ERROR',
                'error': str(e),
                'vertices': None,
                'faces': None
            }
            print(f"\n  RESULT: ERROR")
            print(f"    Exception: {type(e).__name__}: {e}")
            all_passed = False

    # Test unsupported format
    print(f"\n{'=' * 60}")
    print("Testing unsupported format detection")
    print(f"{'=' * 60}")
    try:
        # Create a dummy file to test unsupported format
        dummy_file = output_dir / "test.xyz"
        dummy_file.write_text("dummy content")
        load_mesh_file(dummy_file)
        print("  RESULT: FAILED - Should have raised ValueError")
        results['unsupported_format_detection'] = {'status': 'FAILED'}
        all_passed = False
    except ValueError as e:
        if "Unsupported mesh file format" in str(e):
            print(f"  RESULT: PASSED - Correctly detected unsupported format")
            print(f"    Error message: {e}")
            results['unsupported_format_detection'] = {'status': 'PASSED'}
        else:
            print(f"  RESULT: FAILED - Wrong error message")
            results['unsupported_format_detection'] = {'status': 'FAILED'}
            all_passed = False
    except Exception as e:
        print(f"  RESULT: FAILED - Wrong exception type: {type(e).__name__}")
        results['unsupported_format_detection'] = {'status': 'FAILED'}
        all_passed = False
    finally:
        if dummy_file.exists():
            dummy_file.unlink()

    # Compare mesh statistics across formats
    print(f"\n{'=' * 80}")
    print("MESH COMPARISON ACROSS FORMATS")
    print(f"{'=' * 80}")

    passed_formats = [fmt for fmt, res in results.items()
                      if res.get('status') == 'PASSED' and fmt != 'unsupported_format_detection']

    if len(passed_formats) >= 2:
        print(f"\nComparing meshes loaded from: {', '.join(passed_formats)}")
        print("-" * 60)

        # Get reference values from first format
        ref_format = passed_formats[0]
        ref_n_vertices = results[ref_format]['n_vertices']
        ref_n_faces = results[ref_format]['n_faces']

        print(f"{'Format':<10} {'Vertices':>12} {'Faces':>12} {'Match':>10}")
        print("-" * 60)

        for fmt in passed_formats:
            n_vert = results[fmt]['n_vertices']
            n_face = results[fmt]['n_faces']
            match = "YES" if (n_vert == ref_n_vertices and n_face == ref_n_faces) else "NO"
            print(f"{fmt:<10} {n_vert:>12} {n_face:>12} {match:>10}")

    # Save results summary
    summary_file = output_dir / "test_results_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("MESH LOADING TEST RESULTS\n")
        f.write("=" * 60 + "\n\n")

        for fmt, res in results.items():
            f.write(f"{fmt.upper()}: {res['status']}\n")
            if res['status'] == 'PASSED' and fmt != 'unsupported_format_detection':
                f.write(f"  Vertices: {res['n_vertices']}\n")
                f.write(f"  Faces: {res['n_faces']}\n")
            elif res.get('error'):
                f.write(f"  Error: {res['error']}\n")
            elif res.get('errors'):
                for err in res['errors']:
                    f.write(f"  Error: {err}\n")
            f.write("\n")

        f.write(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}\n")

    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {summary_file}")
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    # Per-format summary
    print("\nPer-format results:")
    for fmt, res in results.items():
        status_symbol = "[PASS]" if res['status'] == 'PASSED' else "[FAIL]"
        print(f"  {status_symbol} {fmt}")

    return results, all_passed


def run_visualization_pipeline_test():
    """
    Test the full visualization pipeline with k=5 using all mesh formats.

    This runs the complete brain modularity visualization pipeline for each
    mesh format (GII, OBJ, MZ3, PLY) and saves HTML outputs.
    """
    from HarrisLabPlotting import run_enhanced_visualization_pipeline

    # Configuration
    test_dir = Path(__file__).parent
    output_dir = Path(r"C:\Users\Azad Azargushasb\Research\HarrisLabPlotting\test_results\mesh_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mesh files to test
    mesh_files = {
        'mz3': test_dir / "brain_filled_2.mz3",
        'obj': test_dir / "brain_filled_2.obj",
        'ply': test_dir / "brain_filled_2.ply",
        'gii': test_dir / "brain_filled_2.gii",
    }

    # k=5 configuration
    k_value = 5
    state_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    # Data paths
    base_matrix_dir = Path(r"G:\My Drive\gmvae_stim_experiments_v2\experiment_k5_15_combined_20251114_163203\omst_filter_matrices\stim_matrices")
    netneurotools_dir = Path(r"G:\My Drive\gmvae_stim_experiments_v2\experiment_k5_15_combined_20251114_163203\modularity_significance_analysis_omst_pos")
    roi_coords_file = r"G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv"

    print("\n" + "=" * 80)
    print("ENHANCED BRAIN MODULARITY VISUALIZATION PIPELINE TEST")
    print("Processing k=5 with all mesh formats")
    print("=" * 80)
    print("\nFeatures:")
    print("  - Interactive camera controls with preset views")
    print("  - Optimized node size differences (2x multiplier)")
    print("  - Enhanced edge thickness (1.0 to 6.0 range)")
    print("  - Proper Q and Z value loading from CSV")
    print("  - Thicker borders (6px) for clear visibility")
    print("  - State mapping for k=5")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)

    # Check matrix file
    matrix_path = base_matrix_dir / f"k{k_value}_strategy_A_positive.npy"
    if not matrix_path.exists():
        print(f"\nERROR: Matrix file not found: {matrix_path}")
        print("Cannot run visualization pipeline test - data files not available")
        return None

    print(f"\nMatrix file: {matrix_path}")
    print(f"State mapping: {state_mapping}")
    print(f"Active clusters: {len(state_mapping)}/{k_value}")

    all_viz_results = {}
    successful_formats = []
    failed_formats = []

    # Process each mesh format
    for format_name, mesh_path in mesh_files.items():
        print(f"\n{'=' * 60}")
        print(f"PROCESSING k={k_value} with {format_name.upper()} mesh")
        print(f"{'=' * 60}")

        if not mesh_path.exists():
            print(f"  ERROR: Mesh file not found: {mesh_path}")
            failed_formats.append(format_name)
            continue

        print(f"  Mesh file: {mesh_path}")

        # Create format-specific output directory
        viz_output_dir = output_dir / f"k{k_value}_{format_name}"
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {viz_output_dir}")

        try:
            results = run_enhanced_visualization_pipeline(
                matrix_path=str(matrix_path),
                netneurotools_results_dir=str(netneurotools_dir),
                mesh_file=str(mesh_path),
                roi_coords_file=roi_coords_file,
                output_dir=str(viz_output_dir),
                k_value=k_value,
                visualization_types=['all', 'intra', 'inter', 'nodes_only', 'significant_only'],
                node_sizing_modes=['pc', 'zscore', 'both'],
                use_thresholding=True,
                n_top_edges=64,
                base_node_size=12,
                max_node_multiplier=2.0,
                show_labels=True,
                show_significance=True,
                state_mapping=state_mapping,
                camera_views=['oblique'],
                enable_interactive_panel=True,
                save_multiple_views=False
            )

            if results:
                all_viz_results[format_name] = results
                successful_formats.append(format_name)
                print(f"\n  SUCCESS: Created {len(results)} visualizations for {format_name.upper()}")

                # Print state summary
                state_info = {}
                for _, data in results.items():
                    state_label = data['state_label']
                    if state_label not in state_info:
                        state_info[state_label] = {
                            'Q': data['Q_total'],
                            'Z': data['Q_z_score'],
                            'count': 0
                        }
                    state_info[state_label]['count'] += 1

                print(f"\n  States summary for {format_name.upper()}:")
                for state_label in sorted(state_info.keys()):
                    info = state_info[state_label]
                    print(f"    State {state_label}: Q={info['Q']:.3f}, Z={info['Z']:.2f}, Visualizations={info['count']}")
            else:
                print(f"\n  WARNING: No visualizations created for {format_name.upper()}")
                failed_formats.append(format_name)

        except Exception as e:
            print(f"\n  ERROR processing {format_name.upper()}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed_formats.append(format_name)

    # Final summary
    print(f"\n{'=' * 80}")
    print("VISUALIZATION PIPELINE TEST - FINAL SUMMARY")
    print(f"{'=' * 80}")

    print(f"\nk value tested: k={k_value}")
    print(f"Mesh formats tested: {len(mesh_files)}")
    print(f"Successful: {len(successful_formats)}")
    print(f"Failed: {len(failed_formats)}")

    if successful_formats:
        print(f"\nSuccessful formats:")
        total_visualizations = 0
        for fmt in successful_formats:
            num_viz = len(all_viz_results[fmt])
            total_visualizations += num_viz
            print(f"  [PASS] {fmt.upper()}: {num_viz} visualizations")

        print(f"\nTotal visualizations created: {total_visualizations}")

    if failed_formats:
        print(f"\nFailed formats:")
        for fmt in failed_formats:
            print(f"  [FAIL] {fmt.upper()}")

    print(f"\nOutput location: {output_dir}")
    print(f"\nHTML files saved in subdirectories:")
    for fmt in successful_formats:
        print(f"  - k{k_value}_{fmt}/")

    # Save summary to file
    summary_file = output_dir / "pipeline_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("VISUALIZATION PIPELINE TEST SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"k value: {k_value}\n")
        f.write(f"State mapping: {state_mapping}\n\n")
        f.write(f"Successful formats: {successful_formats}\n")
        f.write(f"Failed formats: {failed_formats}\n\n")

        if all_viz_results:
            f.write("Visualization counts per format:\n")
            for fmt, results in all_viz_results.items():
                f.write(f"  {fmt.upper()}: {len(results)} visualizations\n")

    print(f"\nSummary saved to: {summary_file}")

    return all_viz_results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MESH TEST SUITE")
    print("=" * 80)

    # Test 1: Mesh loading for all formats
    print("\n" + "=" * 80)
    print("TEST 1: MESH LOADING")
    print("=" * 80)
    results, all_passed = test_mesh_loading()

    # Test 2: Full visualization pipeline with k=5
    print("\n" + "=" * 80)
    print("TEST 2: VISUALIZATION PIPELINE (k=5)")
    print("=" * 80)
    viz_results = run_visualization_pipeline_test()

    # Final summary
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

    print("\nMesh Loading Test: " + ("PASSED" if all_passed else "FAILED"))
    print("Visualization Pipeline Test: " + ("COMPLETED" if viz_results else "SKIPPED/FAILED"))

    if viz_results:
        print(f"\nVisualization outputs saved to:")
        print(r"  C:\Users\Azad Azargushasb\Research\HarrisLabPlotting\test_results\mesh_test")

    print("\n" + "=" * 80)
