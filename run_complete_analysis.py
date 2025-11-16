"""
MASTER EXECUTION SCRIPT
Runs all 7 phases of marketplace fraud detection analysis sequentially
"""

import time
import sys

def print_header():
    """Print analysis header"""
    print("\n" + "="*80)
    print(" "*20 + "BIG DATA CRUNCHER")
    print(" "*10 + "European Secondhand Marketplace Analysis")
    print(" "*15 + "Fraud Detection & Pattern Mining Engine")
    print("="*80)
    print("\n  ANALYSIS PIPELINE:")
    print("    Phase 1: Data Generation & Profiling")
    print("    Phase 2: Univariate Analysis")
    print("    Phase 3: Multivariate Correlation Analysis")
    print("    Phase 4: Categorical & Segmented Analysis")
    print("    Phase 5: Anomaly & Fraud Detection")
    print("    Phase 6: Pattern Discovery & Clustering")
    print("    Phase 7: Time Series Analysis & Trends")
    print("\n" + "="*80)

def run_phase(phase_num, phase_name, module_name):
    """Run a single phase and track timing"""
    print(f"\n{'='*80}")
    print(f" EXECUTING PHASE {phase_num}: {phase_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Import and run the phase module
        exec(f"import {module_name}")
        exec(f"{module_name}.main()")

        elapsed = time.time() - start_time
        print(f"\n✓ Phase {phase_num} completed in {elapsed:.2f} seconds")
        return elapsed, True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Phase {phase_num} FAILED after {elapsed:.2f} seconds")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return elapsed, False

def main():
    """Execute complete analysis pipeline"""
    print_header()

    total_start = time.time()

    phases = [
        (1, "Data Generation & Profiling", "phase1_data_generation"),
        (2, "Univariate Analysis", "phase2_univariate_analysis"),
        (3, "Multivariate Correlation", "phase3_multivariate_analysis"),
        (4, "Categorical Analysis", "phase4_categorical_analysis"),
        (5, "Anomaly & Fraud Detection", "phase5_anomaly_detection"),
        (6, "Pattern Discovery", "phase6_pattern_discovery"),
        (7, "Time Series Analysis", "phase7_timeseries_analysis")
    ]

    results = []

    for phase_num, phase_name, module_name in phases:
        elapsed, success = run_phase(phase_num, phase_name, module_name)
        results.append({
            'phase': phase_num,
            'name': phase_name,
            'time': elapsed,
            'success': success
        })

        if not success:
            print(f"\n⚠ WARNING: Phase {phase_num} failed, continuing with next phase...")
            # Auto-continue on failures

    # Final summary
    total_time = time.time() - total_start

    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE")
    print("="*80)

    print("\n  EXECUTION SUMMARY:")
    print("  " + "-"*76)

    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"    Phase {result['phase']}: {result['name']:30s} - {status:10s} ({result['time']:.2f}s)")

    print("  " + "-"*76)
    print(f"    Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    successful = sum(1 for r in results if r['success'])
    print(f"\n  Phases completed successfully: {successful}/{len(results)}")

    if successful == len(results):
        print("\n  🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("\n  Generated files:")
        print("    - marketplace_data.csv")
        print("    - correlation_matrix.csv")
        print("    - univariate_analysis_results.json")
        print("    - multivariate_correlation_matrix.csv")
        print("    - anomaly_detection_report.csv")
        print("    - anomaly_scores.csv")
        print("    - cluster_analysis.csv")
        print("    - association_rules.csv")

    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
