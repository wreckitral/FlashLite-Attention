"""
Analyze All Results - Generate LaTeX code AND CSV files for dashboard
Reads data from:
  - results/metrics/performance_benchmark.csv (Tables 1 & 5)
  - results/profiles/*.ncu-rep (Tables 2, 3, 4) - exported as CSV

Generates:
  - LaTeX table code (printed to console)
  - CSV files for Streamlit dashboard

Usage:
  python analyze_all_results.py                    # Default: output to dashboard_data/
  python analyze_all_results.py --output my_data   # Output to my_data/
  python analyze_all_results.py -o presentation    # Output to presentation/
"""

import pandas as pd
from pathlib import Path
import subprocess
import re
import json
import argparse


class ResultsAnalyzer:
    def __init__(self, output_dir="dashboard_data"):
        self.results_dir = Path("results")
        self.metrics_dir = self.results_dir / "metrics"
        self.profiles_dir = self.results_dir / "profiles"
        self.dashboard_dir = Path(output_dir)
        self.dashboard_dir.mkdir(exist_ok=True)
        self.output_dir_name = output_dir

    def load_performance_data(self):
        """Load performance benchmark data (Tables 1 & 5)"""
        csv_path = self.metrics_dir / "performance_benchmark.csv"

        if not csv_path.exists():
            print(f"❌ Performance benchmark not found: {csv_path}")
            print("   Run: python benchmarks/benchmark_all_kernels.py")
            return None

        df = pd.read_csv(csv_path)
        return df

    def export_ncu_to_csv(self, ncu_file):
        """Export NCU report to CSV format"""
        csv_file = ncu_file.with_suffix(".csv")

        if csv_file.exists():
            print(f"  ✓ CSV already exists: {csv_file.name}")
            return csv_file

        try:
            # Export using ncu command with proper flags
            print(f"  Exporting {ncu_file.name} to CSV...")
            cmd = ["ncu", "--import", str(ncu_file), "--csv", "--page", "raw"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Write output to CSV file
            csv_file.write_text(result.stdout)
            print(f"  ✓ Exported: {csv_file.name}")
            return csv_file

        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Failed to export {ncu_file.name}")
            print(f"     Error: {e.stderr if e.stderr else 'Unknown error'}")
            return None
        except FileNotFoundError:
            print(f"  ⚠️  'ncu' command not found. Install NSight Compute CLI tools.")
            return None

    def parse_ncu_csv(self, csv_file, metrics):
        """Parse NCU CSV and extract specific metrics - NCU format has metrics as columns"""
        if not csv_file or not csv_file.exists():
            return None

        try:
            # Read CSV - skip the units row (row 1)
            df = pd.read_csv(csv_file, skiprows=[1])

            # NCU format: metrics are column names, data in rows
            # We want to aggregate across all kernels (rows)
            results = {}

            for metric in metrics:
                # Look for column that contains this metric name
                matching_cols = [col for col in df.columns if metric in col]

                if matching_cols:
                    col_name = matching_cols[0]

                    # Get all non-null values from this column
                    values = df[col_name].dropna()

                    if len(values) > 0:
                        # Convert to numeric if possible
                        try:
                            numeric_values = pd.to_numeric(
                                values, errors="coerce"
                            ).dropna()
                            if len(numeric_values) > 0:
                                # Take mean across all kernels
                                results[metric] = float(numeric_values.mean())
                            else:
                                print(f"  Warning: No numeric values for '{metric}'")
                        except Exception as e:
                            print(
                                f"  Warning: Could not convert '{metric}' to numeric: {e}"
                            )
                else:
                    print(f"  Warning: Metric '{metric}' not found in columns")

            return results if results else None

        except Exception as e:
            print(f"  ⚠️  Error parsing {csv_file.name}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_table1(self, df):
        """Generate Table 1: Performance Metrics and Speedup"""
        print("\n" + "=" * 80)
        print("TABLE 1: Metrik Kinerja dan Speedup")
        print("=" * 80)

        latex_rows = []
        csv_data = []

        for _, row in df.iterrows():
            kernel = row["kernel"]
            mean_ms = row["mean_ms"]
            std_ms = row["std_ms"]
            min_ms = row["min_ms"]
            max_ms = row["max_ms"]
            speedup_pytorch = row["speedup_vs_pytorch"]
            speedup_p0 = (
                row["speedup_vs_p0"] if pd.notna(row["speedup_vs_p0"]) else None
            )

            # Format for LaTeX
            speedup_p0_str = (
                f"{speedup_p0:.2f}$\\times$" if speedup_p0 is not None else "-"
            )

            latex_row = (
                f"{kernel} & "
                f"{mean_ms:.3f} & {std_ms:.3f} & {min_ms:.3f} & {max_ms:.3f} & "
                f"{speedup_pytorch:.2f}$\\times$ & {speedup_p0_str} \\\\"
            )

            latex_rows.append(latex_row)
            print(latex_row)

            # CSV data
            csv_data.append(
                {
                    "Kernel": kernel,
                    "Mean (ms)": mean_ms,
                    "Std (ms)": std_ms,
                    "Min (ms)": min_ms,
                    "Max (ms)": max_ms,
                    "Speedup vs PyTorch": speedup_pytorch,
                    "Speedup vs p0": speedup_p0 if speedup_p0 else "",
                }
            )

        print("\n📋 LaTeX Code (copy-paste into thesis):")
        print("\\hline")
        for row in latex_rows:
            print(row)
        print("\\hline")

        # Save CSV for dashboard
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(self.dashboard_dir / "table1_performance.csv", index=False)
        print(f"\n✓ Saved: {self.output_dir_name}/table1_performance.csv")

        return latex_rows

    def generate_table5(self, df):
        """Generate Table 5: Memory and Correctness"""
        print("\n" + "=" * 80)
        print("TABLE 5: Metrik Memori dan Validitas")
        print("=" * 80)

        latex_rows = []
        csv_data = []

        for _, row in df.iterrows():
            kernel = row["kernel"]
            memory_mb = row["memory_mb"]
            mae = row["mae"]

            # Format for LaTeX
            mae_str = f"{mae:.6e}" if mae > 0 else "0.00"

            latex_row = f"{kernel} & {memory_mb:.2f} & {mae_str} \\\\"

            latex_rows.append(latex_row)
            print(latex_row)

            # CSV data
            csv_data.append(
                {
                    "Kernel": kernel,
                    "Memory (MB)": memory_mb,
                    "MAE": mae,
                    "Status": "PASS" if mae < 1e-3 else "FAIL",
                }
            )

        print("\n📋 LaTeX Code (copy-paste into thesis):")
        print("\\hline")
        for row in latex_rows:
            print(row)
        print("\\hline")

        # Save CSV for dashboard
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(self.dashboard_dir / "table5_memory_correctness.csv", index=False)
        print(f"\n✓ Saved: {self.output_dir_name}/table5_memory_correctness.csv")

        return latex_rows

    def inspect_ncu_csv(self, csv_file):
        """Inspect NCU CSV structure for debugging"""
        if not csv_file or not csv_file.exists():
            return

        print(f"\n  📋 Inspecting {csv_file.name}:")
        try:
            df = pd.read_csv(csv_file, nrows=20)  # Read first 20 rows
            print(f"     Shape: {df.shape}")
            print(f"     Columns: {list(df.columns)}")
            print(f"     First few rows:")
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print(f"     Error: {e}")

    def generate_table2(self):
        """Generate Table 2: Bottleneck Metrics"""
        print("\n" + "=" * 80)
        print("TABLE 2: Metrik Bottleneck")
        print("=" * 80)

        kernels = [
            ("pytorch", "PyTorch Reference"),
            ("p0", "p0 (Naive: 3 kernels)"),
            ("p1", "p1 (Tiled + Online Softmax)"),
            ("p2", "p2 (FlashLite Fused)"),
        ]

        metrics = [
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ]

        latex_rows = []
        csv_data = []

        for kernel_id, kernel_name in kernels:
            ncu_file = self.profiles_dir / f"{kernel_id}_bottleneck.ncu-rep"

            if not ncu_file.exists():
                print(f"⚠️  {kernel_name}: Profile not found ({ncu_file.name})")
                latex_row = f"{kernel_name} & - & - & - & - \\\\"
                latex_rows.append(latex_row)
                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Memory BW (GB/s)": None,
                        "Memory %": None,
                        "Compute (TFLOPS)": None,
                        "Compute %": None,
                    }
                )
                continue

            # Export to CSV
            csv_file = self.export_ncu_to_csv(ncu_file)
            data = self.parse_ncu_csv(csv_file, metrics)

            if data:
                mem_pct = data.get(metrics[0], 0)
                compute_pct = data.get(metrics[1], 0)

                # RTX 3050 Laptop specs (adjust if needed)
                peak_mem_bw = 112  # GB/s (approximate for RTX 3050 Laptop)
                peak_compute = 9  # TFLOPS (approximate for RTX 3050 Laptop)

                mem_gbs = mem_pct * peak_mem_bw / 100
                compute_tflops = compute_pct * peak_compute / 100

                latex_row = (
                    f"{kernel_name} & "
                    f"{mem_gbs:.1f} & {mem_pct:.1f} & "
                    f"{compute_tflops:.1f} & {compute_pct:.1f} \\\\"
                )

                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Memory BW (GB/s)": mem_gbs,
                        "Memory %": mem_pct,
                        "Compute (TFLOPS)": compute_tflops,
                        "Compute %": compute_pct,
                    }
                )
            else:
                latex_row = f"{kernel_name} & - & - & - & - \\\\"
                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Memory BW (GB/s)": None,
                        "Memory %": None,
                        "Compute (TFLOPS)": None,
                        "Compute %": None,
                    }
                )

            latex_rows.append(latex_row)
            print(latex_row)

        print("\n⚠️  NOTE: Peak values based on RTX 3050 Laptop specs")
        print("📋 LaTeX Code (copy-paste into thesis):")
        print("\\hline")
        for row in latex_rows:
            print(row)
        print("\\hline")

        # Save CSV for dashboard
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(self.dashboard_dir / "table2_bottleneck.csv", index=False)
        print(f"\n✓ Saved: {self.output_dir_name}/table2_bottleneck.csv")

        return latex_rows

    def generate_table3(self):
        """Generate Table 3: Shared Memory Metrics"""
        print("\n" + "=" * 80)
        print("TABLE 3: Metrik Shared Memory")
        print("=" * 80)

        kernels = [
            ("pytorch", "PyTorch Reference"),
            ("p0", "p0 (Naive: 3 kernels)"),
            ("p1", "p1 (Tiled + Online Softmax)"),
            ("p2", "p2 (FlashLite Fused)"),
        ]

        metrics = [
            "launch__shared_mem_per_block_static",
            "l1tex__data_bank_conflicts_pipe_lsu",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        ]

        latex_rows = []
        csv_data = []

        for kernel_id, kernel_name in kernels:
            ncu_file = self.profiles_dir / f"{kernel_id}_shared_memory.ncu-rep"

            if not ncu_file.exists():
                print(f"⚠️  {kernel_name}: Profile not found")
                latex_row = f"{kernel_name} & - & - & - \\\\"
                latex_rows.append(latex_row)
                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Shared Mem (KB)": None,
                        "Uncoalesced %": None,
                        "Bank Conflicts": None,
                    }
                )
                continue

            # Export to CSV
            csv_file = self.export_ncu_to_csv(ncu_file)
            data = self.parse_ncu_csv(csv_file, metrics)

            if data:
                shared_mem = data.get(metrics[0], 0) / 1024  # Convert to KB
                bank_conflicts = data.get(metrics[1], 0)
                coalescing = data.get(metrics[2], 0)
                uncoalesced_pct = 100 - coalescing  # Invert

                latex_row = (
                    f"{kernel_name} & "
                    f"{shared_mem:.2f} & {uncoalesced_pct:.1f} & {bank_conflicts:.0f} \\\\"
                )

                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Shared Mem (KB)": shared_mem,
                        "Uncoalesced %": uncoalesced_pct,
                        "Bank Conflicts": bank_conflicts,
                    }
                )
            else:
                latex_row = f"{kernel_name} & - & - & - \\\\"
                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Shared Mem (KB)": None,
                        "Uncoalesced %": None,
                        "Bank Conflicts": None,
                    }
                )

            latex_rows.append(latex_row)
            print(latex_row)

        print("\n📋 LaTeX Code (copy-paste into thesis):")
        print("\\hline")
        for row in latex_rows:
            print(row)
        print("\\hline")

        # Save CSV for dashboard
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(self.dashboard_dir / "table3_shared_memory.csv", index=False)
        print(f"\n✓ Saved: {self.output_dir_name}/table3_shared_memory.csv")

        return latex_rows

    def generate_table4(self):
        """Generate Table 4: Occupancy Metrics"""
        print("\n" + "=" * 80)
        print("TABLE 4: Metrik Occupancy")
        print("=" * 80)

        kernels = [
            ("pytorch", "PyTorch Reference"),
            ("p0", "p0 (Naive: 3 kernels)"),
            ("p1", "p1 (Tiled + Online Softmax)"),
            ("p2", "p2 (FlashLite Fused)"),
        ]

        metrics = [
            "launch__occupancy_limit_warps",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]

        latex_rows = []
        csv_data = []

        for kernel_id, kernel_name in kernels:
            ncu_file = self.profiles_dir / f"{kernel_id}_occupancy.ncu-rep"

            if not ncu_file.exists():
                print(f"⚠️  {kernel_name}: Profile not found")
                latex_row = f"{kernel_name} & - & - \\\\"
                latex_rows.append(latex_row)
                csv_data.append(
                    {"Kernel": kernel_name, "Theoretical %": None, "Achieved %": None}
                )
                continue

            # Export to CSV
            csv_file = self.export_ncu_to_csv(ncu_file)
            data = self.parse_ncu_csv(csv_file, metrics)

            if data:
                theoretical = data.get(metrics[0], 0)
                achieved = data.get(metrics[1], 0)

                latex_row = f"{kernel_name} & {theoretical:.1f} & {achieved:.1f} \\\\"

                csv_data.append(
                    {
                        "Kernel": kernel_name,
                        "Theoretical %": theoretical,
                        "Achieved %": achieved,
                    }
                )
            else:
                latex_row = f"{kernel_name} & - & - \\\\"
                csv_data.append(
                    {"Kernel": kernel_name, "Theoretical %": None, "Achieved %": None}
                )

            latex_rows.append(latex_row)
            print(latex_row)

        print("\n📋 LaTeX Code (copy-paste into thesis):")
        print("\\hline")
        for row in latex_rows:
            print(row)
        print("\\hline")

        # Save CSV for dashboard
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(self.dashboard_dir / "table4_occupancy.csv", index=False)
        print(f"\n✓ Saved: {self.output_dir_name}/table4_occupancy.csv")

        return latex_rows

    def generate_summary_json(self):
        """Generate summary JSON for dashboard"""
        summary = {
            "generated_files": [
                "table1_performance.csv",
                "table2_bottleneck.csv",
                "table3_shared_memory.csv",
                "table4_occupancy.csv",
                "table5_memory_correctness.csv",
            ],
            "gpu": "RTX 3050 Laptop 4GB",
            "cuda_version": "12.4",
            "test_config": "2048x2048x64",
        }

        with open(self.dashboard_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved: {self.output_dir_name}/summary.json")

    def run(self):
        """Run complete analysis"""
        print("=" * 80)
        print("ANALYZING ALL RESULTS - Generating LaTeX + CSV for Dashboard")
        print("=" * 80)

        # Table 1 & 5 (from performance benchmark)
        df = self.load_performance_data()
        if df is not None:
            self.generate_table1(df)
            self.generate_table5(df)
        else:
            print("\n⚠️  Skipping Tables 1 & 5 (performance data not found)")
            print("   Run: python benchmarks/benchmark_all_kernels.py")

        # Table 2, 3, 4 (from NCU profiles)
        print("\n⚠️  For Tables 2-4, you need NCU profile data:")
        print("   bash run_bottleneck_profile.sh")
        print("   bash run_shared_memory_profile.sh")
        print("   bash run_occupancy_profile.sh")

        self.generate_table2()
        self.generate_table3()
        self.generate_table4()

        # Generate summary
        self.generate_summary_json()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  📊 LaTeX tables → printed above (copy to thesis)")
        print(f"  📁 CSV files → {self.output_dir_name}/ (for Streamlit)")
        print("\nNext steps:")
        print(
            f"  1. Run: DASHBOARD_DATA_DIR={self.output_dir_name} streamlit run dashboard.py"
        )
        print("  2. Copy LaTeX code into your thesis document")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate CSV/LaTeX tables"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dashboard_data",
        help="Output directory for CSV files (default: dashboard_data)",
    )
    args = parser.parse_args()

    analyzer = ResultsAnalyzer(output_dir=args.output)
    analyzer.run()


if __name__ == "__main__":
    main()
