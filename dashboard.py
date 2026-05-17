import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Page Configuration
st.set_page_config(
    page_title="FlashLite-Attention Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 50%, #1e3a8a 100%);
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stMetric label {
        color: #e0e7ff !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 36px !important;
        font-weight: bold !important;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        padding: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #ffffff;
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Data Loading Functions
@st.cache_data
def load_dashboard_data(data_dir="dashboard_data"):
    """Load all CSV files from specified directory"""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        return None

    data = {}

    # Load each table
    tables = {
        "table1": "table1_performance.csv",
        "table2": "table2_bottleneck.csv",
        "table3": "table3_shared_memory.csv",
        "table4": "table4_occupancy.csv",
        "table5": "table5_memory_correctness.csv",
    }

    for key, filename in tables.items():
        filepath = data_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
        else:
            data[key] = None

    # Load summary
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            data["summary"] = json.load(f)
    else:
        data["summary"] = {}

    return data


def get_summary_stats(data):
    """Extract key statistics for overview"""
    stats = {}

    if data["table1"] is not None:
        df = data["table1"]

        # Find P2 and PyTorch rows
        p2_row = df[df["Kernel"].str.contains("p2", case=False)].iloc[0]
        pytorch_row = df[df["Kernel"].str.contains("PyTorch", case=False)].iloc[0]

        stats["peak_speedup"] = p2_row["Speedup vs PyTorch"]
        stats["p2_time"] = p2_row["Mean (ms)"]
        stats["pytorch_time"] = pytorch_row["Mean (ms)"]

    if data["table5"] is not None:
        df = data["table5"]
        stats["all_pass"] = (df["Status"] == "PASS").all()

        p0_row = df[df["Kernel"].str.contains("p0", case=False)].iloc[0]
        p2_row = df[df["Kernel"].str.contains("p2", case=False)].iloc[0]

        stats["memory_saved"] = (
            (p0_row["Memory (MB)"] - p2_row["Memory (MB)"]) / p0_row["Memory (MB)"]
        ) * 100

    return stats


# Get data directory from environment variable
def get_data_dir():
    import os

    return os.environ.get("DASHBOARD_DATA_DIR", "dashboard_data")


# Load Data
data_dir = get_data_dir()
data = load_dashboard_data(data_dir)

if data is None:
    st.error(f"⚠️ Dashboard data not found in directory: {data_dir}!")
    st.info(f"""
    Looking for data in: `{data_dir}/`
    
    Please run the following steps:

    1. **Generate benchmark data:**
       ```bash
       python benchmarks/benchmark_all_kernels.py
       ```

    2. **Generate profile data (optional but recommended):**
       ```bash
       cd benchmarks
       bash run_bottleneck_profile.sh
       bash run_shared_memory_profile.sh
       bash run_occupancy_profile.sh
       cd ..
       ```

    3. **Generate dashboard CSV files:**
       ```bash
       python analyze_all_results.py -o {data_dir}
       ```

    4. **Run this dashboard:**
       ```bash
       streamlit run dashboard.py
       # Or specify a different data directory:
       DASHBOARD_DATA_DIR={data_dir} streamlit run dashboard.py
       ```
    """)
    st.stop()

# Extract summary stats
stats = get_summary_stats(data)

# Header
st.markdown("<h1>⚡ FlashLite-Attention</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #e0e7ff; font-size: 20px; margin-top: -20px;'>"
    "High-Performance CUDA Implementation of Flash Attention"
    "</p>",
    unsafe_allow_html=True,
)

# Hardware Info Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**🖥️ GPU:** RTX 3050 4GB")
with col2:
    st.markdown("**⚙️ CUDA:** 12.4")
with col3:
    st.markdown("**🔥 PyTorch:** Integrated")
with col4:
    st.markdown("**📊 Config:** 4096×4096x64")

st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation")

    page = st.radio(
        "Select View:",
        [
            "📊 Overview",
            "⚡ Performance (Table 1)",
            "🔍 Bottleneck Analysis (Table 2)",
            "💾 Memory Analysis (Table 3)",
            "📈 Occupancy (Table 4)",
            "✅ Correctness (Table 5)",
        ],
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **FlashLite-Attention** implements three progressive kernel versions:

    - **P0**: Naive baseline (3 kernels)
    - **P1**: Tiled + Online Softmax
    - **P2**: Fully fused FlashLite

    Each version demonstrates different optimization techniques.
    """)

    st.markdown("---")
    st.markdown("### 🎓 Skripsi Project")
    st.markdown("**Author:** Defhanaya Sofhiea")
    st.markdown("**Institution:** Universitas Sriwijaya")

# PAGE: Overview
if page == "📊 Overview":
    st.header("📊 Performance Overview")

    # Key Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🚀 Peak Speedup",
            value=f"{stats.get('peak_speedup', 0):.2f}×",
            delta="vs PyTorch",
        )

    with col2:
        st.metric(
            label="💾 Memory Saved",
            value=f"{stats.get('memory_saved', 0):.1f}%",
            delta="P2 vs P0",
        )

    with col3:
        status_emoji = "✅" if stats.get("all_pass", False) else "❌"
        st.metric(
            label="✅ Correctness",
            value=f"{status_emoji} {'PASS' if stats.get('all_pass', False) else 'FAIL'}",
            delta="All tests",
        )

    st.markdown("---")

    # Execution Time Comparison
    if data["table1"] is not None:
        st.subheader("⏱️ Execution Time Comparison")

        df = data["table1"].copy()

        fig = go.Figure()

        colors = {
            "PyTorch": "#6366f1",
            "p0": "#ef4444",
            "p1": "#f59e0b",
            "p2": "#10b981",
        }

        for idx, row in df.iterrows():
            kernel = row["Kernel"]
            color = next((v for k, v in colors.items() if k in kernel), "#888888")

            fig.add_trace(
                go.Bar(
                    name=kernel,
                    x=[kernel],
                    y=[row["Mean (ms)"]],
                    error_y=dict(type="data", array=[row["Std (ms)"]], visible=True),
                    marker_color=color,
                    text=[f"{row['Mean (ms)']:.2f} ms"],
                    textposition="outside",
                )
            )

        fig.update_layout(
            title="Mean Execution Time with Standard Deviation",
            xaxis_title="Kernel",
            yaxis_title="Time (ms)",
            showlegend=False,
            height=400,
            plot_bgcolor="rgba(255,255,255,0.9)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Speedup Chart
    if data["table1"] is not None:
        st.subheader("📈 Speedup Analysis")

        col1, col2 = st.columns(2)

        with col1:
            df = data["table1"].copy()
            df = df[df["Speedup vs PyTorch"].notna()]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df["Kernel"],
                        y=df["Speedup vs PyTorch"],
                        marker_color=["#6366f1", "#ef4444", "#f59e0b", "#10b981"],
                        text=df["Speedup vs PyTorch"].apply(lambda x: f"{x:.2f}×"),
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                title="Speedup vs PyTorch Baseline",
                xaxis_title="Kernel",
                yaxis_title="Speedup Factor",
                height=350,
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if data["table5"] is not None:
                df = data["table5"].copy()

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=df["Kernel"],
                            y=df["Memory (MB)"],
                            marker_color=["#6366f1", "#ef4444", "#f59e0b", "#10b981"],
                            text=df["Memory (MB)"].apply(lambda x: f"{x:.1f} MB"),
                            textposition="outside",
                        )
                    ]
                )

                fig.update_layout(
                    title="Peak Memory Usage",
                    xaxis_title="Kernel",
                    yaxis_title="Memory (MB)",
                    height=350,
                    plot_bgcolor="rgba(255,255,255,0.9)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True)

# PAGE: Performance (Table 1)
elif page == "⚡ Performance (Table 1)":
    st.header("⚡ Performance Metrics and Speedup (Table 1)")

    if data["table1"] is not None:
        df = data["table1"]

        # Display dataframe
        st.dataframe(
            df.style.highlight_max(
                subset=["Speedup vs PyTorch", "Speedup vs p0"], color="lightgreen"
            ).highlight_min(subset=["Mean (ms)", "Min (ms)"], color="lightgreen"),
            use_container_width=True,
        )

        # Detailed analysis
        st.subheader("📊 Detailed Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Best Performance")
            best_row = df.loc[df["Mean (ms)"].idxmin()]
            st.success(f"""
            **Fastest Kernel:** {best_row["Kernel"]}
            - Mean Time: {best_row["Mean (ms)"]:.3f} ms
            - Speedup vs PyTorch: {best_row["Speedup vs PyTorch"]:.2f}×
            """)

        with col2:
            st.markdown("### Consistency")
            most_consistent = df.loc[df["Std (ms)"].idxmin()]
            st.info(f"""
            **Most Consistent:** {most_consistent["Kernel"]}
            - Std Dev: {most_consistent["Std (ms)"]:.3f} ms
            - Range: {most_consistent["Min (ms)"]:.3f} - {most_consistent["Max (ms)"]:.3f} ms
            """)
    else:
        st.warning(
            "⚠️ Table 1 data not found. Run `python benchmarks/benchmark_all_kernels.py`"
        )

# PAGE: Bottleneck Analysis (Table 2)
elif page == "🔍 Bottleneck Analysis (Table 2)":
    st.header("🔍 Bottleneck Metrics (Table 2)")

    if data["table2"] is not None:
        df = data["table2"]

        # Display dataframe
        st.dataframe(df, use_container_width=True)

        # Visualization
        st.subheader("📊 Memory vs Compute Utilization")

        # Filter out rows with no data
        df_plot = df.dropna()

        if not df_plot.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="Memory %",
                    x=df_plot["Kernel"],
                    y=df_plot["Memory %"],
                    marker_color="#3b82f6",
                    text=df_plot["Memory %"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                )
            )

            fig.add_trace(
                go.Bar(
                    name="Compute %",
                    x=df_plot["Kernel"],
                    y=df_plot["Compute %"],
                    marker_color="#10b981",
                    text=df_plot["Compute %"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                )
            )

            fig.update_layout(
                barmode="group",
                title="Resource Utilization (%)",
                xaxis_title="Kernel",
                yaxis_title="Utilization (%)",
                height=400,
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Analysis
            st.subheader("📋 Bottleneck Analysis")
            for _, row in df_plot.iterrows():
                mem_pct = row["Memory %"]
                comp_pct = row["Compute %"]

                if mem_pct > comp_pct:
                    bottleneck = "Memory-bound"
                    color = "blue"
                else:
                    bottleneck = "Compute-bound"
                    color = "green"

                st.markdown(
                    f"**{row['Kernel']}:** :{color}[{bottleneck}] (Mem: {mem_pct:.1f}%, Compute: {comp_pct:.1f}%)"
                )
    else:
        st.warning("⚠️ Table 2 data not found. Run NSight profiling scripts.")

# PAGE: Memory Analysis (Table 3)
elif page == "💾 Memory Analysis (Table 3)":
    st.header("💾 Shared Memory Metrics (Table 3)")

    if data["table3"] is not None:
        df = data["table3"]

        # Display dataframe
        st.dataframe(df, use_container_width=True)

        # Visualization
        df_plot = df.dropna()

        if not df_plot.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=df_plot["Kernel"],
                            y=df_plot["Shared Mem (KB)"],
                            marker_color="#8b5cf6",
                            text=df_plot["Shared Mem (KB)"].apply(
                                lambda x: f"{x:.2f} KB"
                            ),
                            textposition="outside",
                        )
                    ]
                )

                fig.update_layout(
                    title="Shared Memory Usage",
                    xaxis_title="Kernel",
                    yaxis_title="Shared Memory (KB)",
                    height=350,
                    plot_bgcolor="rgba(255,255,255,0.9)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=df_plot["Kernel"],
                            y=df_plot["Bank Conflicts"],
                            marker_color="#ef4444",
                            text=df_plot["Bank Conflicts"].apply(lambda x: f"{x:.0f}"),
                            textposition="outside",
                        )
                    ]
                )

                fig.update_layout(
                    title="Bank Conflicts",
                    xaxis_title="Kernel",
                    yaxis_title="Number of Conflicts",
                    height=350,
                    plot_bgcolor="rgba(255,255,255,0.9)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Table 3 data not found. Run NSight profiling scripts.")

# PAGE: Occupancy (Table 4)
elif page == "📈 Occupancy (Table 4)":
    st.header("📈 Occupancy Metrics (Table 4)")

    if data["table4"] is not None:
        df = data["table4"]

        # Display dataframe
        st.dataframe(df, use_container_width=True)

        # Visualization
        df_plot = df.dropna()

        if not df_plot.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="Theoretical %",
                    x=df_plot["Kernel"],
                    y=df_plot["Theoretical %"],
                    marker_color="#06b6d4",
                    text=df_plot["Theoretical %"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                )
            )

            fig.add_trace(
                go.Bar(
                    name="Achieved %",
                    x=df_plot["Kernel"],
                    y=df_plot["Achieved %"],
                    marker_color="#10b981",
                    text=df_plot["Achieved %"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                )
            )

            fig.update_layout(
                barmode="group",
                title="Theoretical vs Achieved Occupancy",
                xaxis_title="Kernel",
                yaxis_title="Occupancy (%)",
                height=400,
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Efficiency calculation
            st.subheader("🎯 Occupancy Efficiency")
            for _, row in df_plot.iterrows():
                efficiency = (
                    (row["Achieved %"] / row["Theoretical %"]) * 100
                    if row["Theoretical %"] > 0
                    else 0
                )
                st.markdown(
                    f"**{row['Kernel']}:** {efficiency:.1f}% efficient (Achieved: {row['Achieved %']:.1f}% / Theoretical: {row['Theoretical %']:.1f}%)"
                )
    else:
        st.warning("⚠️ Table 4 data not found. Run NSight profiling scripts.")

# PAGE: Correctness (Table 5)
elif page == "✅ Correctness (Table 5)":
    st.header("✅ Memory and Correctness (Table 5)")

    if data["table5"] is not None:
        df = data["table5"]

        # Display dataframe with color coding
        def color_status(val):
            color = "lightgreen" if val == "PASS" else "lightcoral"
            return f"background-color: {color}"

        st.dataframe(
            df.style.applymap(color_status, subset=["Status"]), use_container_width=True
        )

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df["Kernel"],
                        y=df["Memory (MB)"],
                        marker_color=["#6366f1", "#ef4444", "#f59e0b", "#10b981"],
                        text=df["Memory (MB)"].apply(lambda x: f"{x:.1f} MB"),
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                title="Peak Memory Usage",
                xaxis_title="Kernel",
                yaxis_title="Memory (MB)",
                height=350,
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # MAE visualization (log scale)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df["Kernel"],
                        y=df["MAE"],
                        marker_color=[
                            "#10b981" if s == "PASS" else "#ef4444"
                            for s in df["Status"]
                        ],
                        text=df["MAE"].apply(lambda x: f"{x:.2e}"),
                        textposition="outside",
                    )
                ]
            )

            fig.update_layout(
                title="Mean Absolute Error (MAE)",
                xaxis_title="Kernel",
                yaxis_title="MAE (log scale)",
                yaxis_type="log",
                height=350,
                plot_bgcolor="rgba(255,255,255,0.9)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.subheader("📋 Correctness Summary")
        pass_count = (df["Status"] == "PASS").sum()
        total = len(df)

        if pass_count == total:
            st.success(f"✅ All {total} kernels passed correctness tests!")
        else:
            st.warning(f"⚠️ {pass_count}/{total} kernels passed correctness tests")
    else:
        st.warning(
            "⚠️ Table 5 data not found. Run `python benchmarks/benchmark_all_kernels.py`"
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>"
    "FlashLite-Attention • CUDA 12.4 • PyTorch • RTX 3050 4GB<br>"
    "Thesis Project Dashboard"
    "</p>",
    unsafe_allow_html=True,
)
