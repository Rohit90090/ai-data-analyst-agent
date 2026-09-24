import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =========================================================
# IMPORTS
# =========================================================

from data_loader import (
    load_data,
    clean_column_names,
    get_basic_info
)

from profiler import (
    dataset_profile,
    numeric_summary,
    categorical_summary,
    detect_outliers
)

from analyzer import analyze_question

from report_generator import (
    generate_report,
    money
)

from root_cause import analyze_root_cause


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "selected_analysis" not in st.session_state:
    st.session_state.selected_analysis = None

if "new_analysis_mode" not in st.session_state:
    st.session_state.new_analysis_mode = True

# Selected visualization
if "selected_chart_type" not in st.session_state:
    st.session_state.selected_chart_type = "Bar Chart"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 AI Data Analyst")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Data Upload",
        "📊 Ask Your Data",
        "🤖 AI Analyst Report",
        "🔬 Root-Cause Analysis",
        "🔍 Data Profile",
        "📈 Data Preview"
    ]
)


# =========================================================
# ASK YOUR DATA SIDEBAR
# =========================================================

if page == "📊 Ask Your Data":

    st.sidebar.divider()

    # =====================================================
    # CHART STUDIO
    # =====================================================

    st.sidebar.subheader("🎨 Chart Studio")

    st.sidebar.caption(
        "Select a visualization"
    )

    chart_options = {
        "📊 Bar Chart": "Bar Chart",
        "📈 Line Chart": "Line Chart",
        "🍩 Donut Chart": "Donut Chart",
        "🥧 Pie Chart": "Pie Chart",
        "🔵 Scatter Chart": "Scatter Chart"
    }

    for label, chart_type in chart_options.items():

        is_selected = (
            st.session_state.selected_chart_type
            == chart_type
        )

        if is_selected:
            button_type = "primary"
        else:
            button_type = "secondary"

        if st.sidebar.button(
            label,
            key=f"chart_selector_{chart_type}",
            use_container_width=True,
            type=button_type
        ):

            st.session_state.selected_chart_type = chart_type

            st.rerun()

    st.sidebar.divider()

    # =====================================================
    # ANALYSIS HISTORY
    # =====================================================

    st.sidebar.subheader("💬 Analysis History")

    if st.sidebar.button(
        "＋ New Analysis",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.new_analysis_mode = True
        st.session_state.selected_analysis = None

        st.rerun()

    history = st.session_state.analysis_history

    if not history:

        st.sidebar.caption(
            "Your analysis questions will appear here."
        )

    else:

        for index, item in enumerate(history):

            question_text = item.get(
                "question",
                "Untitled Analysis"
            )

            if len(question_text) > 34:

                button_label = (
                    question_text[:31]
                    + "..."
                )

            else:

                button_label = question_text

            if (
                st.session_state.selected_analysis
                == index
                and not st.session_state.new_analysis_mode
            ):

                button_label = (
                    "🟣 "
                    + button_label
                )

            else:

                button_label = (
                    "💬 "
                    + button_label
                )

            if st.sidebar.button(
                button_label,
                key=f"history_button_{index}",
                use_container_width=True
            ):

                st.session_state.selected_analysis = index
                st.session_state.new_analysis_mode = False

                st.rerun()


# =========================================================
# DATA CHECK FUNCTION
# =========================================================

def require_data():

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first from "
            "📂 Data Upload."
        )

        return False

    return True


# =========================================================
# CHART DATA HELPERS
# =========================================================

def _to_dataframe(value):
    """Convert common analyzer outputs into a DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if value is None:
        return pd.DataFrame()

    if isinstance(value, dict):
        try:
            # Single record dictionary
            if all(not isinstance(v, (list, tuple, pd.Series, pd.Index)) for v in value.values()):
                return pd.DataFrame([value])
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, (list, tuple)):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def _get_chart_source(result):
    """
    Analyzer may return data under either `table` or `chart_data`.
    Prefer table, then fall back to chart_data.
    """
    table = _to_dataframe(result.get("table"))

    if not table.empty:
        return table

    chart_data = _to_dataframe(result.get("chart_data"))

    if not chart_data.empty:
        return chart_data

    return pd.DataFrame()


def get_chart_columns(table):
    """Find sensible x/y columns for the selected visualization."""

    chart_df = _to_dataframe(table)

    if chart_df.empty:
        return None, None, [], []

    # Remove completely empty columns
    chart_df = chart_df.dropna(axis=1, how="all")

    if chart_df.empty:
        return None, None, [], []

    # Treat datetime columns as categorical/time dimensions
    datetime_columns = [
        col for col in chart_df.columns
        if pd.api.types.is_datetime64_any_dtype(chart_df[col])
    ]

    numeric_columns = [
        col
        for col in chart_df.columns
        if pd.api.types.is_numeric_dtype(chart_df[col])
    ]

    categorical_columns = [
        col
        for col in chart_df.columns
        if col not in numeric_columns
    ]

    # Datetime columns should be valid dimensions
    for col in datetime_columns:
        if col not in categorical_columns:
            categorical_columns.insert(0, col)

    # If a column contains date/month strings, try to detect it
    for col in list(categorical_columns):
        name = str(col).lower()
        if any(k in name for k in ["date", "month", "year", "time"]):
            if col not in datetime_columns:
                categorical_columns.remove(col)
                categorical_columns.insert(0, col)

    if not numeric_columns:
        return None, None, numeric_columns, categorical_columns

    # Preferred numeric measure
    preferred_numeric_keywords = [
        "revenue", "sales", "amount", "profit",
        "quantity", "orders", "transactions",
        "customers", "price", "value", "total",
        "change", "count"
    ]

    y_col = None

    for keyword in preferred_numeric_keywords:
        for col in numeric_columns:
            if keyword in str(col).lower():
                y_col = col
                break
        if y_col is not None:
            break

    if y_col is None:
        y_col = numeric_columns[0]

    # Preferred dimension
    preferred_dimension_keywords = [
        "product", "description", "country", "category",
        "customer", "month", "date", "year",
        "region", "segment", "stockcode", "name"
    ]

    x_col = None

    for keyword in preferred_dimension_keywords:
        for col in categorical_columns:
            if keyword in str(col).lower():
                x_col = col
                break
        if x_col is not None:
            break

    if x_col is None and categorical_columns:
        x_col = categorical_columns[0]

    # If no categorical column exists, use a non-selected numeric
    # column as x. This makes scatter work on numeric-only results.
    if x_col is None and len(numeric_columns) >= 2:
        x_col = numeric_columns[0]
        if y_col == x_col:
            y_col = numeric_columns[1]

    return x_col, y_col, numeric_columns, categorical_columns


def prepare_chart_data(table):
    chart_df = _to_dataframe(table)

    if chart_df.empty:
        return None

    chart_df = chart_df.dropna(axis=1, how="all")

    if chart_df.empty:
        return None

    x_col, y_col, numeric_columns, categorical_columns = get_chart_columns(chart_df)

    if y_col is None:
        return None

    # Convert obvious date/month columns
    if x_col is not None:
        x_name = str(x_col).lower()

        if any(k in x_name for k in ["date", "month", "time"]):
            converted = pd.to_datetime(
                chart_df[x_col],
                errors="coerce"
            )

            if converted.notna().sum() > 0:
                chart_df[x_col] = converted

    # Remove invalid values
    required_columns = [y_col]

    if x_col is not None:
        required_columns.append(x_col)

    chart_df = chart_df.dropna(subset=required_columns)

    if chart_df.empty:
        return None

    return {
        "data": chart_df,
        "x": x_col,
        "y": y_col,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns
    }


# =========================================================
# GENERATE SELECTED CHART
# =========================================================

def generate_selected_chart(table, chart_type):

    prepared = prepare_chart_data(table)

    if prepared is None:
        return None, (
            "No structured numeric data was returned for this analysis."
        )

    chart_df = prepared["data"].copy()
    x_col = prepared["x"]
    y_col = prepared["y"]
    numeric_columns = prepared["numeric_columns"]

    # ---------------------------------------------------------
    # BAR
    # ---------------------------------------------------------
    if chart_type == "Bar Chart":

        if x_col is None:
            return None, "Bar Chart needs a category or time dimension."

        plot_df = chart_df.copy()

        # Aggregate duplicate categories so bars represent totals.
        if not pd.api.types.is_numeric_dtype(plot_df[x_col]):
            plot_df = (
                plot_df.groupby(x_col, as_index=False)[y_col]
                .sum()
                .sort_values(y_col, ascending=False)
            )

        plot_df = plot_df.head(15)

        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            title=f"{y_col} by {x_col}",
            text_auto=".3s"
        )

        fig.update_layout(
            xaxis_title=str(x_col),
            yaxis_title=str(y_col),
            xaxis_tickangle=-45,
            hovermode="x unified",
            height=550
        )

        return fig, None

    # ---------------------------------------------------------
    # LINE
    # ---------------------------------------------------------
    if chart_type == "Line Chart":

        if x_col is None:
            return None, "Line Chart needs an ordered dimension."

        plot_df = chart_df.copy()

        x_name = str(x_col).lower()

        if (
            not pd.api.types.is_datetime64_any_dtype(plot_df[x_col])
            and any(k in x_name for k in ["date", "month", "time"])
        ):
            plot_df[x_col] = pd.to_datetime(
                plot_df[x_col],
                errors="coerce"
            )

        plot_df = plot_df.dropna(subset=[x_col, y_col])

        if plot_df.empty:
            return None, "No valid observations are available for Line Chart."

        # Aggregate repeated dates/months.
        if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
            plot_df = (
                plot_df.groupby(x_col, as_index=False)[y_col]
                .sum()
                .sort_values(x_col)
            )
        else:
            plot_df = plot_df.sort_values(x_col)

        fig = px.line(
            plot_df,
            x=x_col,
            y=y_col,
            markers=True,
            title=f"{y_col} Trend"
        )

        fig.update_layout(
            xaxis_title=str(x_col),
            yaxis_title=str(y_col),
            hovermode="x unified",
            height=550
        )

        return fig, None

    # ---------------------------------------------------------
    # DONUT
    # ---------------------------------------------------------
    if chart_type == "Donut Chart":

        if x_col is None:
            return None, "Donut Chart needs a category dimension."

        plot_df = chart_df.copy()

        # Aggregate categories first.
        if not pd.api.types.is_numeric_dtype(plot_df[x_col]):
            plot_df = (
                plot_df.groupby(x_col, as_index=False)[y_col]
                .sum()
                .sort_values(y_col, ascending=False)
            )

        plot_df = plot_df.head(10)

        fig = px.pie(
            plot_df,
            names=x_col,
            values=y_col,
            hole=0.55,
            title=f"{y_col} Distribution"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            showlegend=True,
            height=550
        )

        return fig, None

    # ---------------------------------------------------------
    # PIE
    # ---------------------------------------------------------
    if chart_type == "Pie Chart":

        if x_col is None:
            return None, "Pie Chart needs a category dimension."

        plot_df = chart_df.copy()

        if not pd.api.types.is_numeric_dtype(plot_df[x_col]):
            plot_df = (
                plot_df.groupby(x_col, as_index=False)[y_col]
                .sum()
                .sort_values(y_col, ascending=False)
            )

        plot_df = plot_df.head(8)

        fig = px.pie(
            plot_df,
            names=x_col,
            values=y_col,
            title=f"{y_col} Composition"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            showlegend=True,
            height=550
        )

        return fig, None

    # ---------------------------------------------------------
    # SCATTER
    # ---------------------------------------------------------
    if chart_type == "Scatter Chart":

        if len(numeric_columns) < 2:
            return None, (
                "Scatter Chart needs at least two numeric columns "
                "in the analysis result."
            )

        # Prefer meaningful numeric pairs.
        scatter_x = numeric_columns[0]
        scatter_y = numeric_columns[1]

        x_keywords = [
            "price", "cost", "quantity",
            "orders", "customers", "amount"
        ]

        y_keywords = [
            "revenue", "sales", "profit",
            "quantity", "value", "amount"
        ]

        for keyword in x_keywords:
            matches = [
                col for col in numeric_columns
                if keyword in str(col).lower()
            ]
            if matches:
                scatter_x = matches[0]
                break

        for keyword in y_keywords:
            matches = [
                col for col in numeric_columns
                if col != scatter_x and keyword in str(col).lower()
            ]
            if matches:
                scatter_y = matches[0]
                break

        plot_df = chart_df.dropna(
            subset=[scatter_x, scatter_y]
        )

        if plot_df.empty:
            return None, "No valid numeric observations for Scatter Chart."

        fig = px.scatter(
            plot_df,
            x=scatter_x,
            y=scatter_y,
            title=f"{scatter_y} vs {scatter_x}",
            trendline="ols" if len(plot_df) >= 3 else None
        )

        fig.update_layout(
            xaxis_title=str(scatter_x),
            yaxis_title=str(scatter_y),
            hovermode="closest",
            height=550
        )

        return fig, None

    return None, "Unknown chart type."


# =========================================================
# DISPLAY ANALYSIS RESULT
# =========================================================

def display_analysis_result(item):

    if not isinstance(item, dict):
        st.write(item)
        return

    result = item.get("result", {})

    if not isinstance(result, dict):
        st.write(result)
        return

    title = result.get("title")

    if title:
        st.subheader(title)

    question = item.get("question")

    if question:
        st.caption(f"💬 {question}")

    answer = result.get("answer")

    if answer:
        st.success(str(answer))

    # ---------------------------------------------------------
    # Get chart/table data
    # ---------------------------------------------------------

    table = _get_chart_source(result)

    if not table.empty:

        st.subheader("📊 Analysis Visualization")

        selected_chart = st.session_state.get(
            "selected_chart_type",
            "Bar Chart"
        )

        st.caption(
            f"🎨 Current visualization: **{selected_chart}**"
        )

        # Show chart first so the visualization is immediately visible.
        chart, chart_error = generate_selected_chart(
            table,
            selected_chart
        )

        if chart is not None:
            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "responsive": True
                },
                key=(
                    f"analysis_chart_"
                    f"{item.get('id', 'default')}_"
                    f"{selected_chart.replace(' ', '_')}"
                )
            )
        else:
            st.warning(
                f"⚠️ {chart_error}"
            )

        # Keep the underlying analysis table below the chart.
        st.subheader("📋 Analysis Result")

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "This analysis returned a text answer but no structured "
            "chart data. Try a question such as 'Show top 10 products "
            "by revenue' or 'What is the monthly revenue trend?'"
        )

    insight = result.get("insight")

    if insight:
        st.info(f"💡 {insight}")


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title(
        "🤖 AI Data Analyst Agent"
    )

    st.markdown(
        """
        ### Your AI-powered business analysis workspace

        Upload a CSV or Excel file and automatically perform:

        - 📊 Dataset analysis
        - 🔎 Natural-language business questions
        - 🤖 Automated analyst report
        - 🔬 Root-cause analysis
        - 📈 Data profiling
        - 📦 Product analysis
        - 💰 Revenue analysis
        - 🌍 Country analysis
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Analysis",
            "Enabled"
        )

    with col2:

        st.metric(
            "Root Cause",
            "Enabled"
        )

    with col3:

        st.metric(
            "CSV / Excel",
            "Supported"
        )

    st.info(
        "Go to Data Upload and upload your dataset to begin."
    )


# =========================================================
# DATA UPLOAD
# =========================================================

elif page == "📂 Data Upload":

    st.title(
        "📂 Data Upload"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        try:

            df = load_data(
                uploaded_file
            )

            df = clean_column_names(
                df
            )

            st.session_state.df = df

            st.session_state.analysis_history = []
            st.session_state.selected_analysis = None
            st.session_state.new_analysis_mode = True

            st.success(
                f"Dataset loaded successfully: "
                f"{df.shape[0]:,} rows × "
                f"{df.shape[1]:,} columns"
            )

            info = get_basic_info(
                df
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Rows",
                    f"{info['rows']:,}"
                )

            with col2:

                st.metric(
                    "Columns",
                    f"{info['columns']:,}"
                )

            with col3:

                st.metric(
                    "Missing Values",
                    f"{info['missing_values']:,}"
                )

            with col4:

                st.metric(
                    "Duplicates",
                    f"{info['duplicate_rows']:,}"
                )

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Could not load dataset: {e}"
            )


# =========================================================
# ASK YOUR DATA
# =========================================================

elif page == "📊 Ask Your Data":

    st.title(
        "📊 Ask Your Data"
    )

    st.caption(
        "Ask business questions in normal English. "
        "Then switch the visualization from the Chart Studio "
        "in the sidebar."
    )

    if require_data():

        df = st.session_state.df

        # =================================================
        # EXISTING ANALYSIS
        # =================================================

        if (
            not st.session_state.new_analysis_mode
            and
            st.session_state.selected_analysis is not None
            and
            st.session_state.selected_analysis
            <
            len(
                st.session_state.analysis_history
            )
        ):

            selected_index = (
                st.session_state.selected_analysis
            )

            selected_item = (
                st.session_state.analysis_history[
                    selected_index
                ]
            )

            st.subheader(
                "🔎 Selected Analysis"
            )

            display_analysis_result(
                selected_item
            )

            st.divider()

            if st.button(
                "＋ Ask a New Question",
                type="primary"
            ):

                st.session_state.new_analysis_mode = True
                st.session_state.selected_analysis = None

                st.rerun()

        # =================================================
        # NEW ANALYSIS
        # =================================================

        else:

            st.subheader(
                "💬 New Analysis"
            )

            question = st.text_input(
                "Ask a business question",
                placeholder=(
                    "Example: Which product generated "
                    "the highest revenue?"
                ),
                key="new_analysis_question"
            )

            # -------------------------------------------------
            # QUICK QUESTIONS
            # -------------------------------------------------

            st.caption(
                "Quick questions"
            )

            quick_col1, quick_col2, quick_col3 = st.columns(3)

            quick_questions = [
                "Which product generated the highest revenue?",
                "Show top 5 products by revenue.",
                "What is the monthly revenue trend?",
                "Which country generated the most revenue?",
                "Which product sold the most units?",
                "What is the total revenue?"
            ]

            with quick_col1:

                if st.button(
                    "🏆 Top Product",
                    use_container_width=True
                ):

                    st.session_state.new_analysis_question = (
                        quick_questions[0]
                    )

                    st.rerun()

            with quick_col2:

                if st.button(
                    "📈 Revenue Trend",
                    use_container_width=True
                ):

                    st.session_state.new_analysis_question = (
                        quick_questions[2]
                    )

                    st.rerun()

            with quick_col3:

                if st.button(
                    "🌍 Top Country",
                    use_container_width=True
                ):

                    st.session_state.new_analysis_question = (
                        quick_questions[3]
                    )

                    st.rerun()

            # -------------------------------------------------
            # ANALYZE BUTTON
            # -------------------------------------------------

            if st.button(
                "🔎 Analyze",
                type="primary",
                use_container_width=True
            ):

                question = st.session_state.get(
                    "new_analysis_question",
                    ""
                )

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Analyzing your dataset..."
                    ):

                        try:

                            result = analyze_question(
                                df,
                                question
                            )

                            history_item = {
                                "id": len(
                                    st.session_state.analysis_history
                                ),
                                "question": question,
                                "result": result
                            }

                            st.session_state.analysis_history.append(
                                history_item
                            )

                            st.session_state.selected_analysis = (
                                len(
                                    st.session_state.analysis_history
                                ) - 1
                            )

                            st.session_state.new_analysis_mode = False
                            st.session_state.new_analysis_question = ""

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Analysis error: {e}"
                            )


# =========================================================
# AI ANALYST REPORT
# =========================================================

elif page == "🤖 AI Analyst Report":

    st.title(
        "🤖 AI Analyst Report"
    )

    if require_data():

        df = st.session_state.df

        with st.spinner(
            "Generating analyst report..."
        ):

            try:

                report = generate_report(
                    df
                )

                st.header(
                    "📋 Dataset Overview"
                )

                overview = report.get(
                    "overview",
                    {}
                )

                if overview:

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:

                        st.metric(
                            "Rows",
                            f"{overview.get('rows', 0):,}"
                        )

                    with c2:

                        st.metric(
                            "Columns",
                            f"{overview.get('columns', 0):,}"
                        )

                    with c3:

                        st.metric(
                            "Missing Values",
                            f"{overview.get('missing_values', 0):,}"
                        )

                    with c4:

                        st.metric(
                            "Duplicates",
                            f"{overview.get('duplicates', 0):,}"
                        )

                st.divider()

                st.header(
                    "💰 Revenue Summary"
                )

                revenue_summary = report.get(
                    "revenue_summary",
                    {}
                )

                if revenue_summary:

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Total Revenue",
                            revenue_summary.get(
                                "total_revenue",
                                "N/A"
                            )
                        )

                    with c2:

                        st.metric(
                            "Average Revenue",
                            revenue_summary.get(
                                "average_revenue",
                                "N/A"
                            )
                        )

                    with c3:

                        st.metric(
                            "Top 10 Contribution",
                            revenue_summary.get(
                                "top_10_contribution",
                                "N/A"
                            )
                        )

                st.header(
                    "🏆 Top 10 Products"
                )

                top_products = report.get(
                    "top_products",
                    pd.DataFrame()
                )

                if (
                    isinstance(
                        top_products,
                        pd.DataFrame
                    )
                    and
                    not top_products.empty
                ):

                    st.dataframe(
                        top_products,
                        use_container_width=True
                    )

                    revenue_columns = [
                        col
                        for col in top_products.columns
                        if "revenue"
                        in str(col).lower()
                    ]

                    if revenue_columns:

                        chart = px.bar(
                            top_products,
                            x=top_products.columns[0],
                            y=revenue_columns[0],
                            title="Top Products by Revenue"
                        )

                        chart.update_layout(
                            xaxis_tickangle=-45
                        )

                        st.plotly_chart(
                            chart,
                            use_container_width=True
                        )

                st.header(
                    "📦 Quantity Summary"
                )

                quantity_summary = report.get(
                    "quantity_summary",
                    {}
                )

                if quantity_summary:

                    for key, value in quantity_summary.items():

                        st.write(
                            f"**{key}:** {value}"
                        )

                st.header(
                    "👥 Customer Summary"
                )

                customer_summary = report.get(
                    "customer_summary",
                    {}
                )

                if customer_summary:

                    for key, value in customer_summary.items():

                        st.write(
                            f"**{key}:** {value}"
                        )

                st.header(
                    "📈 Revenue Trend"
                )

                time_trend = report.get(
                    "time_trend",
                    pd.DataFrame()
                )

                if (
                    isinstance(
                        time_trend,
                        pd.DataFrame
                    )
                    and
                    not time_trend.empty
                ):

                    st.dataframe(
                        time_trend,
                        use_container_width=True
                    )

                    trend_numeric = [
                        col
                        for col in time_trend.columns
                        if pd.api.types.is_numeric_dtype(
                            time_trend[col]
                        )
                    ]

                    if trend_numeric:

                        date_column = (
                            time_trend.columns[0]
                        )

                        trend_chart = px.line(
                            time_trend,
                            x=date_column,
                            y=trend_numeric[0],
                            markers=True,
                            title="Revenue Trend"
                        )

                        st.plotly_chart(
                            trend_chart,
                            use_container_width=True
                        )

                st.header(
                    "🔍 Data Quality"
                )

                quality = report.get(
                    "data_quality",
                    {}
                )

                if quality:

                    for key, value in quality.items():

                        st.write(
                            f"**{key}:** {value}"
                        )

                st.header(
                    "💡 Key Business Insights"
                )

                insights = report.get(
                    "business_insights",
                    []
                )

                if isinstance(
                    insights,
                    list
                ):

                    for insight in insights:

                        st.info(
                            str(insight)
                        )

            except Exception as e:

                st.error(
                    f"Report generation error: {e}"
                )


# =========================================================
# ROOT-CAUSE ANALYSIS
# =========================================================

elif page == "🔬 Root-Cause Analysis":

    st.title(
        "🔬 AI Root-Cause Analysis"
    )

    st.markdown(
        """
        Find the major factors behind month-to-month
        business performance changes.

        **Analyzed drivers:**

        - 💰 Revenue
        - 📦 Quantity
        - 🧾 Transactions
        - 🛒 Average Order Value
        - 🏆 Products
        - 🌍 Countries
        """
    )

    st.divider()

    if require_data():

        df = st.session_state.df.copy()

        possible_date_columns = [
            "invoicedate",
            "invoice_date",
            "date",
            "datetime",
            "month"
        ]

        date_col = None

        for col in df.columns:

            col_clean = str(
                col
            ).strip().lower()

            if col_clean in possible_date_columns:

                date_col = col
                break

        if date_col is None:

            st.error(
                "❌ No date/month column was detected."
            )

            st.info(
                "Required column example: "
                "InvoiceDate or Month"
            )

        else:

            parsed_dates = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

            valid_dates = parsed_dates.dropna()

            if valid_dates.empty:

                st.error(
                    "The detected date column contains "
                    "no valid dates."
                )

            else:

                months = (
                    valid_dates
                    .dt.to_period("M")
                    .astype(str)
                    .drop_duplicates()
                    .sort_values()
                    .tolist()
                )

                if len(months) < 2:

                    st.warning(
                        "At least two months are required "
                        "for root-cause analysis."
                    )

                else:

                    selected_month = st.selectbox(
                        "📅 Select month to investigate",
                        months,
                        index=len(months) - 1
                    )

                    previous_index = (
                        months.index(
                            selected_month
                        ) - 1
                    )

                    if previous_index >= 0:

                        st.caption(
                            f"Comparison: "
                            f"{months[previous_index]} → "
                            f"{selected_month}"
                        )

                    if st.button(
                        "🔬 Find Root Cause",
                        type="primary"
                    ):

                        with st.spinner(
                            "Analyzing business drivers..."
                        ):

                            try:

                                result = analyze_root_cause(
                                    df,
                                    selected_month
                                )

                            except Exception as e:

                                result = {
                                    "success": False,
                                    "message": str(e)
                                }

                        if not result.get("success"):

                            st.error(
                                result.get(
                                    "message",
                                    "Root-cause analysis failed."
                                )
                            )

                        else:

                            previous_month = result[
                                "previous_month"
                            ]

                            current_month = result[
                                "current_month"
                            ]

                            st.success(
                                f"Analysis completed: "
                                f"{previous_month} → "
                                f"{current_month}"
                            )

                            st.subheader(
                                "📊 Business Performance"
                            )

                            c1, c2, c3, c4 = st.columns(4)

                            revenue_pct = result[
                                "revenue_change_pct"
                            ]

                            quantity_pct = result[
                                "quantity_change_pct"
                            ]

                            transaction_pct = result[
                                "transaction_change_pct"
                            ]

                            aov_pct = result[
                                "aov_change_pct"
                            ]

                            with c1:

                                st.metric(
                                    "Revenue",
                                    money(
                                        result[
                                            "current_revenue"
                                        ]
                                    ),
                                    f"{revenue_pct:+.1f}%"
                                )

                            with c2:

                                st.metric(
                                    "Quantity",
                                    f"{result['current_quantity']:,.0f}",
                                    f"{quantity_pct:+.1f}%"
                                )

                            with c3:

                                st.metric(
                                    "Transactions",
                                    f"{result['current_transactions']:,.0f}",
                                    f"{transaction_pct:+.1f}%"
                                )

                            with c4:

                                st.metric(
                                    "AOV",
                                    money(
                                        result[
                                            "current_aov"
                                        ]
                                    ),
                                    f"{aov_pct:+.1f}%"
                                )

                            st.divider()

                            st.subheader(
                                "🧠 Root-Cause Explanation"
                            )

                            explanations = result.get(
                                "explanations",
                                []
                            )

                            if explanations:

                                for explanation in explanations:

                                    st.info(
                                        explanation
                                    )

                            else:

                                st.info(
                                    "No strong driver was detected."
                                )

                            st.subheader(
                                "🎯 Detected Drivers"
                            )

                            drivers = result.get(
                                "drivers",
                                pd.DataFrame()
                            )

                            if (
                                isinstance(
                                    drivers,
                                    pd.DataFrame
                                )
                                and
                                not drivers.empty
                            ):

                                st.dataframe(
                                    drivers,
                                    use_container_width=True
                                )

                            else:

                                st.info(
                                    "No major driver crossed "
                                    "the detection threshold."
                                )

                            st.divider()

                            st.subheader(
                                "📈 Monthly Revenue Trend"
                            )

                            monthly = result.get(
                                "monthly",
                                pd.DataFrame()
                            )

                            if (
                                isinstance(
                                    monthly,
                                    pd.DataFrame
                                )
                                and
                                not monthly.empty
                            ):

                                trend_chart = px.line(
                                    monthly,
                                    x="_rc_month",
                                    y="Revenue",
                                    markers=True,
                                    title="Monthly Revenue"
                                )

                                trend_chart.update_layout(
                                    xaxis_title="Month",
                                    yaxis_title="Revenue"
                                )

                                st.plotly_chart(
                                    trend_chart,
                                    use_container_width=True
                                )

                            st.divider()

                            st.subheader(
                                "🏆 Product-Level Drivers"
                            )

                            negative_products = result.get(
                                "top_negative_products",
                                pd.DataFrame()
                            )

                            positive_products = result.get(
                                "top_positive_products",
                                pd.DataFrame()
                            )

                            tab1, tab2 = st.tabs(
                                [
                                    "📉 Negative Drivers",
                                    "📈 Positive Drivers"
                                ]
                            )

                            with tab1:

                                if (
                                    isinstance(
                                        negative_products,
                                        pd.DataFrame
                                    )
                                    and
                                    not negative_products.empty
                                ):

                                    st.dataframe(
                                        negative_products,
                                        use_container_width=True
                                    )

                                    if (
                                        "Revenue Change"
                                        in negative_products.columns
                                    ):

                                        product_col = (
                                            negative_products.columns[0]
                                        )

                                        chart = px.bar(
                                            negative_products.head(10),
                                            x="Revenue Change",
                                            y=product_col,
                                            orientation="h",
                                            title=(
                                                "Largest Product "
                                                "Revenue Declines"
                                            )
                                        )

                                        st.plotly_chart(
                                            chart,
                                            use_container_width=True
                                        )

                                else:

                                    st.info(
                                        "No negative product "
                                        "drivers found."
                                    )

                            with tab2:

                                if (
                                    isinstance(
                                        positive_products,
                                        pd.DataFrame
                                    )
                                    and
                                    not positive_products.empty
                                ):

                                    st.dataframe(
                                        positive_products,
                                        use_container_width=True
                                    )

                                    if (
                                        "Revenue Change"
                                        in positive_products.columns
                                    ):

                                        product_col = (
                                            positive_products.columns[0]
                                        )

                                        chart = px.bar(
                                            positive_products.head(10),
                                            x="Revenue Change",
                                            y=product_col,
                                            orientation="h",
                                            title=(
                                                "Largest Product "
                                                "Revenue Increases"
                                            )
                                        )

                                        st.plotly_chart(
                                            chart,
                                            use_container_width=True
                                        )

                                else:

                                    st.info(
                                        "No positive product "
                                        "drivers found."
                                    )

                            st.divider()

                            st.subheader(
                                "🌍 Country-Level Drivers"
                            )

                            negative_countries = result.get(
                                "top_negative_countries",
                                pd.DataFrame()
                            )

                            if (
                                isinstance(
                                    negative_countries,
                                    pd.DataFrame
                                )
                                and
                                not negative_countries.empty
                            ):

                                st.dataframe(
                                    negative_countries,
                                    use_container_width=True
                                )

                                if (
                                    "Revenue Change"
                                    in negative_countries.columns
                                ):

                                    country_col = (
                                        negative_countries.columns[0]
                                    )

                                    chart = px.bar(
                                        negative_countries.head(10),
                                        x="Revenue Change",
                                        y=country_col,
                                        orientation="h",
                                        title=(
                                            "Largest Country "
                                            "Revenue Declines"
                                        )
                                    )

                                    st.plotly_chart(
                                        chart,
                                        use_container_width=True
                                    )

                            else:

                                st.info(
                                    "Country-level analysis "
                                    "is unavailable for this dataset."
                                )


# =========================================================
# DATA PROFILE
# =========================================================

elif page == "🔍 Data Profile":

    st.title(
        "🔍 Data Profile"
    )

    if require_data():

        df = st.session_state.df

        st.subheader(
            "Column Profile"
        )

        profile = dataset_profile(
            df
        )

        st.dataframe(
            profile,
            use_container_width=True
        )

        st.subheader(
            "Numeric Summary"
        )

        numeric = numeric_summary(
            df
        )

        if not numeric.empty:

            st.dataframe(
                numeric,
                use_container_width=True
            )

        st.subheader(
            "Categorical Summary"
        )

        categorical = categorical_summary(
            df
        )

        if not categorical.empty:

            st.dataframe(
                categorical,
                use_container_width=True
            )

        st.subheader(
            "Outlier Detection"
        )

        outliers = detect_outliers(
            df
        )

        if not outliers.empty:

            st.dataframe(
                outliers,
                use_container_width=True
            )


# =========================================================
# DATA PREVIEW
# =========================================================

elif page == "📈 Data Preview":

    st.title(
        "📈 Data Preview"
    )

    if require_data():

        df = st.session_state.df

        max_rows = len(
            df
        )

        if max_rows > 0:

            max_slider = min(
                max_rows,
                1000
            )

            default_rows = min(
                100,
                max_slider
            )

            preview_rows = st.slider(
                "Rows to display",
                min_value=1,
                max_value=max_slider,
                value=default_rows
            )

            st.dataframe(
                df.head(
                    preview_rows
                ),
                use_container_width=True
            )

            st.write(
                f"Showing first {preview_rows:,} "
                f"of {max_rows:,} rows."
            )

        else:

            st.warning(
                "The uploaded dataset is empty."
            )