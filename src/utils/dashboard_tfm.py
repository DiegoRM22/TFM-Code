"""
Dashboard interactivo para visualización de métricas RAGAS del TFM.
Experimentos: RAG S2a, RAG Avanzado S2b, Híbrido S4, Híbrido Avanzado S4, FT, Baseline.
Lanzar con: streamlit run src/utils/dashboard_tfm.py
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

DATASETS = {
    "RAG S2a (14q)": {
        "path": os.path.join(BASE_DIR, "rag", "rag_evaluation_test_results.csv"),
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "group": "RAG Pipeline",
    },
    "RAG Avanzado S2b (14q)": {
        "path": os.path.join(BASE_DIR, "rag", "rag_advanced_evaluation_results.csv"),
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "group": "RAG Pipeline",
    },
    "Híbrido RAG+FT (14q)": {
        "path": os.path.join(BASE_DIR, "hybrid", "hybrid_evaluation_results.csv"),
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "group": "Híbrido",
    },
    "Híbrido Avanzado S4 (14q)": {
        "path": os.path.join(BASE_DIR, "hybrid", "hybrid_advanced_evaluation_results.csv"),
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "group": "Híbrido",
    },
    "Fine-tuning (14q)": {
        "path": os.path.join(BASE_DIR, "fine_tuning", "fine_tuning_evaluation_results.csv"),
        "metrics": ["answer_relevancy", "answer_correctness"],
        "group": "Fine-tuning",
    },
    "Baseline LLaMA-3 (14q)": {
        "path": os.path.join(BASE_DIR, "baseline", "baseline_evaluation_results.csv"),
        "metrics": ["answer_relevancy", "answer_correctness"],
        "group": "Baseline",
    },
}

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
    "answer_similarity": "Answer Similarity",
}

COLOR_MAP = {
    "RAG S2a (14q)": "#2196F3",
    "RAG Avanzado S2b (14q)": "#03A9F4",
    "Híbrido RAG+FT (14q)": "#FF9800",
    "Híbrido Avanzado S4 (14q)": "#FF5722",
    "Fine-tuning (14q)": "#9C27B0",
    "Baseline LLaMA-3 (14q)": "#F44336",
}


# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def load_all() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, cfg in DATASETS.items():
        try:
            df = pd.read_csv(cfg["path"])
            df["_experimento"] = name
            dfs[name] = df
        except FileNotFoundError:
            st.warning(f"Archivo no encontrado: {cfg['path']}")
    return dfs


def mean_metrics(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        cfg = DATASETS.get(name)
        if cfg is None:
            continue
        metrics = cfg["metrics"]
        row = {"Experimento": name, "Grupo": cfg["group"], "N": len(df)}
        for m in metrics:
            if m in df.columns:
                row[METRIC_LABELS[m]] = round(df[m].dropna().mean(), 4)
            else:
                row[METRIC_LABELS[m]] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ── Página: Resumen ───────────────────────────────────────────────────────────
def page_resumen(dfs: dict[str, pd.DataFrame]):
    st.header("Resumen de métricas por experimento")

    summary = mean_metrics(dfs)
    metric_cols = [c for c in summary.columns if c not in ("Experimento", "Grupo", "N")]

    st.dataframe(
        summary.set_index("Experimento").style.format(
            {c: "{:.4f}" for c in metric_cols if c in summary.columns}, na_rep="—"
        ).background_gradient(cmap="RdYlGn", subset=metric_cols, vmin=0, vmax=1),
        use_container_width=True,
    )

    st.subheader("Comparación por métrica")
    all_metrics = list(METRIC_LABELS.values())
    selected_metrics = st.multiselect(
        "Selecciona métricas", all_metrics, default=["Faithfulness", "Answer Relevancy"]
    )
    if not selected_metrics:
        st.info("Selecciona al menos una métrica.")
        return

    plot_data = summary.melt(
        id_vars=["Experimento", "N"], value_vars=selected_metrics,
        var_name="Métrica", value_name="Valor"
    ).dropna()

    fig = px.bar(
        plot_data, x="Métrica", y="Valor", color="Experimento",
        barmode="group", range_y=[0, 1],
        color_discrete_map=COLOR_MAP,
        labels={"Valor": "Media", "Métrica": ""},
        height=420,
    )
    fig.update_layout(legend_title_text="Experimento", plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)


# ── Página: Distribuciones ────────────────────────────────────────────────────
def page_distribuciones(dfs: dict[str, pd.DataFrame]):
    st.header("Distribución de métricas por experimento")

    exp_names = list(dfs.keys())
    selected_exps = st.multiselect("Experimentos", exp_names, default=exp_names[:2])
    if not selected_exps:
        st.info("Selecciona al menos un experimento.")
        return

    all_metrics_in_sel = set()
    for exp in selected_exps:
        all_metrics_in_sel.update(DATASETS.get(exp, {}).get("metrics", []))

    metric_raw = st.selectbox(
        "Métrica",
        [m for m in all_metrics_in_sel],
        format_func=lambda m: METRIC_LABELS.get(m, m),
    )

    plot_df = pd.concat(
        [dfs[exp][["_experimento", metric_raw]].dropna()
         for exp in selected_exps if metric_raw in dfs[exp].columns],
        ignore_index=True,
    )

    if plot_df.empty:
        st.warning("Sin datos para esta combinación.")
        return

    chart_type = st.radio("Tipo de gráfico", ["Box", "Violin", "Histograma"], horizontal=True)

    if chart_type == "Box":
        fig = px.box(
            plot_df, x="_experimento", y=metric_raw, color="_experimento",
            color_discrete_map=COLOR_MAP, points="all",
            labels={"_experimento": "Experimento", metric_raw: METRIC_LABELS[metric_raw]},
            height=420,
        )
    elif chart_type == "Violin":
        fig = px.violin(
            plot_df, x="_experimento", y=metric_raw, color="_experimento",
            color_discrete_map=COLOR_MAP, box=True, points="all",
            labels={"_experimento": "Experimento", metric_raw: METRIC_LABELS[metric_raw]},
            height=420,
        )
    else:
        fig = px.histogram(
            plot_df, x=metric_raw, color="_experimento",
            color_discrete_map=COLOR_MAP, nbins=20, barmode="overlay", opacity=0.7,
            labels={"_experimento": "Experimento", metric_raw: METRIC_LABELS[metric_raw]},
            height=420,
        )

    fig.update_layout(showlegend=True, plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    stats = (
        plot_df.groupby("_experimento")[metric_raw]
        .describe().round(4)
        .rename_axis("Experimento")
    )
    st.dataframe(stats, use_container_width=True)


# ── Página: Detalle por pregunta ──────────────────────────────────────────────
def page_detalle(dfs: dict[str, pd.DataFrame]):
    st.header("Detalle por pregunta")

    exp = st.selectbox("Experimento", list(dfs.keys()))
    df = dfs[exp].copy()
    metrics = DATASETS.get(exp, {}).get("metrics", [])
    available = [m for m in metrics if m in df.columns]

    display_cols = ["user_input"] + available
    display_df = df[display_cols].copy()
    display_df.columns = ["Pregunta"] + [METRIC_LABELS[m] for m in available]

    sort_col = st.selectbox("Ordenar por", [METRIC_LABELS[m] for m in available])
    display_df = display_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df.style.format(
            {METRIC_LABELS[m]: "{:.4f}" for m in available}, na_rep="—"
        ).background_gradient(
            cmap="RdYlGn",
            subset=[METRIC_LABELS[m] for m in available],
            vmin=0, vmax=1,
        ),
        use_container_width=True,
        height=400,
    )

    if len(available) >= 3:
        st.subheader("Radar — pregunta seleccionada")
        idx = st.number_input("Índice de fila (0-based)", 0, len(display_df) - 1, 0)
        row = display_df.iloc[idx]
        radar_vals = [row[METRIC_LABELS[m]] for m in available]
        cats = [METRIC_LABELS[m] for m in available]

        fig = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            line_color=COLOR_MAP.get(exp, "#2196F3"),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Pregunta:** {row['Pregunta'][:300]}")


# ── Página: Comparación todas arquitecturas ───────────────────────────────────
def page_comparacion(dfs: dict[str, pd.DataFrame]):
    st.header("Comparación de todas las arquitecturas")
    st.markdown(
        "**Answer Relevancy** como métrica común entre todos los experimentos. "
        "Baseline (LLaMA-3-8B sin LoRA) como pivote comparativo."
    )

    metric = "answer_relevancy"
    all_exps = [e for e in dfs if metric in dfs[e].columns]
    if not all_exps:
        st.warning("Sin datos de answer_relevancy.")
        return

    plot_df = pd.concat(
        [dfs[e][["_experimento", metric]].dropna() for e in all_exps],
        ignore_index=True,
    )

    fig = px.box(
        plot_df, x="_experimento", y=metric, color="_experimento",
        color_discrete_map=COLOR_MAP, points="all",
        labels={"_experimento": "Experimento", metric: "Answer Relevancy"},
        height=420,
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#eee", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    means = {e: dfs[e][metric].dropna().mean() for e in all_exps}
    cols = st.columns(len(means))
    for col, (exp, val) in zip(cols, means.items()):
        col.metric(exp, f"{val:.4f}")

    st.divider()
    st.subheader("Todas las métricas RAGAS — medias")
    summary = mean_metrics(dfs)
    st.dataframe(
        summary.drop(columns=["N"]).set_index("Experimento"),
        use_container_width=True,
    )


# ── Página: S2a vs S2b (chunking recursivo vs semántico) ─────────────────────
def page_s2a_vs_s2b(dfs: dict[str, pd.DataFrame]):
    st.header("S2a vs S2b — Chunking recursivo vs semántico")
    st.markdown(
        "Compara el impacto del **chunking semántico** (S2b) frente al "
        "**chunking recursivo** (S2a) en las métricas RAG. "
        "S2b añade además EnsembleRetriever + CrossEncoder reranking + HyDE."
    )

    s2a = "RAG S2a (14q)"
    s2b = "RAG Avanzado S2b (14q)"

    if s2a not in dfs or s2b not in dfs:
        st.warning("Faltan datos de S2a o S2b. Ejecuta primero los scripts de inferencia y evaluación.")
        return

    rag_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Medias lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RAG S2a — chunking recursivo")
        for m in rag_metrics:
            if m in dfs[s2a].columns:
                val = dfs[s2a][m].dropna().mean()
                st.metric(METRIC_LABELS[m], f"{val:.4f}")

    with col2:
        st.subheader("RAG Avanzado S2b — chunking semántico")
        for m in rag_metrics:
            if m in dfs[s2b].columns:
                val_s2b = dfs[s2b][m].dropna().mean()
                val_s2a = dfs[s2a][m].dropna().mean() if m in dfs[s2a].columns else None
                delta = f"{val_s2b - val_s2a:+.4f}" if val_s2a is not None else None
                st.metric(METRIC_LABELS[m], f"{val_s2b:.4f}", delta=delta)

    st.divider()

    # Gráfico comparativo
    plot_df = pd.concat(
        [dfs[e][["_experimento"] + [m for m in rag_metrics if m in dfs[e].columns]]
         for e in [s2a, s2b] if e in dfs],
        ignore_index=True,
    )
    summary_plot = (
        plot_df.groupby("_experimento")[rag_metrics]
        .mean()
        .round(4)
        .reset_index()
        .melt(id_vars="_experimento", var_name="Métrica", value_name="Valor")
    )
    summary_plot["Métrica"] = summary_plot["Métrica"].map(METRIC_LABELS)
    summary_plot.dropna(inplace=True)

    fig = px.bar(
        summary_plot, x="Métrica", y="Valor", color="_experimento",
        barmode="group", range_y=[0, 1],
        color_discrete_map=COLOR_MAP,
        labels={"Valor": "Media", "_experimento": "Experimento", "Métrica": ""},
        height=400,
    )
    fig.update_layout(legend_title_text="", plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    # Distribuciones por métrica
    st.subheader("Distribución por métrica")
    metric_sel = st.selectbox(
        "Métrica",
        rag_metrics,
        format_func=lambda m: METRIC_LABELS.get(m, m),
    )
    dist_df = pd.concat(
        [dfs[e][["_experimento", metric_sel]].dropna()
         for e in [s2a, s2b] if e in dfs and metric_sel in dfs[e].columns],
        ignore_index=True,
    )
    fig2 = px.box(
        dist_df, x="_experimento", y=metric_sel, color="_experimento",
        color_discrete_map=COLOR_MAP, points="all",
        labels={"_experimento": "", metric_sel: METRIC_LABELS[metric_sel]},
        height=380,
    )
    fig2.update_layout(showlegend=False, plot_bgcolor="white")
    fig2.update_yaxes(gridcolor="#eee", range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TFM — RAGAS Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Dashboard de Evaluación — TFM RGPD RAG")
    st.markdown(
        "Visualización interactiva de métricas **RAGAS** para los experimentos "
        "de RAG pipeline y fine-tuning sobre documentación del RGPD."
    )

    load_all.clear()
    dfs = load_all()
    if not dfs:
        st.error("No se pudieron cargar los datos.")
        return

    pages = {
        "Resumen": page_resumen,
        "Distribuciones": page_distribuciones,
        "Detalle por pregunta": page_detalle,
        "Todas las arquitecturas": page_comparacion,
        "S2a vs S2b": page_s2a_vs_s2b,
    }

    with st.sidebar:
        st.header("Navegación")
        page = st.radio("Sección", list(pages.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("Experimentos cargados:")
        for name, df in dfs.items():
            grp = DATASETS.get(name, {}).get("group", "—")
            st.caption(f"• **{name}** ({len(df)} preguntas) — {grp}")

    pages[page](dfs)


if __name__ == "__main__":
    main()
