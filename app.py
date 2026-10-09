"""SentimentAI: local customer review intelligence dashboard."""

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.prediction import LABELS, load_model, predict_reviews

st.set_page_config(page_title="SentimentAI | Review Intelligence", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")

COLORS = {"Positive": "#16a879", "Negative": "#e05261", "Neutral": "#e8a52b"}
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#18213a; --muted:#768098; --purple:#7357e8; }
.stApp { background:#f6f7fb; color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
.block-container { padding-top:2rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#131a30; }
[data-testid="stSidebar"] * { color:#eef0fa !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.3rem 0; }
[data-testid="stMetric"] { background:white; padding:1.1rem 1.2rem; border:1px solid #eceef4; border-radius:16px; box-shadow:0 5px 22px #25315408; }
[data-testid="stMetricLabel"] { color:#768098 !important; }
[data-testid="stMetricValue"] { font-family:'Manrope',sans-serif; color:#18213a; }
.hero { background:linear-gradient(115deg,#171f3b,#393274 58%,#6654c8); border-radius:22px; padding:2.1rem 2.3rem; color:#fff; margin-bottom:1.4rem; box-shadow:0 14px 35px #34336a24; }
.hero h1 { color:white !important; margin:0 0 .45rem 0; font-size:2.1rem; }
.hero p { color:#e2e4fa; margin:0; font-size:1.02rem; }
.eyebrow { color:#bdb4ff; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; margin-bottom:.6rem; }
.section-title { font:700 1.15rem 'Manrope',sans-serif; margin:1.3rem 0 .8rem; }
.soft-card { background:white; border:1px solid #eceef4; border-radius:16px; padding:1.25rem; box-shadow:0 5px 22px #25315408; }
.small-muted { color:#768098; font-size:.91rem; }
.stButton > button[kind="primary"] { background:linear-gradient(105deg,#7055e8,#4d75e8); border:0; border-radius:10px; font-weight:650; }
.stButton > button { border-radius:10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading your local model…")
def get_artifact():
    return load_model()


def init_state():
    st.session_state.setdefault("results", pd.DataFrame(columns=["Row", "Review", "Predicted sentiment", "Confidence"]))


def add_results(incoming: pd.DataFrame):
    old = st.session_state.results
    incoming = incoming.copy()
    incoming["Row"] = range(len(old) + 1, len(old) + len(incoming) + 1)
    st.session_state.results = pd.concat([old, incoming], ignore_index=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">CUSTOMER REVIEW INTELLIGENCE</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def sentiment_counts(results: pd.DataFrame) -> pd.Series:
    return results["Predicted sentiment"].value_counts().reindex(LABELS, fill_value=0)


def results_table(results: pd.DataFrame):
    if results.empty:
        st.info("No reviews yet. Analyze a review to see results here.")
    else:
        st.dataframe(results, use_container_width=True, hide_index=True,
                     column_config={"Confidence": st.column_config.TextColumn("Confidence", help="Model probability for its selected label. It is not a guarantee of correctness."),
                                    "Review": st.column_config.TextColumn("Review", width="large")})


init_state()
with st.sidebar:
    st.markdown("<div style='font:800 1.35rem Manrope,sans-serif; padding:.4rem 0 1.4rem'>✦ SentimentAI</div>", unsafe_allow_html=True)
    pages = ["Home dashboard", "Analyze reviews", "Visualizations", "Review results", "About & model"]
    if st.session_state.pop("navigate_to_analyze", False):
        st.session_state.page_navigation = "Analyze reviews"
    page = st.radio("Workspace", pages, label_visibility="collapsed", key="page_navigation")
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.caption("LOCAL ML WORKSPACE")
    try:
        artifact = get_artifact()
        st.success("Model ready")
    except (FileNotFoundError, ValueError, OSError) as exc:
        artifact = None
        st.error("Model not found. From the project folder, run `python train_model.py`.")
        st.caption(str(exc))

data = st.session_state.results
counts = sentiment_counts(data)

if page == "Home dashboard":
    hero("Understand every customer voice.", "Analyze feedback locally with a transparent machine learning workflow.")
    a, b, c, d = st.columns(4)
    a.metric("Reviews analyzed", f"{len(data):,}")
    b.metric("Positive", f"{counts['Positive']:,}")
    c.metric("Negative", f"{counts['Negative']:,}")
    d.metric("Neutral", f"{counts['Neutral']:,}")
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        st.markdown("<div class='section-title'>Sentiment distribution</div>", unsafe_allow_html=True)
        if data.empty:
            st.markdown("<div class='soft-card small-muted'>Your workspace is ready. Analyze reviews to see the sentiment mix.</div>", unsafe_allow_html=True)
        else:
            fig = px.pie(values=counts.values, names=counts.index, hole=.68,
                         color=counts.index, color_discrete_map=COLORS)
            fig.update_traces(textinfo="percent", textfont_size=13)
            fig.update_layout(margin=dict(t=12,b=12,l=8,r=8), height=330, legend_title=None,
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#526079")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("<div class='section-title'>Recent analyses</div>", unsafe_allow_html=True)
        if data.empty:
            st.markdown("<div class='soft-card small-muted'>No reviews analyzed yet. Start with a single review or upload a CSV file.</div>", unsafe_allow_html=True)
        else:
            results_table(data.tail(5).iloc[::-1])
    if st.button("✦  Analyze reviews", type="primary", key="home_analyze"):
        st.session_state.navigate_to_analyze = True
        st.rerun()
    st.caption("Prediction confidence is a model probability, not a guarantee that the label is correct.")

elif page == "Analyze reviews":
    hero("Analyze reviews", "Paste customer feedback or bring a CSV to classify reviews in batches.")
    if not artifact:
        st.warning("Train the model first with `python train_model.py` from the project folder.")
    typed, bulk = st.tabs(["✍️  Write reviews", "📄  Upload CSV"])
    with typed:
        review_text = st.text_area("One review per line", height=190, key="typed_reviews", placeholder="The product arrived quickly and works perfectly.\nThe battery stopped working after a week.", help="Blank lines are ignored; each remaining line is analyzed separately.")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            analyze = st.button("Analyze sentiment", type="primary", disabled=not bool(artifact), use_container_width=True)
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.typed_reviews = ""
                st.rerun()
        if analyze:
            reviews = [line.strip() for line in review_text.splitlines() if line.strip()]
            if not reviews:
                st.error("Enter at least one non-empty review.")
            else:
                with st.spinner(f"Analyzing {len(reviews)} review(s)…"):
                    try:
                        predictions = predict_reviews(reviews, artifact).drop(columns="Row")
                        add_results(predictions)
                        st.success(f"Analyzed {len(reviews)} review(s).")
                        results_table(st.session_state.results.tail(len(reviews)))
                    except (ValueError, TypeError) as exc:
                        st.error(f"Could not analyze these reviews: {exc}")
    with bulk:
        st.markdown("Upload a CSV with a **text** column. Other columns are kept out of model inputs.")
        uploaded = st.file_uploader("Choose a review CSV", type=["csv"], key="csv_upload")
        if uploaded:
            try:
                uploaded_data = pd.read_csv(BytesIO(uploaded.getvalue()))
                if "text" not in uploaded_data.columns:
                    st.error("CSV needs a column named `text` containing review strings.")
                else:
                    st.caption(f"Found {len(uploaded_data):,} rows. Preview:")
                    st.dataframe(uploaded_data.head(5), use_container_width=True, hide_index=True)
                    if st.button("Analyze uploaded reviews", type="primary", disabled=not bool(artifact)):
                        valid = uploaded_data["text"].dropna().astype(str).str.strip()
                        valid = valid[valid != ""]
                        if valid.empty:
                            st.error("The CSV has no non-empty reviews in its `text` column.")
                        else:
                            try:
                                with st.spinner(f"Analyzing {len(valid):,} review(s)…"):
                                    predictions = predict_reviews(valid.tolist(), artifact).drop(columns="Row")
                                    add_results(predictions)
                                st.success(f"Analyzed {len(valid):,} review(s). Empty rows were skipped.")
                                results_table(st.session_state.results.tail(len(valid)))
                            except (ValueError, TypeError) as exc:
                                st.error(f"Could not analyze this file: {exc}")
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                st.error(f"Could not read this CSV. Check its encoding and comma-separated format. Details: {exc}")

elif page == "Visualizations":
    hero("Sentiment at a glance", "Explore the distribution and the language appearing in your analyzed reviews.")
    if data.empty:
        st.info("Visualizations will appear after reviews have been analyzed in this session.")
    else:
        selection = st.multiselect("Filter review table by sentiment", LABELS, default=list(LABELS))
        filtered = data[data["Predicted sentiment"].isin(selection)]
        total = len(data)
        values = sentiment_counts(data)
        k1, k2, k3 = st.columns(3)
        for column, label in zip((k1, k2, k3), LABELS):
            pct = values[label] / total * 100 if total else 0
            column.metric(f"{label} share", f"{pct:.1f}%", f"{values[label]} reviews")
        left, right = st.columns(2, gap="large")
        with left:
            fig = px.pie(values=values.values, names=values.index, hole=.60, color=values.index, color_discrete_map=COLORS, title="Overall distribution")
            fig.update_layout(margin=dict(t=50,b=12,l=8,r=8), paper_bgcolor="rgba(0,0,0,0)", legend_title=None)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            bar = px.bar(x=values.index, y=values.values, color=values.index, color_discrete_map=COLORS, title="Review count by sentiment", labels={"x":"Sentiment", "y":"Reviews"})
            bar.update_layout(showlegend=False, margin=dict(t=50,b=12,l=8,r=8), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, use_container_width=True)
        st.markdown("<div class='section-title'>Most common terms</div><div class='small-muted'>Counted from reviews analyzed in this session, after basic token cleanup.</div>", unsafe_allow_html=True)
        stop = {"the","and","for","with","this","that","was","are","but","not","have","has","had","from","very","its","it","you","would","after","all","one","too","been","my","i","a","an","to","of","in","on","is","as","at","be","by","or","if","no","so","do","did","they","their","we","our","me","your","than","then","when","what","which","into","about","up","out","more","less"}
        from collections import Counter
        import re
        def terms_for(label):
            text = " ".join(data.loc[data["Predicted sentiment"] == label, "Review"].astype(str)).lower()
            words = [w for w in re.findall(r"[a-z']{2,}", text) if w not in stop]
            return Counter(words).most_common(10)
        term_col1, term_col2 = st.columns(2)
        for col, label in ((term_col1, "Positive"), (term_col2, "Negative")):
            with col:
                st.markdown(f"**{label} reviews**")
                terms = terms_for(label)
                if terms:
                    st.dataframe(pd.DataFrame(terms, columns=["Term", "Occurrences"]), use_container_width=True, hide_index=True)
                else:
                    st.caption(f"No {label.lower()} reviews to summarize yet.")
        st.markdown("<div class='section-title'>Filtered reviews</div>", unsafe_allow_html=True)
        results_table(filtered)

elif page == "Review results":
    hero("Review results", "Browse your current session and export a portable CSV report.")
    if data.empty:
        st.markdown("<div class='soft-card small-muted'>No results yet. Analyze reviews to build your report.</div>", unsafe_allow_html=True)
    else:
        filt = st.multiselect("Show sentiments", LABELS, default=list(LABELS))
        visible = data[data["Predicted sentiment"].isin(filt)]
        st.caption(f"Showing {len(visible):,} of {len(data):,} analyzed reviews.")
        results_table(visible)
        csv = visible.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download results as CSV", csv, file_name="sentimentai_results.csv", mime="text/csv", type="primary")
        if st.button("Clear session results"):
            st.session_state.results = pd.DataFrame(columns=data.columns)
            st.rerun()

elif page == "About & model":
    hero("About the model", "A transparent, fully local baseline that is easy to inspect and retrain.")
    st.markdown("### How it works")
    st.write("Review text is normalized and lightly stemmed with NLTK's English Snowball stemmer. We retain stopwords, including negation words such as *not*, *never*, and *no*. TF-IDF represents words and adjacent word pairs as weighted numeric features, then Logistic Regression predicts a sentiment class.")
    st.write("The included `data/sample_reviews.csv` is a small, hand-written teaching sample. It has no external dataset license; it is supplied for demonstration only and should not be treated as representative customer data or as a benchmark. Replace it with a properly licensed dataset for substantive evaluation.")
    st.markdown("### Training and evaluation")
    if artifact:
        m = artifact.get("test_metrics", {})
        st.caption(f"Model file trained from: `{artifact.get('data_source', 'unknown')}` · {artifact.get('training_rows', '—')} training rows · {artifact.get('test_rows', '—')} held-out test rows · stratified split: {artifact.get('stratified_split', False)}")
        if m:
            st.metric("Held-out accuracy", f"{m['accuracy']:.1%}")
            report = m.get("report", {})
            rows = [{"Sentiment": label, "Precision": report.get(label, {}).get("precision", 0), "Recall": report.get(label, {}).get("recall", 0), "F1 score": report.get(label, {}).get("f1-score", 0), "Test support": report.get(label, {}).get("support", 0)} for label in LABELS]
            st.dataframe(pd.DataFrame(rows).style.format({"Precision":"{:.2f}","Recall":"{:.2f}","F1 score":"{:.2f}"}), use_container_width=True, hide_index=True)
            st.caption("These metrics are calculated from the actual held-out predictions saved by train_model.py. Results on the included tiny sample are illustrative only.")
            cm = pd.DataFrame(m["confusion_matrix"], index=LABELS, columns=LABELS)
            st.markdown("**Confusion matrix** · rows are actual labels; columns are predictions")
            st.dataframe(cm, use_container_width=True)
    else:
        st.info("No trained model is available. Run `python train_model.py` to create one.")
    st.markdown("### Confidence and limitations")
    st.write("Confidence is the Logistic Regression model's highest class probability. It describes the model's relative preference and is not a guarantee of correctness or a calibrated measure of truth. This bag-of-words baseline can miss sarcasm, context, domain-specific language, and mixed sentiment. It predicts only classes observed during training; a missing class cannot be learned from absent examples.")
    st.markdown("### Data format")
    st.code("text,sentiment\n\"The delivery was fast and the product works perfectly\",Positive\n\"It stopped working after a day\",Negative\n\"I received the blue version\",Neutral", language="csv")
    st.write("Accepted sentiment labels are Positive, Negative, and Neutral (case-insensitive). The trainer drops blank and duplicate rows, ignores invalid labels, and requires at least two classes.")

st.markdown("<div style='height:1rem'></div><div class='small-muted' style='text-align:center'>SentimentAI · Local machine learning for customer feedback</div>", unsafe_allow_html=True)
