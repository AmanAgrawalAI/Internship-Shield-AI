
import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import to_rgb
from pathlib import Path


# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Internship Shield AI",
    page_icon="🛡️",
    layout="wide"
)


# =====================================
# 2. LOAD ML MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():

    model = joblib.load(BASE_DIR / "internship_shield_model.pkl")
    tfidf = joblib.load(BASE_DIR / "internship_shield_tfidf.pkl")

    return model, tfidf


model, tfidf = load_models()


LABEL_MAP = {
    0: "Legitimate",
    1: "Suspicious",
    2: "Fraudulent"
}


def class_name(c):

    """Model class (0/1/2 ya string) ko readable naam me badalta hai."""

    try:
        return LABEL_MAP.get(int(c), str(c))

    except (ValueError, TypeError):
        return str(c).strip().title()


# =====================================
# 3. FEE DETECTION
# =====================================

def detect_fee_signals(text):

    text = text.lower()

    fee_types = [
        "registration",
        "application",
        "processing",
        "training",
        "interview",
        "document verification",
        "certificate"
    ]

    positive_signals = []
    risk_signals = []
    waived_fees = set()

    # Detect multiple fee waivers

    fee_names = (
        r"registration|application|processing|training|"
        r"interview|document verification|certificate"
    )

    waived_pattern = (
        rf"\bno\s+(?:any\s+)?(?:{fee_names})\s+fee"
        rf"(?:\s+(?:or|and)\s+(?:{fee_names})\s+fee)*"
    )

    waived_matches = re.finditer(waived_pattern, text)

    for match in waived_matches:

        fees_found = re.findall(
            rf"({fee_names})\s+fee",
            match.group()
        )

        waived_fees.update(fees_found)

    # Check each fee type

    for fee in fee_types:

        no_fee_pattern = (
            rf"\bno\s+(?:any\s+)?{re.escape(fee)}\s+fee"
            rf"(?:\s+required)?\b"
        )

        no_fee_found = (
            bool(re.search(no_fee_pattern, text))
            or fee in waived_fees
        )

        if no_fee_found:

            positive_signals.append(f"No {fee} fee")

        else:

            risk_patterns = [
                rf"\b(?:must\s+)?pay\s+(?:a\s+|an\s+|the\s+)?{re.escape(fee)}\s+fee\b",
                rf"\b{re.escape(fee)}\s+fee\s+(?:is\s+)?(?:required|mandatory)\b"
            ]

            if any(re.search(p, text) for p in risk_patterns):

                risk_signals.append(
                    f"{fee.title()} fee demand"
                )

    # Security deposit detection

    if re.search(
        r"\b(?:pay|submit|deposit)\s+(?:a\s+|the\s+)?security deposit\b",
        text
    ):

        risk_signals.append("Security deposit demand")

    # Free internship / no payment

    if re.search(
        r"\b(no payment required|free internship|without any fee)\b",
        text
    ):

        positive_signals.append("No payment / Free internship")

    # Unusually high monthly payment

    salary_patterns = [
        r"(?:₹|rs\.?|inr)\s*(\d[\d,]*)\s*(?:per month|/month|monthly)?",
        r"(\d[\d,]*)\s*(?:rs\.?|inr)?\s*(?:per month|/month|monthly)"
    ]

    for pattern in salary_patterns:

        amounts = re.findall(pattern, text)

        for amount in amounts:

            if int(amount.replace(",", "")) >= 50000:

                risk_signals.append(
                    "Unusually high monthly internship payment"
                )

                break

        if "Unusually high monthly internship payment" in risk_signals:
            break

    # GUARANTEED SELECTION DETECTION - FIXED

    guaranteed_patterns = [
        r"\bguaranteed\s+(?:internship\s+)?selection\b",
        r"\bguaranteed\s+job\b",
        r"\b100%\s+(?:internship\s+)?selection\b",
        r"\bselection\s+guaranteed\b",
        r"\bwithout\s+interview\b",
        r"\bwe\s+guarantee\b.{0,60}\bselection\b"
    ]

    if any(
        re.search(pattern, text)
        for pattern in guaranteed_patterns
    ):

        risk_signals.append("Guaranteed selection claim")

    # Urgency pressure tactics

    if re.search(
        r"\b(limited seats|apply immediately|"
        r"apply now|urgent hiring|hurry up)\b",
        text
    ):

        risk_signals.append("Urgency pressure tactics")

    return {
        "Positive Signals": list(dict.fromkeys(positive_signals)),
        "Risk Signals": list(dict.fromkeys(risk_signals))
    }


# =====================================
# 4. COMPANY DETECTION
# =====================================

def detect_company_signals(text):

    text = text.lower()

    positive_signals = []
    risk_signals = []

    # 1. Personal email detection

    personal_domains = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "rediffmail.com"
    ]

    for domain in personal_domains:

        if re.search(
            rf"\b[a-z0-9._%+-]+@{re.escape(domain)}\b",
            text
        ):

            risk_signals.append(
                "Recruitment uses a personal email address"
            )

            break

    # 2. WhatsApp-only communication detection

    whatsapp_patterns = [
        r"\bwhatsapp\s+only\b",
        r"\bcontact\s+only\s+on\s+whatsapp\b",
        r"\bapply\s+only\s+through\s+whatsapp\b",
        r"\bwhatsapp\s+number\s+only\b",
        r"\b(?:contact|communicate|reach|message|apply)\b"
        r".{0,60}\bonly\s+(?:through|via|on)\s+whatsapp\b",
        r"\bonly\s+(?:through|via|on)\s+whatsapp\b"
    ]

    if any(
        re.search(pattern, text)
        for pattern in whatsapp_patterns
    ):

        risk_signals.append(
            "WhatsApp-only communication"
        )

    # 3. Website and domain detection

    text_without_emails = re.sub(
        r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
        "",
        text
    )

    website_url_pattern = (
        r"\b(?:https?://|www\.)[^\s]+|"
        r"\b[a-z0-9-]+\.(?:com|in|org|net|co\.in)\b"
    )

    website_phrase_patterns = [
        r"\bofficial\s+company\s+website\b",
        r"\bcompany\s+website\b",
        r"\bofficial\s+website\b",
        r"\bvisit\s+(?:our|the)\s+website\b",
        r"\bapply\s+(?:through|via|on)\s+"
        r"(?:our|the|official)?\s*website\b",
        r"\bcareers?\s+page\b"
    ]

    website_found = (
        bool(re.search(website_url_pattern, text_without_emails))
        or any(
            re.search(pattern, text_without_emails)
            for pattern in website_phrase_patterns
        )
    )

    if website_found:

        positive_signals.append(
            "Website or domain mentioned"
        )

    # 4. Company registration reference

    if re.search(
        r"\b(cin|corporate identification number|"
        r"company registration number|registered under companies act)\b",
        text
    ):

        positive_signals.append(
            "Company registration reference mentioned"
        )

    return {
        "Positive Company Signals": positive_signals,
        "Company Risk Signals": risk_signals
    }


# =====================================
# 5. HYBRID AI ENGINE
# =====================================

def internship_shield(text):

    # ML prediction

    vector = tfidf.transform([text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    ml_result = class_name(prediction)

    ml_scores = {
        class_name(c): round(float(p) * 100, 2)
        for c, p in zip(model.classes_, probabilities)
    }

    # Fee detection

    fee_result = detect_fee_signals(text)

    fee_positive = fee_result["Positive Signals"]
    fee_risks = fee_result["Risk Signals"]

    # Company detection

    company_result = detect_company_signals(text)

    company_positive = company_result["Positive Company Signals"]
    company_risks = company_result["Company Risk Signals"]

    # Combine signals

    all_positive = fee_positive + company_positive
    all_risks = fee_risks + company_risks

    # Final assessment

    if ml_result == "Fraudulent" and all_risks:

        assessment = "High Risk - ML and Rule Checks Agree"
        level = "high"

    elif all_risks and all_positive:

        assessment = "Mixed Signals - Manual Verification Required"
        level = "medium"

    elif ml_result == "Legitimate" and all_risks:

        assessment = "Warning - ML Says Legitimate but Risk Signals Found"
        level = "medium"

    elif ml_result == "Suspicious" and all_risks:

        assessment = "Suspicious - Additional Verification Required"
        level = "medium"

    elif all_risks:

        assessment = "Risk Signals Found - Verify Independently"
        level = "medium"

    elif ml_result == "Fraudulent":

        assessment = "ML Flags Fraud - Further Verification Required"
        level = "high"

    elif ml_result == "Suspicious":

        assessment = "Suspicious - Further Verification Required"
        level = "medium"

    elif all_positive:

        assessment = "No Explicit Risk Found - Verify Independently"
        level = "low"

    else:

        assessment = "Insufficient Verification Evidence - Check Independently"
        level = "medium"

    return {
        "ML Prediction": ml_result,
        "ML Scores": ml_scores,
        "Fee Positive Signals": fee_positive,
        "Fee Risk Signals": fee_risks,
        "Company Positive Signals": company_positive,
        "Company Risk Signals": company_risks,
        "Final Risk Assessment": assessment,
        "Risk Level": level
    }


# =====================================
# 6. GRADIENT + SPLIT + SHADOW CHART
# =====================================

GRADIENTS = {
    "Legitimate": ("#8CF0B4", "#1E9E5A"),   # green
    "Suspicious": ("#A5DDFF", "#1E88E5"),   # sky blue
    "Fraudulent": ("#FF9A94", "#C0392B"),   # red
}

DEFAULT_GRADIENT = ("#CCCCCC", "#666666")

OUTER_IN, OUTER_OUT = 0.78, 1.00
INNER_OUT = 0.68


def blend(c1, c2, t):

    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))

    return tuple(a + (b - a) * t)


def gradient_wedge(
    ax, center, r_in, r_out, t1, t2, c1, c2, steps=40
):

    radii = np.linspace(r_in, r_out, steps + 1)

    for i in range(steps):

        color = blend(c1, c2, i / (steps - 1))

        width = min(
            radii[i + 1],
            radii[i + 1] - radii[i] + 0.003
        )

        ax.add_patch(
            Wedge(
                center,
                radii[i + 1],
                t1,
                t2,
                width=width,
                facecolor=color,
                edgecolor=color,
                linewidth=0.3
            )
        )


def shadow_wedge(
    ax, center, r_in, r_out, t1, t2, dx=0.03, dy=-0.04
):

    ax.add_patch(
        Wedge(
            (center[0] + dx, center[1] + dy),
            r_out,
            t1,
            t2,
            width=(r_out - r_in) if r_in > 0 else None,
            facecolor="black",
            edgecolor="none",
            alpha=0.25
        )
    )


def gradient_donut_pie(
    scores, predicted, base_explode=0.05, top_explode=0.10
):

    labels = list(scores.keys())

    values = np.array(list(scores.values()), dtype=float)

    fracs = values / values.sum()

    fig, ax = plt.subplots(figsize=(7, 7))

    fig.patch.set_facecolor("white")

    # Slice angles + split offset

    slices, cum = [], 0.0

    for lab, f in zip(labels, fracs):

        t2 = 90 - cum * 360
        t1 = 90 - (cum + f) * 360

        mid = np.deg2rad((t1 + t2) / 2)

        explode = top_explode if lab == predicted else base_explode

        center = (
            explode * np.cos(mid),
            explode * np.sin(mid)
        )

        slices.append((lab, f, t1, t2, mid, center))

        cum += f

    # 1) Shadows

    for lab, f, t1, t2, mid, center in slices:

        shadow_wedge(
            ax, center, OUTER_IN, OUTER_OUT, t1, t2
        )

        shadow_wedge(
            ax, center, 0, INNER_OUT, t1, t2
        )

    # 2) Gradient slices

    for lab, f, t1, t2, mid, center in slices:

        c1, c2 = GRADIENTS.get(lab, DEFAULT_GRADIENT)

        gradient_wedge(
            ax, center, OUTER_IN, OUTER_OUT,
            t1, t2, c1, c2
        )

        gradient_wedge(
            ax, center, 0.0, INNER_OUT,
            t1, t2, c1, c2
        )

        for r_in, r_out in [
            (OUTER_IN, OUTER_OUT),
            (0, INNER_OUT)
        ]:

            ax.add_patch(
                Wedge(
                    center,
                    r_out,
                    t1,
                    t2,
                    width=(r_out - r_in) if r_in > 0 else None,
                    fill=False,
                    edgecolor="white",
                    linewidth=2
                )
            )

    # 3) Texts

    for lab, f, t1, t2, mid, center in slices:

        cx, cy = np.cos(mid), np.sin(mid)

        pct = f"{f * 100:.1f}%"

        if f >= 0.05:

            ax.text(
                center[0] + 0.42 * cx,
                center[1] + 0.42 * cy,
                pct,
                ha="center",
                va="center",
                color="white",
                fontsize=13,
                fontweight="bold"
            )

            outer_text = lab

        else:

            outer_text = f"{lab}\n{pct}"

        ax.text(
            center[0] + 1.18 * cx,
            center[1] + 1.18 * cy,
            outer_text,
            ha="left" if cx >= 0 else "right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#222222"
        )

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.45, 1.45)

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "ML Prediction Score Distribution",
        fontsize=14,
        pad=10,
        color="#222222"
    )

    return fig


# =====================================
# 7. STREAMLIT WEBSITE DESIGN
# =====================================

st.title("🛡️ Internship Shield AI")

st.subheader("AI-Based Internship and Job Fraud Detection System")

st.write(
    "Analyze internship emails, job descriptions "
    "and recruitment messages for potential risk signals."
)

st.info("Paste the complete internship or job description below.")

user_text = st.text_area(
    "Internship / Job Description",
    height=250,
    placeholder="Paste your internship email or job description here..."
)


# =====================================
# 8. ANALYZE BUTTON
# =====================================

if st.button("🔍 Analyze Internship", type="primary"):

    if not user_text.strip():

        st.warning("Please enter an internship description.")

    else:

        with st.spinner("Analyzing your internship..."):

            result = internship_shield(user_text)

        st.divider()

        st.header("📊 Analysis Report")

        # ML PREDICTION

        st.subheader("ML Model Prediction")

        st.write(
            f"**Prediction:** {result['ML Prediction']}"
        )

        st.write("**ML Model Score Distribution:**")

        fig = gradient_donut_pie(
            result["ML Scores"],
            predicted=result["ML Prediction"]
        )

        st.pyplot(fig, use_container_width=False)

        plt.close(fig)

        # FINAL RISK ASSESSMENT

        st.subheader("Final Risk Assessment")

        if result["Risk Level"] == "high":

            st.error(result["Final Risk Assessment"])

        elif result["Risk Level"] == "low":

            st.success(result["Final Risk Assessment"])

        else:

            st.warning(result["Final Risk Assessment"])

        # FEE DETECTION

        st.subheader("💰 Fee Detection")

        st.write("**Positive Signals:**")

        if result["Fee Positive Signals"]:

            for signal in result["Fee Positive Signals"]:

                st.success(signal)

        else:

            st.write("No supported positive fee signals found.")

        st.write("**Risk Signals:**")

        if result["Fee Risk Signals"]:

            for signal in result["Fee Risk Signals"]:

                st.error(signal)

        else:

            st.write("No supported fee-demand patterns detected.")

        # COMPANY VERIFICATION

        st.subheader("🏢 Company Verification Signals")

        st.write("**Positive Company Signals:**")

        if result["Company Positive Signals"]:

            for signal in result["Company Positive Signals"]:

                st.success(signal)

        else:

            st.write("No supported positive company signals found.")

        st.write("**Company Risk Signals:**")

        if result["Company Risk Signals"]:

            for signal in result["Company Risk Signals"]:

                st.error(signal)

        else:

            st.write("No supported company risk patterns detected.")

        # DISCLAIMER

        st.divider()

        st.caption(
            "Disclaimer: This tool provides preliminary risk indicators, "
            "not a confirmed fraud verdict. Model scores are not calibrated "
            "fraud probabilities. Always verify the employer and recruitment "
            "details independently."
        )


# =====================================
# 9. FOOTER
# =====================================

st.divider()

st.caption(
    "Internship Shield AI | Machine Learning + Rule-Based Analysis"
)
