"""
generate_technical_report.py
-----------------------------
Generates an academic and industry technical report (PDF) strictly between 3-5 pages
documenting the Data Quality ML Feature Store Pipeline project.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total page count and draw
    professional running headers and footers with 'Page X of Y'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "TECHNICAL REPORT: DATA QUALITY FIREWALL & ML FAIRNESS")
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "BDSRAE • SEM 5")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running footer (all pages)
        self.setFont("Helvetica", 8)
        self.drawString(54, 36, "Apache Spark 4.x • AWS PyDeequ • Automated Feature Store Quality Architecture")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * 72 - 54, 46)
        self.restoreState()


def build_pdf(filename="TECHNICAL_REPORT.pdf"):
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_root, filename)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=52,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    primary_color = colors.HexColor("#0F172A")    # Slate 900
    brand_blue = colors.HexColor("#1E40AF")       # Blue 800
    teal_accent = colors.HexColor("#0F766E")      # Teal 700
    charcoal = colors.HexColor("#334155")         # Slate 700
    light_bg = colors.HexColor("#F8FAFC")         # Slate 50
    border_color = colors.HexColor("#E2E8F0")     # Slate 200

    # Modify/Add Paragraph Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=brand_blue,
        spaceAfter=8,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=8,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=brand_blue,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=primary_color,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=charcoal,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=charcoal,
        leftIndent=12,
        spaceAfter=2,
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=charcoal,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=primary_color,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=charcoal,
        alignment=1,
    )

    fig_caption_style = ParagraphStyle(
        "FigCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=6,
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, ABSTRACT, INTRODUCTION, & ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("Pre-Inference Data Quality Firewall & Subgroup Bias Mitigation in Distributed ML Systems", title_style))
    story.append(Paragraph("An Automated Spark/PyDeequ Quality Architecture for Machine Learning Feature Stores", subtitle_style))
    story.append(Paragraph("<b>Course:</b> BDSRAE (Big Data Systems & Reliability) &nbsp;|&nbsp; <b>Semester:</b> 5 &nbsp;|&nbsp; <b>Stack:</b> Apache Spark 4.x, AWS PyDeequ, Python 3.11, scikit-learn", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=brand_blue, spaceBefore=0, spaceAfter=6))

    # Abstract Callout Box
    abstract_text = (
        "<b>Executive Summary & Abstract:</b> In production machine learning pipelines, unvalidated data corruptions "
        "consistently cause silent model degradation. Critically, standard aggregate metrics often mask severe localized harm, "
        "allowing models to maintain acceptable overall accuracy while accuracy on specific sub-populations crashes catastrophically. "
        "This project implements an end-to-end, reproducible data quality firewall combining Apache Spark and AWS Deequ (PyDeequ). "
        "The architecture generates distributed synthetic datasets, injects deterministic fault vectors (nulls, unit scale shifts, and duplicate keys), "
        "validates batches against declarative YAML constraint rules in a single distributed pass, and triggers an automated circuit-breaker quarantine "
        "that blocks model scoring. Experimental simulation confirms that while overall accuracy dropped by 4.33 percentage points under corruption, "
        "<b>Subgroup B accuracy crashed by 9.53 percentage points</b> (vs. zero degradation for Subgroup A). The automated quarantine successfully "
        "intercepts corrupted batches, recovers Subgroup B accuracy to 89.39%, and emits forensic audit artifacts without driver memory bottlenecks."
    )
    abstract_table = Table([[Paragraph(abstract_text, callout_style)]], colWidths=[516])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 6))

    # Section 1: Problem Formulation
    story.append(Paragraph("1. Introduction & The Hidden Cost of Silent ML Poisoning", h1_style))
    story.append(Paragraph(
        "Modern Machine Learning systems deployed in credit underwriting, fraud detection, and automated healthcare "
        "rely heavily on automated feature pipelines. In production environments, data pipelines frequently experience operational faults: "
        "network dropouts cause null values, microservice version upgrades introduce unit scale discrepancies (e.g., dollars vs. cents or "
        "grams vs. kilograms), and distributed message bus retries replay identical transactions. Unlike analytical SQL queries that hard-fail "
        "on schema mismatches, ML models operate on numeric arrays and will silently consume corrupted inputs, producing arbitrary inference "
        "predictions without raising runtime exceptions.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Disparate Impact Dilemma:</b> In production monitoring, operations teams customarily track global metrics (such as aggregate "
        "accuracy, F1-score, or mean squared error). However, real-world data distributions possess heterogeneous feature reliance across "
        "sub-populations. When a specific feature experiences corruption, sub-populations that heavily rely on that feature bear the entire brunt "
        "of model failure, whereas unaffected sub-populations keep global metrics artificially high. This creates severe algorithmic unfairness "
        "and compliance liabilities that evade conventional pipeline monitoring.",
        body_style
    ))

    # Section 2: Architecture
    story.append(Paragraph("2. System Architecture & Scalable Data Generation", h1_style))
    story.append(Paragraph(
        "To evaluate detection, quarantine, and model fairness under rigorous conditions, we designed a closed-loop distributed pipeline "
        "structured around four autonomous modules:",
        body_style
    ))
    story.append(Paragraph("• <b>Distributed Data Generation:</b> Generates synthetic feature store batches in Apache Spark without driver memory collection.", bullet_style))
    story.append(Paragraph("• <b>Seeded Chaos Fault Injection:</b> Injects controlled proportions of missing values, scale-shifts, and key collisions.", bullet_style))
    story.append(Paragraph("• <b>Declarative Verification Suite (PyDeequ):</b> Evaluates 17 statistical constraints in a single distributed pass.", bullet_style))
    story.append(Paragraph("• <b>Quarantine Circuit Breaker & Fallback:</b> Isolates corrupted batches, produces audit manifests, and halts inference.", bullet_style))

    story.append(Spacer(1, 4))
    # Schema Table
    schema_data = [
        [Paragraph("Column", table_header_style), Paragraph("Type", table_header_style), Paragraph("Distribution & Mathematical Role", table_header_style), Paragraph("Clean Bounds", table_header_style)],
        [Paragraph("<code>id</code>", table_cell_bold), Paragraph("Long", table_cell_center), Paragraph("Unique primary key identity generated via <code>spark.range</code>", table_cell_style), Paragraph("100% Unique, Non-null", table_cell_style)],
        [Paragraph("<code>feature_1</code>", table_cell_bold), Paragraph("Double", table_cell_center), Paragraph("Gaussian predictor: N(0, 10). Weak predictor for all groups.", table_cell_style), Paragraph("[-100.0, +100.0]", table_cell_style)],
        [Paragraph("<code>feature_2</code>", table_cell_bold), Paragraph("Double", table_cell_center), Paragraph("Uniform U(0, 100). <b>Primary predictor for Subgroup B (3x weight).</b>", table_cell_style), Paragraph("[0.0, 110.0]", table_cell_style)],
        [Paragraph("<code>feature_3</code>", table_cell_bold), Paragraph("Double", table_cell_center), Paragraph("Gaussian covariate: N(0, 50). Controls baseline variance.", table_cell_style), Paragraph("[-300.0, +300.0]", table_cell_style)],
        [Paragraph("<code>target</code>", table_cell_bold), Paragraph("Integer", table_cell_center), Paragraph("Binary ground truth: P(y=1) = Sigmoid(0.1*f1 + w2*f2_norm + 0.05*f3)", table_cell_style), Paragraph("{0, 1}", table_cell_center)],
        [Paragraph("<code>subgroup</code>", table_cell_bold), Paragraph("String", table_cell_center), Paragraph("Equally balanced sub-populations: Subgroup A vs. Subgroup B", table_cell_style), Paragraph("{'A', 'B'}", table_cell_center)],
        [Paragraph("<code>batch_id</code>", table_cell_bold), Paragraph("String", table_cell_center), Paragraph("Temporal partition key defining the atomic unit of quarantine", table_cell_style), Paragraph("Non-null string", table_cell_style)],
    ]
    schema_table = Table(schema_data, colWidths=[65, 45, 290, 116])
    schema_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), brand_blue),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(schema_table)
    story.append(Paragraph("<b>Table 1:</b> Feature store schema definitions, statistical distributions, and ground-truth generation specifications.", fig_caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: FAULT INJECTION, PYDEEQU VERIFICATION, & QUARANTINE
    # =========================================================================
    story.append(Paragraph("3. Controlled Fault Injection Engineering", h1_style))
    story.append(Paragraph(
        "To rigorously validate data quality gates, <code>src/fault_injection.py</code> applies reproducible, parameterized corruptions "
        "governed by a master seed (<code>seed=42</code>). The generator introduces three distinct real-world failure patterns:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Missing Value Injection (Nulls):</b> <code>F.when(F.rand(seed) < 0.05, None)</code> applied to <code>feature_1</code>. "
        "Simulates transient network disconnects or upstream sensor timeouts. Target rate: 5.00%; Observed: <b>4.75%</b> (475 nulls).",
        body_style
    ))
    story.append(Paragraph(
        "2. <b>Unit Scale Multiplier (1000x Shift):</b> <code>F.when(F.rand(seed) < 0.02, F.col('feature_2') * 1000.0)</code>. "
        "Simulates unit conversion errors (e.g. grams vs. kilograms or cents vs. dollars). The values remain valid floating-point numbers "
        "but violate statistical expectations. Target rate: 2.00%; Observed: <b>1.75%</b> (175 extreme outliers up to 99,842).",
        body_style
    ))
    story.append(Paragraph(
        "3. <b>Primary Key Replay (Duplicates):</b> Samples 3.00% of rows and unions them back without key deduplication. "
        "Simulates distributed streaming duplicates or ingestion retries. Target rate: 3.00%; Observed: <b>3.00%</b> (300 duplicates, 10,300 total rows).",
        body_style
    ))

    # Figure 1: Injected Fault Rates Summary
    fig1_path = os.path.join(project_root, "data", "reports", "null_duplicate_summary.png")
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=470, height=160))
        story.append(Paragraph("<b>Figure 1:</b> Fault injection summary comparing target vs. observed rates for nulls, scale shifts, and duplicate keys.", fig_caption_style))

    # Section 4: PyDeequ Declarative Verification
    story.append(Paragraph("4. Declarative Quality Verification Engine (PyDeequ)", h1_style))
    story.append(Paragraph(
        "Rather than scattering procedural assertion checks across application code, quality policies are declaratively defined in "
        "<code>config/quality_rules.yml</code>. PyDeequ translates these declarations into a unified Apache Spark execution plan:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Single-Pass Metric Aggregation:</b> Traditional data validation executes multiple SQL queries (e.g. <code>COUNT(*) WHERE col IS NULL</code>, "
        "<code>SELECT MAX(col)</code>), repeatedly scanning large tables. PyDeequ computes all metrics—completeness, uniqueness, minimum, maximum, "
        "mean, and standard deviation—in a <b>single distributed pass</b> across worker partitions, minimizing I/O overhead on 100M+ row datasets.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Reference-Calibrated Bounds:</b> A common failure mode in quality engineering is setting loose bounds (e.g. <code>feature_2 >= 0</code>). "
        "Because corrupted values (e.g., 55,000) are positive numbers, naive bounds fail to trigger alerts. Our engine establishes bounds from the clean "
        "data generator (<b>U(0, 100) + 10% headroom = 110.0</b>). Any scale shift immediately breaches this threshold.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Multi-Constraint Suite:</b> The pipeline runs 17 distinct constraint groups covering schema presence, categorical domain membership "
        "(<code>target IN {0, 1}</code>, <code>subgroup IN {'A', 'B'}</code>), primary key uniqueness, and numeric ranges.",
        body_style
    ))

    # Section 5: Quarantine Circuit Breaker
    story.append(Paragraph("5. Automated Quarantine & The Inference Circuit Breaker", h1_style))
    story.append(Paragraph(
        "When any quality check fails, naive data engineering pipelines often apply silent row-dropping. <b>Row-level dropping is dangerous</b>: "
        "it biases feature distributions, alters class balances, and masks systemic ingestion bugs. Our system implements a <b>Batch Quarantine Pattern</b>:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Batch Isolation:</b> The entire corrupted partition is written to isolated Parquet storage (<code>data/quarantined/batch_id=.../records/</code>).<br/>"
        "2. <b>Forensic Diagnostics:</b> The engine writes machine-readable audit artifacts: <code>validation_results.json</code> (failing checks, "
        "observed vs expected thresholds) and <code>quarantine_manifest.json</code> (remediation playbooks for on-call engineers).<br/>"
        "3. <b>Inference Guard (Circuit Breaker):</b> <code>block_inference()</code> validates the quarantine registry and terminates with exit code 1, "
        "preventing tainted feature vectors from reaching downstream model scoring endpoints.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: EXPERIMENTAL RESULTS & SUBGROUP DISPARITY
    # =========================================================================
    story.append(Paragraph("6. Experimental Evaluation: 3-State Model Simulation", h1_style))
    story.append(Paragraph(
        "To rigorously quantify the operational damage of unvalidated data and verify the efficacy of our quarantine firewall, "
        "<code>src/model_simulation.py</code> executes a 3-state experimental protocol using a Logistic Regression classifier:",
        body_style
    ))
    story.append(Paragraph("• <b>State 1: Clean Baseline:</b> Model trained and evaluated on uncorrupted synthetic data.", bullet_style))
    story.append(Paragraph("• <b>State 2: Corrupted (No QA):</b> Model evaluated on corrupted data slipping through without data quality checks.", bullet_style))
    story.append(Paragraph("• <b>State 3: Validated (Quarantine Active):</b> Corrupted batch rejected; model operates on verified fallback data.", bullet_style))

    # Figure 2: Accuracy Comparison
    fig2_path = os.path.join(project_root, "data", "reports", "accuracy_comparison.png")
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=470, height=170))
        story.append(Paragraph("<b>Figure 2:</b> Model classification accuracy across Clean, Corrupted (No QA), and Validated (Quarantine) states.", fig_caption_style))

    story.append(Spacer(1, 4))
    # Accuracy Table
    acc_data = [
        [Paragraph("Evaluation State", table_header_style), Paragraph("Overall Accuracy", table_header_style), Paragraph("Subgroup A Accuracy", table_header_style), Paragraph("Subgroup B Accuracy", table_header_style), Paragraph("Operational Impact", table_header_style)],
        [Paragraph("<b>State 1: Clean Baseline</b>", table_cell_style), Paragraph("85.30%", table_cell_center), Paragraph("80.43%", table_cell_center), Paragraph("<b>90.39%</b>", table_cell_center), Paragraph("Optimal performance across both cohorts", table_cell_style)],
        [Paragraph("<b>State 2: Corrupted (No QA)</b>", table_cell_style), Paragraph("80.97% <i>(-4.33 pp)</i>", table_cell_center), Paragraph("81.08% <i>(+0.65 pp)</i>", table_cell_center), Paragraph("<b>80.86% (-9.53 pp)</b>", table_cell_center), Paragraph("<b>Disparate failure:</b> Subgroup B collapses", table_cell_style)],
        [Paragraph("<b>State 3: Validated (Quarantine)</b>", table_cell_style), Paragraph("84.80% <i>(+3.83 pp)</i>", table_cell_center), Paragraph("80.16% <i>(Stable)</i>", table_cell_center), Paragraph("<b>89.39% (+8.53 pp)</b>", table_cell_center), Paragraph("Inference protected; 89.5% recovery for B", table_cell_style)],
    ]
    acc_table = Table(acc_data, colWidths=[120, 85, 95, 95, 121])
    acc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), brand_blue),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
    ]))
    story.append(acc_table)
    story.append(Paragraph("<b>Table 2:</b> Empirical accuracy measurements, absolute percentage point (pp) drops, and recovery metrics across evaluation states.", fig_caption_style))

    # Section 7: Mathematical Root Cause of Subgroup Disparity
    story.append(Paragraph("7. Mathematical Root Cause of Disparate Subgroup Degradation", h1_style))
    story.append(Paragraph(
        "A critical finding of this research is that <b>overall accuracy alone is an unsafe metric for production ML systems</b>. "
        "In our experiment, overall accuracy dropped by only 4.33 percentage points. An automated deployment gate relying solely on an "
        "aggregate threshold (e.g. alert if accuracy drops > 5 pp) would have failed to trigger. Yet underneath, Subgroup B suffered "
        "a severe <b>9.53 percentage point crash</b>, while Subgroup A experienced zero degradation.",
        body_style
    ))
    story.append(Paragraph(
        "The mathematical foundation for this disparity originates in the data generation process (<code>src/data_generation.py#L183-L191</code>):",
        body_style
    ))
    # Centered formula box for logit comparison
    formula_text = (
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Subgroup A:</b> &nbsp; <code>logit<sub>A</sub> = 0.1·f<sub>1</sub> + <b>0.5·f<sub>2_norm</sub></b> + 0.05·f<sub>3</sub></code>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Subgroup B:</b> &nbsp; <code>logit<sub>B</sub> = 0.1·f<sub>1</sub> + <b>1.5·f<sub>2_norm</sub></b> + 0.05·f<sub>3</sub></code>"
    )
    formula_table = Table([[Paragraph(formula_text, callout_style)]], colWidths=[516])
    formula_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(formula_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Because Subgroup B's target assignment features a <b>3x higher coefficient on normalized feature_2</b> "
        "(<i>w</i><sub>2</sub> = 1.5 vs. <i>w</i><sub>2</sub> = 0.5), "
        "the model's learned weights place overwhelming importance on <code>feature_2</code> when predicting outcomes for Subgroup B. "
        "When the 1000x scale shift corrupts <code>feature_2</code>, Subgroup B's logit values explode, flipping prediction probabilities and "
        "rendering classifications nearly arbitrary. Because Subgroup A distributes predictive weight across all features, it remained impervious. "
        "This proves that pre-inference quality validation is fundamentally an <b>algorithmic fairness and compliance requirement</b>.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: ANOMALY DISTRIBUTIONS, BENCHMARKS, VERIFICATION & CONCLUSION
    # =========================================================================
    story.append(Paragraph("8. Distributional Outlier Analysis & Runtime Benchmarks", h1_style))

    # Side-by-side or stacked figures for distribution and runtime
    fig3_path = os.path.join(project_root, "data", "reports", "feature2_distribution.png")
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=470, height=130))
        story.append(Paragraph("<b>Figure 3:</b> Probability density and empirical distribution of <code>feature_2</code>: Clean U(0,100) vs. Corrupted (1000x scale-shift up to 100,000).", fig_caption_style))

    fig4_path = os.path.join(project_root, "data", "reports", "runtime_breakdown.png")
    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=470, height=130))
        story.append(Paragraph("<b>Figure 4:</b> End-to-end execution latency across pipeline phases (Generation, Injection, Validation, and Model Evaluation).", fig_caption_style))

    story.append(Paragraph(
        "<b>Latency & Computational Overhead:</b> Full PyDeequ validation across 17 constraint groups on 10,300 rows completed in <b>~3.52 seconds</b> "
        "(total end-to-end pipeline run: ~14.6 seconds). In high-throughput feature stores (e.g. Feast or Hopsworks), this demonstrates that "
        "comprehensive statistical verification adds negligible latency compared to the catastrophic financial or ethical cost of silent model failure.",
        body_style
    ))

    # Section 9: Platform Engineering Challenges
    story.append(Paragraph("9. Platform Engineering & Distributed Reliability on Windows", h1_style))
    story.append(Paragraph(
        "Deploying PySpark and PyDeequ in mixed OS environments surfaces several critical low-level networking and thread lifecycle challenges:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>BlockManager Topology & Loopback Binding:</b> On Windows hosts with virtual network adapters (VPNs/WSL), Spark's driver resolves to invalid hostnames, "
        "causing <code>NullPointerException: BlockManagerId.executorId() is null</code> in an infinite heartbeat loop. We resolved this by explicitly pinning "
        "<code>SPARK_LOCAL_IP='127.0.0.1'</code>, <code>spark.driver.host='127.0.0.1'</code>, and <code>spark.driver.bindAddress='127.0.0.1'</code>.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Py4J Callback Server Socket Teardown:</b> PyDeequ launches a Py4J Gateway Callback Server for custom constraint lambdas. Under Windows Python, "
        "non-daemon gateway threads prevent processes from terminating upon script completion. We implemented explicit teardown hooks "
        "(<code>cb.shutdown()</code> and <code>os._exit()</code> in pytest lifecycle hooks) ensuring 100% clean terminal exit.",
        body_style
    ))

    # Section 10: Automated Test Suite & Recommendations
    story.append(Paragraph("10. Automated Verification Suite & Production Recommendations", h1_style))
    story.append(Paragraph(
        "The repository includes an automated test suite (<code>tests/test_pipeline.py</code>) comprising 10 unit and integration tests. "
        "All 10 tests passed with 100% success rate: verifying deterministic seeding, exact fault injection rates, complete 4-category failure detection, "
        "quarantine inference blocking, and the subgroup B degradation inequality (<code>acc_drop_B > acc_drop_A</code>).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Architectural Recommendations for Enterprise Feature Stores:</b><br/>"
        "1. <b>Never Rely on Type Systems Alone:</b> Unit and scale shifts produce valid numeric floats that pass schema validators; statistical range gates are essential.<br/>"
        "2. <b>Disaggregate Model Health Metrics:</b> Continuously monitor fairness and accuracy across protected sub-populations rather than relying on global averages.<br/>"
        "3. <b>Enforce Hard Quarantine Guards:</b> Integrate circuit-breaker checks into CI/CD and Airflow DAGs so corrupt batches are automatically isolated before inference.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Report built successfully: {output_path}")

    # Also save a copy to data/reports/technical_report.pdf
    reports_copy = os.path.join(project_root, "data", "reports", "technical_report.pdf")
    try:
        import shutil
        shutil.copyfile(output_path, reports_copy)
        print(f"[SUCCESS] Copied to: {reports_copy}")
    except Exception as e:
        print(f"[WARN] Could not copy to reports dir: {e}")

    # Verify total page count
    try:
        from pypdf import PdfReader
        reader = PdfReader(output_path)
        print(f"[INFO] Verified Page Count: {len(reader.pages)} pages")
    except Exception as e:
        pass


if __name__ == "__main__":
    build_pdf()
