# Semi-Automated Qualitative Evidence Extraction from Research Papers in PDFs
## Offline, Rule-Based Pipeline

### Purpose of the Pipeline

This pipeline supports qualitative and mixed-methods evidence synthesis from research papers in PDF format, particularly in public health, transport, and intervention research. Manual extraction and coding of narrative evidence is time-consuming, difficult to reproduce, and often inconsistent across reviewers.

The pipeline provides a semi-automated, fully offline method for extracting text from PDFs, classifying relevant segments into review-relevant categories, and documenting all inclusion and exclusion decisions in a transparent and auditable manner. It does not replace expert judgement; it standardises mechanical extraction while leaving interpretation to the researcher.

The system was validated using peer-reviewed public health intervention trials and is designed to be conservative, prioritising methodological defensibility over maximal recall.


### General Workflow

The pipeline operates in three deterministic stages:

1. Text is extracted from a PDF and divided into paragraph-level segments.
2. Each segment is classified using explicit rule-based criteria reflecting common structures in intervention trials and public health research.
3. Results are written to structured outputs that support synthesis and full audit of decisions.

Running the pipeline multiple times on the same input produces identical results.


### Text Extraction and Segmentation

The full text of a PDF is converted into plain text and split into paragraph-like segments using spacing patterns and length thresholds. Segments preserve sufficient context while representing a single unit of evidence.

Each segment receives a unique identifier and retains a reference to the source file. No content is discarded at this stage; filtering occurs only during classification.

---

### Classification into Evidence Categories

Each segment is evaluated using predefined linguistic rules and assigned to one primary category or marked as non-extractable. Categories reflect common elements of qualitative and intervention-focused reviews, such as interventions, barriers, impacts, or frequencies.

Classification is rule-based rather than model-based. For example, intervention segments describe program components or delivery mechanisms, while impact segments contain outcome language, group comparisons, or time-based changes.

Segments that do not meet extractable criteria are assigned to a neutral category and excluded from synthesis.



### Strict and Lenient Review Modes

The pipeline supports two review modes:

**Strict mode (default, recommended):**  
Only segments reporting observed and extractable evidence are retained. Background discussion, theoretical framing, study design descriptions, eligibility criteria, outcome measure descriptions without results, baseline-only statistics, tables, and references are excluded. Outcome-related categories require trial-style result language.

**Lenient mode:**  
Less restrictive rules apply. Descriptive, contextual, and some methodological content may be retained. This mode is intended for early scoping rather than final synthesis.

All exclusion logic is recorded in the output audit trail.



### Explicit Exclusion Logic and Audit Trail

Every segment is either included or excluded with an explicit reason. Exclusion reasons include structural noise, background framing, eligibility criteria, methodological content, theoretical discussion, or tabular material.

All decisions are written to the output CSV alongside the original text, enabling inspection, verification, and reproducibility. No content is silently discarded.



### Output Files

The primary output is a CSV file containing all extracted segments. Each row includes the source file name, segment identifier, assigned category, classification rationale, exclusion reason (if applicable), an exclusion flag, and the original text.

An optional Markdown file summarises included segments into a thematic evidence matrix. Themes are derived from recurring phrases and illustrated using verbatim evidence, supporting narrative synthesis and manuscript preparation.



### Intended Use and Scope

The pipeline is intended for researchers conducting scoping reviews, qualitative syntheses, or mixed-methods reviews in public health, transport studies, urban planning, social science, and sustainability research. It is particularly suited to intervention trials where evidence is embedded in dense narrative text.

The system does not assess study quality, risk of bias, or causal validity.


### Technical Requirements

The pipeline runs locally using Python and does not require internet access, APIs, or proprietary software.

Required packages:
```bash
pip install pypdf pandas tqdm
Running the Pipeline
bash
Copy code
python cycle_extraction_pipeline_offline_v1.py Article.pdf output_segments.csv --markdown output_matrix.md --mode strict
Example with a validated study:

bash
Copy code
python cycle_extraction_pipeline_offline_v1.py Bernshtein2017.pdf bernstein_segments.csv --markdown bernshtein_matrix.md --mode strict
Reproducibility
All classification decisions are deterministic and rule-based. There is no stochastic behaviour, model training, or dependence on external services. This ensures auditability and suitability for peer-reviewed research.



This repository provides a reproducible and auditable foundation for semi-automated qualitative evidence extraction from PDF-based research literature.


Adaptation to Other Research Areas
Although tuned for public health and cycling intervention studies, the pipeline can be adapted to other domains by modifying category definitions, keyword patterns, and relevance checks without redesigning the overall structure
