from pathlib import Path
import json
import csv

ROOT = Path("/home/abdullah2025/fyp_project")
OUT = ROOT / "outputs" / "xai_z2_complete"

gaps = [
    {
        "gap_id": "G01",
        "category": "Scope",
        "title": "Direct ensemble attribution",
        "status": "Open",
        "severity": "Medium",
        "description": "Deep XAI explains the provenance-selected component models of Z2_tuned rather than applying SHAP or gradients directly to the full Z2 ensemble.",
    },
    {
        "gap_id": "G02",
        "category": "Scope",
        "title": "Concept-level counterfactuals",
        "status": "Open",
        "severity": "Medium",
        "description": "Concept-level counterfactual explanations were not implemented because the A1 feature mapping is a clinical feature/domain mapping rather than a formal concept representation.",
    },
    {
        "gap_id": "G03",
        "category": "Scope",
        "title": "Clinical-note explanation",
        "status": "Open",
        "severity": "Medium",
        "description": "Text explanations are provided at modality/embedding level; token-level or sentence-level attribution for clinical notes is not included.",
    },
    {
        "gap_id": "G04",
        "category": "Coverage",
        "title": "XAI sample coverage",
        "status": "Open",
        "severity": "Low",
        "description": "The explanation cohort is deterministic but is not an exhaustive explanation of every test-set prediction.",
    },
    {
        "gap_id": "G05",
        "category": "Methodology",
        "title": "Tree SHAP methodological interpretation",
        "status": "Open",
        "severity": "Medium",
        "description": "Tree SHAP provides model attribution and should not be interpreted as a causal treatment effect.",
    },
    {
        "gap_id": "G06",
        "category": "Methodology",
        "title": "Integrated gradients methodological interpretation",
        "status": "Open",
        "severity": "Medium",
        "description": "Integrated gradients are baseline-sensitive model explanations and are not causal effects.",
    },
    {
        "gap_id": "G07",
        "category": "Methodology",
        "title": "Temporal occlusion interpretation",
        "status": "Open",
        "severity": "Medium",
        "description": "Temporal occlusion depends on the replacement strategy and one-hour window and should not be interpreted as causal evidence.",
    },
    {
        "gap_id": "G08",
        "category": "Methodology",
        "title": "MoE routing interpretation",
        "status": "Open",
        "severity": "Medium",
        "description": "MoE router weights describe architectural routing behavior and do not represent clinical responsibility or causal importance.",
    },
    {
        "gap_id": "G09",
        "category": "Interpretation",
        "title": "Causal interpretation",
        "status": "Open",
        "severity": "High",
        "description": "Predictive XAI is not causal inference and should not be interpreted as a causal treatment effect.",
    },
    {
        "gap_id": "G10",
        "category": "Leakage",
        "title": "Test-set methodological leakage",
        "status": "Controlled",
        "severity": "Low",
        "description": "XAI settings were frozen using validation data before application to the test cohort.",
    },
    {
        "gap_id": "G11",
        "category": "Provenance",
        "title": "Checkpoint provenance",
        "status": "Closed",
        "severity": "Low",
        "description": "The exact locked provenance models were used and passed the prediction-parity audit.",
    },
]

# CSV gap register
csv_path = OUT / "37_XAI_GAP_REGISTER.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["gap_id", "category", "title", "status", "severity", "description"]
    )
    writer.writeheader()
    writer.writerows(gaps)

# Markdown gap register
md_path = OUT / "38_XAI_GAP_REGISTER.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# XAI Gap Register\n\n")
    f.write("| ID | Category | Title | Status | Severity | Description |\n")
    f.write("|---|---|---|---|---|---|\n")
    for g in gaps:
        desc = g["description"].replace("|", "\\|")
        f.write(
            f'| {g["gap_id"]} | {g["category"]} | {g["title"]} | '
            f'{g["status"]} | {g["severity"]} | {desc} |\n'
        )

# Update evidence-gaps CSV while preserving the same gap content
evidence_path = OUT / "34_deep_xai_evidence_gaps.csv"
with evidence_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["gap_id", "category", "title", "status", "severity", "description"]
    )
    writer.writeheader()
    writer.writerows(gaps)

# Update package manifest
manifest_path = OUT / "36_deep_xai_package_manifest.json"
manifest = {}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}

manifest["xai_gap_register"] = {
    "csv": "37_XAI_GAP_REGISTER.csv",
    "markdown": "38_XAI_GAP_REGISTER.md",
    "entries": len(gaps),
    "high_severity_open_gaps": sum(
        1 for g in gaps if g["status"] == "Open" and g["severity"] == "High"
    ),
}

manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("XAI GAP REGISTER FINALIZED")
print(csv_path)
print(md_path)
print(evidence_path)
print(manifest_path)
print("entries:", len(gaps))
