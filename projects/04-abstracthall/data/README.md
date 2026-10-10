# Eight real yeast abstracts — reconstructed study corpus

This dataset follows Table 2 of Jaaouine & King, *Mitigating Hallucinations in Zero-Shot Scientific Summarisation: A Pilot Study*, https://arxiv.org/abs/2512.00931v1 . It is a reconstruction from public arXiv records, not an official author dataset release. All eight records are real abstracts; none are synthetic.

## Files
- `abstracts.jsonl`: one record per line; recommended input for the experiment. `text` holds the abstract, `source` the version-pinned source URL, and `id` the stable record ID.
- `abstracts.json`: the same records as a readable JSON array.
- `abstracts.csv`: the same records for tabular import (authors joined with semicolons).
- `manifest.json`: collection and extraction metadata.
- `validation.txt`: integrity and word-count checks.

## Provenance and version limits
Collected 2026-10-10 from arXiv abstract-page citation metadata. HTML entities were decoded and whitespace collapsed. Wording, numbers, punctuation, and source spellings were not corrected or paraphrased. Each abstract was checked against its page's visible abstract block, allowing the page's rendering of a LaTeX umlaut in the membrane-trafficking record. The dataset preserves that record's original LaTeX accent notation from metadata. Metadata author lists are used rather than arXiv's abbreviated download-button labels.

The pinned versions are the latest versions on the retrieved pages; their submission histories all predate the study's 2025-11-30 arXiv submission. The study gives no exact version numbers or machine-readable original input files, so exact reproduction of its inputs is unverified. Seven of eight whitespace word counts match Table 2. The gene-expression abstract counts 211 versus the reported 207; it has not been edited to force a match. The GAL paper's arXiv record, 2107.06823v1, was independently located by matching title and authors because reference 14 omits an arXiv link.

## Corpus
| ID | Title | Version | Words / study |
|---|---|---|---|
| yeast_01 | [Engineering Yeast Cells to Facilitate Information Exchange](https://arxiv.org/abs/2401.13712v2) | v2 | 198 / 198 |
| yeast_02 | [On the modeling of endocytosis in yeast](https://arxiv.org/abs/1310.8652v3) | v3 | 200 / 200 |
| yeast_03 | [Membrane Trafficking in the Yeast Saccharomyces cerevisiae Model](https://arxiv.org/abs/1804.07523v1) | v1 | 214 / 214 |
| yeast_04 | [The Biosensor based on electrochemical dynamics of fermentation in yeast Saccharomyces Cerevisiae](https://arxiv.org/abs/2202.07795v1) | v1 | 131 / 131 |
| yeast_05 | [Gateway vectors for efficient artificial gene assembly in vitro and expression in yeast Saccharomyces cerevisiae](https://arxiv.org/abs/1212.5109v3) | v3 | 199 / 199 |
| yeast_06 | [Quantitative Analysis of the Effective Functional Structure in Yeast Glycolysis](https://arxiv.org/abs/1009.3627v2) | v2 | 228 / 228 |
| yeast_07 | [The evolution of the GALactose utilization pathway in budding yeasts](https://arxiv.org/abs/2107.06823v1) | v1 | 119 / 119 |
| yeast_08 | [Evolution at two levels of gene expression in yeast](https://arxiv.org/abs/1311.7140v1) | v1 | 211 / 207 |

## Load in Python
```python
import json
from pathlib import Path
items = [json.loads(line) for line in Path("abstracts.jsonl").read_text(encoding="utf-8").splitlines()]
for item in items:
    prompt = "Summarise: " + item["text"]
```

## Evaluation boundaries
This package contains source inputs, not the study's generated summaries, key-sentence selections, reference summaries, or human factuality labels. It does not certify that the abstracts' scientific claims are independently true. Faithfulness evaluation should compare summaries to the supplied source. Do not treat word overlap as proof of factual support. Eight abstracts remain eight source documents regardless of generation repeat count.

## Attribution and rights
Authors and arXiv license links are recorded per item. The records have mixed licenses; public arXiv access does not imply a uniform unrestricted reuse license. No new license is assigned to the source abstracts. Preserve source attribution and check the recorded terms before redistribution.
