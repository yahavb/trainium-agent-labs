"""Small, curated official-document corpus with deterministic keyword retrieval."""

import re

ISA = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html"
PERF = "https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/nki_perf_guide.html"
CARDS = [
    dict(id="copy-removal", keywords="copy copies intermediate product gradient sbuf psum throughput",
         source=ISA, guidance="tensor_tensor permits PSUM/SBUF operands, but not two PSUM operands. "
         "Consider adding the SBUF bias directly to the PSUM matmul output rather than copying it first."),
    dict(id="scalar-fusion", keywords="fusion fuse multiply add subtract scalar instructions throughput",
         source=ISA, guidance="tensor_scalar can combine two supported operations with scalar/per-partition "
         "operands. Preserve alpha: subtracting an unscaled gradient changes the solver. "
         "New reviewed scale-fused form precomputes negative_rate=-rate, then forms "
         "gradient*negative_rate+impulses in one instruction. This removes scaling/update separation, "
         "unlike the slower earlier -scaled+impulses variant. Verify installed API and rounding."),
    dict(id="matmul-layout", keywords="matmul transpose matrix layout correctness stationary moving",
         source=ISA, guidance="nc_matmul computes stationary.T @ moving with SBUF inputs and PSUM output. "
         "The supplied packed matrix already stores A.T; do not transpose it again."),
    dict(id="profile-first", keywords="timing throughput latency slower utilization profile bottleneck",
         source=PERF, guidance="Profile data movement and engine utilization before choosing larger changes. "
         "On-chip reuse and fusion are hypotheses to benchmark, not guaranteed gains. This historical "
         "guide supplies principles; installed SDK signatures govern actual APIs."),
]


def retrieve(query, count=3):
    terms = set(re.findall(r"[a-z]+", query.lower()))
    ranked = sorted(enumerate(CARDS), key=lambda item: (
        -len(terms & set(item[1]["keywords"].split())), item[0]))
    selected = [card for _, card in ranked[:count]]
    text = "Retrieved documentation guidance (curated paraphrases, not measured benefits):\n" + "\n".join(
        f"[{card['id']}] {card['guidance']} Source: {card['source']}" for card in selected)
    return text, dict(method="keyword retrieval over curated official-document cards",
                      query=query, cards=selected, live_search=False, embeddings=False)
