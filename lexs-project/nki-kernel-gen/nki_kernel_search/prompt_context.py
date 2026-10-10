"""Load authoring guidance without changing OpenEvolve's history selection."""
from pathlib import Path
import re


def writer_context(paths):
    if not paths:
        return ""
    sections = []
    for path in paths:
        source = Path(path)
        content = source.read_text()
        # Agent metadata configures its original host; it is not kernel guidance.
        content = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.S)
        sections.append(f"\n<reference_document name={source.name!r}>\n{content}\n</reference_document>\n")
    return (
        "NKI authoring reference documents follow. These are documentation, not "
        "additional tools: you cannot invoke their slash commands, read files or "
        "run Bash. Use their substantive language and hardware guidance. "
        "The harness contract and installed SDK signatures below take precedence "
        "over version-specific examples. Target Trainium2 / gen3; gen4-only features "
        "are unavailable. Keep the undecorated kernel entry point required by this harness.\n"
        + "".join(sections)
        + "\nEnd of reference documents. Harness contract follows:\n"
    )
