#!/usr/bin/env python3
"""Minimal Markdown -> LaTeX (XeLaTeX, Russian) converter for the paper drafts: headings, paragraphs, **bold**, *italic*,
`code`, pipe tables, bullet lists, horizontal rules, images ![cap](path)."""
import re, sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text(); out = pathlib.Path(sys.argv[2])
def esc(t):
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("^", "\\^{}"), ("~", "\\textasciitilde{}"), ("$", "\\$"), ("<", "\\textless{}"), (">", "\\textgreater{}"), ("{", "\\{"), ("}", "\\}")):
        t = t.replace(a, b)
    return t
def inline(t):
    parts = re.split(r"(`[^`]*`)", t); res = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            res.append("\\texttt{" + esc(p[1:-1]) + "}")
        else:
            p = esc(p)
            p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
            p = re.sub(r"(?<![\w\\])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\\emph{\1}", p)
            res.append(p)
    return "".join(res)
lines = src.split("\n"); body = []; i = 0; title = None
while i < len(lines):
    ln = lines[i]
    if ln.startswith("# ") and title is None:
        title = inline(ln[2:]); i += 1; continue
    if ln.startswith("## "):
        body.append("\\section*{" + inline(ln[3:]) + "}"); i += 1; continue
    if ln.startswith("### "):
        body.append("\\subsection*{" + inline(ln[4:]) + "}"); i += 1; continue
    if ln.strip() == "---":
        body.append("\\bigskip\\hrule\\bigskip"); i += 1; continue
    m = re.match(r"!\[(.*?)\]\((.*?)\)", ln.strip())
    if m:
        body.append("\\begin{figure}[H]\\centering\\includegraphics[width=\\textwidth]{" + m.group(2) + "}\\caption*{" + inline(m.group(1)) + "}\\end{figure}"); i += 1; continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append(lines[i]); i += 1
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[\s\-:|]+\|$", r.strip())]
        ncol = max(len(c) for c in cells)
        body.append("\\begin{center}\\footnotesize\\begin{tabularx}{\\textwidth}{" + "l" + "X" * (ncol - 1) + "}\\toprule")
        for k, c in enumerate(cells):
            c = c + [""] * (ncol - len(c))
            body.append(" & ".join(inline(x) for x in c) + " \\\\" + (" \\midrule" if k == 0 else ""))
        body.append("\\bottomrule\\end{tabularx}\\end{center}"); continue
    if ln.startswith("- ") or re.match(r"^\d+\. ", ln):
        items = []
        while i < len(lines) and (lines[i].startswith("- ") or re.match(r"^\d+\. ", lines[i]) or (lines[i].startswith("   ") and items)):
            if lines[i].startswith("   "):
                items[-1] += " " + lines[i].strip()
            else:
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
            i += 1
        body.append("\\begin{itemize}" + "".join("\\item " + inline(x) for x in items) + "\\end{itemize}"); continue
    if ln.strip() == "":
        body.append(""); i += 1; continue
    para = [ln]; i += 1
    while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith(("#", "|", "- ", "![", "---")) and not re.match(r"^\d+\. ", lines[i]):
        para.append(lines[i]); i += 1
    body.append(inline(" ".join(para)))
tex = r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{fontspec}\setmainfont{PT Serif}\setmonofont{Menlo}[Scale=0.85]
\usepackage[russian,english]{babel}
\usepackage{graphicx,booktabs,tabularx,float,caption,amsmath,amssymb}
\captionsetup{font=small}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\begin{document}
\selectlanguage{russian}
{\LARGE\bfseries """ + (title or "") + r"""\par}\bigskip
""" + "\n".join(body) + "\n\\end{document}\n"
out.write_text(tex); print("written", out)
