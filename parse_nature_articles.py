import argparse, os, glob, sys, re, json, csv
from bs4 import BeautifulSoup

try:
    from chempp.crawler import parse_html  # noqa
    parser_available = True
    print("[INFO] Using ChemDataParser for HTML parsing.")
except Exception as e:
    parser_available = False
    print(f"[INFO] ChemDataParser unavailable ({e!r}); falling back to BeautifulSoup.")

# -------------------- KEYWORDS --------------------
PRIMARY_RE = re.compile(r"\b2\s*[-]?\s*D\s*[-]?\s*Materials\b", re.IGNORECASE)

SECONDARY_RES = [
    re.compile(r"\bMemristor\b", re.IGNORECASE),
    re.compile(r"\bResistive\s+switching\s+device\b", re.IGNORECASE),
    re.compile(r"\bMemristor\s+device\b", re.IGNORECASE),
]

def has_primary(text: str) -> bool:
    return bool(PRIMARY_RE.search(text or ""))

def has_secondary(text: str) -> bool:
    t = text or ""
    return any(rx.search(t) for rx in SECONDARY_RES)

def load_html(path: str):
    # safer read: ignore errors so odd encodings don’t crash
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_title_abstract_body_bs4(path: str):
    html = load_html(path)
    soup = BeautifulSoup(html, "html.parser")

    # ---- Title ----
    title = ""
    og = soup.find("meta", {"property": "og:title"})
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    abstract = ""
    for sel in [
        ("meta", {"name": "dc.Description"}),
        ("meta", {"name": "dc.description"}),
        ("meta", {"name": "description"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "citation_abstract"}),
    ]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            abstract = tag["content"].strip()
            if abstract:
                break

    if not abstract:
        for s in soup.find_all("script", {"type": "application/ld+json"}):
            try:
                data = json.loads(s.string or "")
            except Exception:
                continue
            # Normalize to list
            candidates = data if isinstance(data, list) else [data]
            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                typ = obj.get("@type") or obj.get("@type".lower())
                if isinstance(typ, list):
                    typ = next((t for t in typ if isinstance(t, str)), None)
                if isinstance(typ, str) and "ScholarlyArticle" in typ:
                    head = obj.get("headline") or obj.get("name") or ""
                    desc = obj.get("description") or ""
                    if not title and isinstance(head, str):
                        title = head.strip()
                    if isinstance(desc, str):
                        abstract = desc.strip()
                    break
            if abstract:
                break

    if not abstract:
        sec = soup.select_one(
            'section.abstract, div.abstract, div[class*="Abstract"], '
            'section[id*="bstract"], section[role="doc-abstract"]'
        )
        if sec:
            abstract = sec.get_text(" ", strip=True)

    body_nodes = soup.select("article, main")
    if body_nodes:
        body = " ".join(node.get_text(" ", strip=True) for node in body_nodes)
    else:
        body = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))

    def norm(s): return re.sub(r"\s+", " ", (s or "").strip())
    return norm(title), norm(abstract), norm(body)

def collect_html_files(in_dir: str):
    files = []
    for pat in ("*.html", "*.htm"):
        files += glob.glob(os.path.join(in_dir, pat))
    if not files:
        for pat in ("**/*.html", "**/*.htm"):
            files += glob.glob(os.path.join(in_dir, pat), recursive=True)
    # filter out saved resources
    files = [f for f in sorted(set(files)) if not os.path.basename(f).lower().startswith("saved_resource")]
    return files

def main():
    ap = argparse.ArgumentParser(description="Parse and filter HTML articles for 2D Materials + memristor keywords")
    ap.add_argument("-i", "--in_dir", default="HTMLfiles", help="Directory with .html/.htm files (default: HTMLfiles)")
    ap.add_argument("-o", "--out_dir", default="OUTPUT", help="Output directory (default: OUTPUT)")
    ap.add_argument("--fmt", choices=("csv","jsonl","both"), default="both", help="Output format (default: both)")
    ap.add_argument("--debug", action="store_true", help="Show per-file match diagnostics")
    args = ap.parse_args()

    if not os.path.isdir(args.in_dir):
        print(f"[ERROR] in_dir not found: {args.in_dir!r}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_html_files(args.in_dir)
    if not files:
        print(f"[ERROR] No .html/.htm files found under '{args.in_dir}' (including subfolders).")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} HTML files; parsing…")

    matched = []
    unmatched_rows = []  # for a CSV explaining why each file failed
    for fn in files:
        try:
            if parser_available:
                art, _ = parse_html(fn)
                title    = (getattr(art.title,    "text", "") or "").strip()
                abstract = (getattr(art.abstract, "text", "") or "").strip()
                body     = ""  # chem parser: we’ll just rely on title+abstract here
            else:
                title, abstract, body = extract_title_abstract_body_bs4(fn)

            combined = f"{title}\n\n{abstract}\n\n{body}"
            p = has_primary(combined)
            s = has_secondary(combined)
            if args.debug:
                print(f"[DEBUG] {os.path.basename(fn)} -> primary={p} secondary={s}")

            if p and s:
                matched.append({"file": os.path.basename(fn), "title": title, "abstract": abstract})
            else:
                unmatched_rows.append({
                    "file": os.path.basename(fn),
                    "primary_found": p,
                    "secondary_found": s,
                    "title_sample": title[:120],
                    "abstract_sample": abstract[:160],
                })

        except Exception as ex:
            print(f"[WARN] Skipping {fn!r} due to parse error: {ex}")

    if not matched:
        print("[INFO] No articles matched your keywords.")
        if unmatched_rows:
            diag = os.path.join(args.out_dir, "nature_unmatched.csv")
            with open(diag, "w", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=["file","primary_found","secondary_found","title_sample","abstract_sample"])
                w.writeheader()
                w.writerows(unmatched_rows)
            print(f"[INFO] Wrote unmatched diagnostics → {diag}")
        sys.exit(0)

    if args.fmt in ("jsonl","both"):
        out = os.path.join(args.out_dir, "nature_articles.jsonl")
        with open(out, "w", encoding="utf-8") as fp:
            for rec in matched:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote JSONL → {out}")

    if args.fmt in ("csv","both"):
        out = os.path.join(args.out_dir, "nature_articles.csv")
        with open(out, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["file","title","abstract"])
            w.writeheader()
            w.writerows(matched)
        print(f"[OK] Wrote CSV   → {out}")

    if unmatched_rows:
        diag = os.path.join(args.out_dir, "nature_unmatched.csv")
        with open(diag, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["file","primary_found","secondary_found","title_sample","abstract_sample"])
            w.writeheader()
            w.writerows(unmatched_rows)
        print(f"[INFO] Wrote unmatched diagnostics → {diag}")

    print(f"[DONE] {len(matched)} article(s) parsed and filtered (out of {len(files)}).")


if __name__ == "__main__":
    main()
