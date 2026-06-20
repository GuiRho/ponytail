import argparse, json, shutil, subprocess, sys
from pathlib import Path

def clone_repo(url, target):
    if target.exists(): return print(f"Skipping {target}")
    subprocess.run(["git", "clone", url, str(target)], check=True)

def make_archive(project_dir):
    history = project_dir / "v1" / "history"
    history.mkdir(parents=True, exist_ok=True)
    for item in list(project_dir.iterdir()):
        name = item.name
        if name.startswith(".") or name in ("v1", "v2"): continue
        dest = history / name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True); shutil.rmtree(item)
        else:
            shutil.copy2(item, dest); item.unlink()
    return history

def create_structure(project_dir):
    for d in ("src", "tests", "config", "data"):
        (project_dir / d).mkdir(exist_ok=True)

def clean_file(path, root):
    try: text = path.read_text("utf-8")
    except UnicodeDecodeError: return []
    orig = text
    text = text.replace("\r\n", "\n")
    text = "\n".join(l.rstrip() for l in text.splitlines())
    text = text.strip() + "\n"
    if text != orig:
        path.write_text(text, "utf-8")
        return [str(path.relative_to(root))]
    return []

def clean_project(project_dir):
    changes = []
    for path in sorted(project_dir.rglob("*")):
        if path.is_dir() or any(p.startswith(".") for p in path.parts): continue
        if path.suffix in (".py", ".md", ".json", ".txt", ".yaml", ".yml", ".toml"):
            changes.extend(clean_file(path, project_dir))
    return changes

def inventory(project_dir):
    files = []
    for path in sorted(project_dir.rglob("*")):
        if path.is_dir() or any(p.startswith(".") for p in path.parts): continue
        files.append({"path": str(path.relative_to(project_dir)), "size": path.stat().st_size})
    return {"root": str(project_dir), "file_count": len(files), "files": files}

def count_lines(path):
    return sum(len(f.read_text("utf-8", errors="ignore").splitlines()) for f in path.rglob("*.py") if f.is_file())

def count_files(path, pattern="*.py"):
    return len(list(path.rglob(pattern)))

def count_hardcoded_paths(path):
    return sum(f.read_text("utf-8", errors="ignore").count(p) for f in path.rglob("*.py") for p in ("C:/Users/", "/home/", "/Users/"))

def count_tests(path):
    return len(list(path.rglob("test_*.py")))

def count_duplicates(files):
    prefixes = {}
    for f in files:
        stem = Path(f).stem
        base = stem.replace("_0", "").replace("_v1", "").replace("_v2", "").replace("new_", "").replace("docker_", "").replace("main_in_dev_", "")
        if base != stem:
            prefixes.setdefault(base, []).append(f)
    return sum(len(v) - 1 for v in prefixes.values() if len(v) > 1)

def scan(repo_root):
    py_files = [str(p.relative_to(repo_root)) for p in repo_root.rglob("*.py")]
    return {
        "total_lines": count_lines(repo_root),
        "py_files": count_files(repo_root),
        "test_files": count_tests(repo_root),
        "hardcoded_paths": count_hardcoded_paths(repo_root),
        "duplicate_variants": count_duplicates(py_files),
        "data_size_mb": round(sum(f.stat().st_size for f in repo_root.rglob("*") if f.is_file() and f.suffix in (".pkl", ".parquet", ".csv", ".html")) / (1024 * 1024), 1),
    }

def generate_report(before, after):
    def d(k):
        b, a = before.get(k, 0), after.get(k, 0)
        return f"{b} -> {a} ({'+' if a > b else ''}{a - b})"
    return f"""# Cleaning Report
Metrics: py_files {d('py_files')} | lines {d('total_lines')} | tests {d('test_files')} | hardcoded {d('hardcoded_paths')} | duplicates {d('duplicate_variants')} | data_mb {d('data_size_mb')}"""

def cmd_clean(args):
    target = Path(args.target or args.url.rstrip("/").split("/")[-1].replace(".git", ""))
    clone_repo(args.url, target)
    before = inventory(target)
    if args.archive: make_archive(target)
    create_structure(target)
    changes = clean_project(target)
    after = inventory(target)
    report = {"repo": args.url, "target": str(target), "files_before": before["file_count"], "files_after": after["file_count"], "changes": changes}
    (target / "superclean-report.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(f"Cleaned {len(changes)} files. Before: {before['file_count']}, After: {after['file_count']}")

def cmd_report(args):
    repo = Path(args.repo_root).resolve()
    history = repo / "v1" / "history"
    before = scan(history) if history.exists() else {}
    after = scan(repo)
    report = generate_report(before, after)
    if args.output:
        Path(args.output).write_text(report, "utf-8")
        print(f"Written to {args.output}")
    else:
        print(report)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cp = sub.add_parser("clean", help="Clone and clean a repo")
    cp.add_argument("url", nargs="?", default="", help="GitHub URL")
    cp.add_argument("--target", "-t")
    cp.add_argument("--no-archive", "-n", action="store_false", dest="archive")
    rp = sub.add_parser("report", help="Generate cleaning report")
    rp.add_argument("repo_root", help="Path to cleaned repo")
    rp.add_argument("--output", "-o")
    args = parser.parse_args()
    if args.command == "clean" or (args.command is None and getattr(args, "url", "")):
        if args.command is None:
            args.target = getattr(args, "target", None)
            args.archive = getattr(args, "archive", True)
        cmd_clean(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "test.py").write_bytes(b"  hello\r\n  world  \n")
    assert clean_file(d / "test.py", d) == ["test.py"]
    result = (d / "test.py").read_text("utf-8")
    assert result.rstrip() == "hello\n  world"
    assert not result.startswith(" ")
    assert inventory(d)["file_count"] >= 1
    (d / "v1" / "history").mkdir(parents=True, exist_ok=True)
    assert count_lines(d) == 2
    assert count_tests(d) == 0
    assert count_files(d, "*.py") >= 1
    assert isinstance(count_duplicates(["foo.py", "foo_v1.py"]), int)
    assert isinstance(scan(d), dict)
    shutil.rmtree(d)
    print("All pull-and-clean asserts passed")
