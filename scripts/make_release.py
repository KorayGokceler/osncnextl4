#!/usr/bin/env python
'''
Assemble the self-contained, runnable release handed to a collaborator.

    python scripts/make_release.py [--out dist/oscnext_l4_release] \\
        [--models L4_output/models_pass2] [--partial-models]

WHAT GOES IN IS COMPUTED, NOT LISTED.  The release serves three things: L3 .i3
-> L4 .i3 (process_L4.py), the training (make_dataset.py,
train_L4_classifier.py) and the models; check_application.py ships with them
as the first check to run.  From those entry scripts:

  * modules: the import closure inside `oscnext_l4`, read with ast -- every
    `import oscnext_l4...`, `from oscnext_l4... import`, `from . import` and
    `from .x import`, at any depth, function-level imports included (the tray
    only imports classifier.py inside compute_L4_cut).  Nothing is imported,
    so this runs without icetray.
  * scripts: any script in scripts/ that a shipped file NAMES in a string it
    shows the user (an error message pointing at scripts/diagnose_env.py, a
    help text pointing at scan_files.py), and then ITS closure, to a fixpoint.
    A release that tells its reader to run a script it does not contain is not
    self-contained.
  * config: every file in config/ whose name a shipped file or a shipped
    config file mentions in a string (varmap opens variables.json;
    productions.json points at config/README.md).

IT FAILS LOUDLY, and leaves nothing behind, when:
  * an internal import names a module that does not exist, or a name the
    target module does not define;
  * a relative import leaves the package, or a script imports relatively;
  * after copying, a shipped file's import does not resolve INSIDE the release
    (checked on the copy, not the source), or a file does not compile;
  * --models is given and a model lacks its sidecar, its sidecar's feature
    order is not the booster's, a feature is not in the variable table, or one
    of the two models is absent (unless --partial-models);
  * a README field is left unfilled.

The release is built in a temporary directory beside --out and renamed into
place only when every check passed; an existing --out is replaced only if it
carries the marker file an earlier build wrote.

The README is release/README.md with release/physics_caveats.md in it; the
model-dependent parts (efficiency at the default cut, signal sets and
weighting, muon background) are written from the model sidecars, never typed.
A fact a sidecar does not carry is reported as "not recorded", with the
reason.

Models are data and never enter the git repository; they come from --models.
'''

import os
import re
import ast
import sys
import json
import shutil
import hashlib
import argparse
import datetime
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "oscnext_l4"
ENTRY_SCRIPTS = ("scripts/process_L4.py", "scripts/make_dataset.py",
                 "scripts/train_L4_classifier.py",
                 "scripts/check_application.py")
TEMPLATE_DIR = os.path.join(ROOT, "release")
MARKER = ".oscnext_l4_release"
TAGS = ("noise", "muon")


class ReleaseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Static import analysis
# ---------------------------------------------------------------------------

def _parse(path):
    with open(path) as fh:
        return ast.parse(fh.read(), filename=path)


def _module_path(root, modname):
    """oscnext_l4.x -> <root>/oscnext_l4/x.py, oscnext_l4 -> .../__init__.py"""
    parts = modname.split(".")
    if parts[0] != PKG or len(parts) > 2:
        return None
    rel = os.path.join(PKG, "__init__.py") if len(parts) == 1 \
        else os.path.join(PKG, parts[1] + ".py")
    return rel if os.path.exists(os.path.join(root, rel)) else None


def _defined_names(tree):
    """Top-level names a module defines (including in top-level if/try)."""
    names = set()

    def visit(stmts):
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    names.add(a.asname or a.name)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    for sub in getattr(node, field, []) or []:
                        visit(sub.body if isinstance(sub, ast.ExceptHandler)
                              else [sub])
    visit(tree.body)
    return names


def _imports(tree, rel):
    """
    (internal, external, errors) of one file.

    internal: [(module, [names], lineno)] inside the package
    external: set of top-level package names
    """
    in_pkg = rel.startswith(PKG + os.sep)
    internal, external, errors = [], set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PKG or a.name.startswith(PKG + "."):
                    internal.append((a.name, [], node.lineno))
                else:
                    external.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not in_pkg:
                    errors.append("%s:%d: relative import outside the package"
                                  % (rel, node.lineno))
                    continue
                if node.level > 1:
                    errors.append("%s:%d: relative import leaves the package"
                                  % (rel, node.lineno))
                    continue
                mod = PKG + ("." + node.module if node.module else "")
            elif node.module and (node.module == PKG
                                  or node.module.startswith(PKG + ".")):
                mod = node.module
            else:
                if node.module:
                    external.add(node.module.split(".")[0])
                continue
            internal.append((mod, [a.name for a in node.names], node.lineno))
        elif isinstance(node, ast.Call):
            # importlib.import_module("oscnext_l4...") would escape the ast
            # scan above; refuse it rather than miss it.
            f = node.func
            name = getattr(f, "attr", getattr(f, "id", ""))
            if name in ("import_module", "__import__") and node.args and \
                    isinstance(node.args[0], ast.Constant) and \
                    isinstance(node.args[0].value, str) and \
                    node.args[0].value.lstrip(".").startswith(PKG):
                errors.append("%s:%d: dynamic import of %r cannot be followed"
                              % (rel, node.lineno, node.args[0].value))
    return internal, external, errors


def _strings(tree, docstrings=False):
    """String constants of a module, docstrings excluded unless asked."""
    skip = set()
    if not docstrings:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)) and \
                    node.body and isinstance(node.body[0], ast.Expr) and \
                    isinstance(node.body[0].value, ast.Constant):
                skip.add(id(node.body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip]


def _json_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_strings(v)


def closure(root, entry_scripts):
    """
    -> (files, external, errors): every file the release needs, relative to
    `root`, starting from `entry_scripts`.  See the module docstring.
    """
    scripts_available = {f for f in os.listdir(os.path.join(root, "scripts"))
                         if f.endswith(".py")} \
        if os.path.isdir(os.path.join(root, "scripts")) else set()
    config_available = sorted(os.listdir(os.path.join(root, "config"))) \
        if os.path.isdir(os.path.join(root, "config")) else []

    files, external, errors = set(), set(), []
    todo = list(entry_scripts) + [os.path.join(PKG, "__init__.py")]
    strings = []
    while todo:
        rel = todo.pop()
        if rel in files:
            continue
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            errors.append("%s does not exist" % rel)
            continue
        files.add(rel)
        tree = _parse(path)
        internal, ext, errs = _imports(tree, rel)
        external |= ext
        errors += errs
        for mod, names, lineno in internal:
            target = _module_path(root, mod)
            if target is None:
                errors.append("%s:%d: imports %s, which is not a module of %s"
                              % (rel, lineno, mod, PKG))
                continue
            todo.append(target)
            if names and names != ["*"]:
                defined = _defined_names(_parse(os.path.join(root, target)))
                for n in names:
                    sub = _module_path(root, mod + "." + n) \
                        if mod == PKG else None
                    if sub:
                        todo.append(sub)
                    elif n not in defined:
                        errors.append("%s:%d: imports %r from %s, which does "
                                      "not define it" % (rel, lineno, n, mod))
        # Scripts a shipped file points its reader at.
        own = _strings(tree)
        strings += own
        for s in own:
            for m in re.finditer(r"(?<![\w/])(?:scripts/)?([A-Za-z_]\w*\.py)\b", s):
                if m.group(1) in scripts_available:
                    todo.append(os.path.join("scripts", m.group(1)))
    # Config files named by a shipped file or by a shipped config file.
    changed = True
    while changed:
        changed = False
        for name in config_available:
            rel = os.path.join("config", name)
            if rel in files:
                continue
            if any(s == name or ("config/" + name) in s for s in strings):
                files.add(rel)
                changed = True
                if name.endswith(".json"):
                    with open(os.path.join(root, rel)) as fh:
                        strings += list(_json_strings(json.load(fh)))
    return files, external, errors


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _booster_features(txt):
    """The `feature_names=` line of a LightGBM text model -- no lightgbm needed."""
    with open(txt) as fh:
        for line in fh:
            if line.startswith("feature_names="):
                return line.strip().split("=", 1)[1].split(" ")
            if line.startswith("Tree="):
                break
    return None


def check_models(model_dir, partial):
    """-> ({tag: {txt, json, meta, sha256, warnings}}, missing_tags)."""
    sys.path.insert(0, ROOT)
    from oscnext_l4.varmap import FEATURE_MAP, NOISE_FEATURES, MUON_FEATURES
    standard = {"noise": NOISE_FEATURES, "muon": MUON_FEATURES}
    if not os.path.isdir(model_dir):
        raise ReleaseError("--models %s is not a directory" % model_dir)
    found, missing, errors = {}, [], []
    for tag in TAGS:
        txt = os.path.join(model_dir, "L4_%s_model.txt" % tag)
        js = os.path.join(model_dir, "L4_%s_model.json" % tag)
        if not os.path.exists(txt) and not os.path.exists(js):
            missing.append(tag)
            continue
        if not (os.path.exists(txt) and os.path.exists(js)):
            errors.append("%s: %s without %s -- ship both or neither"
                          % (tag, *((txt, js) if os.path.exists(txt) else (js, txt))))
            continue
        with open(js) as fh:
            meta = json.load(fh)
        feats = list(meta.get("features") or [])
        booster = _booster_features(txt)
        if booster != feats:
            errors.append("%s: the sidecar's feature order %s is not the "
                          "booster's %s" % (tag, feats, booster))
        unknown = [f for f in feats if f not in FEATURE_MAP]
        if unknown:
            errors.append("%s: %s not in the variable table's frame side; the "
                          "tray could not read them" % (tag, unknown))
        warnings = []
        if feats != standard[tag]:
            warnings.append("trained on a non-standard input list %s (the "
                            "production's is %s)" % (feats, standard[tag]))
        found[tag] = dict(txt=txt, json=js, meta=meta, sha256=_sha256(txt),
                          warnings=warnings)
    if errors:
        raise ReleaseError("models:\n  " + "\n  ".join(errors))
    if not found:
        raise ReleaseError("--models %s holds neither L4_noise_model nor "
                           "L4_muon_model" % model_dir)
    if missing and not partial:
        raise ReleaseError(
            "--models %s has no L4_%s_model.txt/.json.  process_L4.py "
            "--apply-cut runs BOTH classifiers, so this release could not "
            "write the L4 scores or L4_oscNext_bool.  Train it, or pass "
            "--partial-models to ship what exists with models/README.md "
            "saying what is missing." % (model_dir, "_model / L4_".join(missing)))
    return found, missing


# ---------------------------------------------------------------------------
# README facts, from the sidecars
# ---------------------------------------------------------------------------

FLAVOUR = {"nue": "νe", "numu": "νμ", "nutau": "ντ"}

# What the technical note says about a GENIE set, by the last four digits of
# 1x####.  ONLY what the note says; anything else is "not identified".
NOTE_SETS = {
    "0511": ("a bulk-ice set with scattering −30 %", "Table 5"),
    "0519": ("a bulk-ice set with scattering −5 %", "Table 5"),
    "1122": ("a SPICE-BFR-v2 set", "Table 6"),
    "9002": ("a nominal set", "sec. 3.6.2"),
    "0000": ("a nominal set", "sec. 3.6.2"),
}


def _dataset_ids(patterns):
    ids = []
    for p in patterns:
        for m in re.findall(r"/(\d{4,6})/", p):
            if m not in ids:
                ids.append(m)
    return ids


def _note_set(ds):
    m = re.match(r"^1[246](\d{4})$", ds)
    return NOTE_SETS.get(m.group(1)) if m else None


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else \
        ", ".join(items[:-1]) + " and " + items[-1]


def _pct(x):
    return "%.1f %%" % (100.0 * x)


def text_at_default_cut(models):
    if not models:
        return ("Efficiency and rejection at the default cut: not recorded, "
                "no model was shipped with this release.")
    out = []
    for tag in TAGS:
        if tag not in models:
            out.append("The %s model is not shipped, so its efficiency is not "
                       "recorded." % tag)
            continue
        m = models[tag]["meta"]
        a = (m.get("metrics") or {}).get("at_default_cut")
        bg = "noise" if tag == "noise" else "muons"
        if not a:
            out.append("For our %s model the efficiency at %s is not recorded: "
                       "the model predates that record." % (tag, m.get("default_cut")))
        else:
            out.append("On its own test set (event counts), our %s model keeps "
                       "%s of the neutrinos and rejects %s of the %s at "
                       "P ≥ %.2f (%d background events left; trained %s)."
                       % (tag, _pct(a["eff"]), _pct(a["rej"]), bg, a["cut"],
                          a["bg_kept"], m.get("trained_at", "on an unrecorded date")))
    return " ".join(out)


def _provenance(models, tag):
    return (models.get(tag) or {}).get("meta", {}).get("provenance")


def text_signal(models):
    provs = [(t, _provenance(models, t)) for t in TAGS if t in models]
    if not provs:
        return ("Not recorded: no model was shipped with this release, so "
                "there is no training record to report the signal from.")
    known = [(t, p) for t, p in provs if p]
    if not known:
        return ("Not recorded: the shipped models predate provenance, so which "
                "signal sets and weights they were trained with is not known "
                "from the files.")
    parts = []
    signals = {json.dumps(p["signal"], sort_keys=True) for _, p in known}
    if len(signals) > 1:
        parts.append("[!] The noise and muon models were trained on DIFFERENT "
                     "signal samples; each is described below.")
    for tag, p in known if len(signals) > 1 else known[:1]:
        schemes = p.get("signal_weighting", {}).get("schemes", [])
        genie = schemes == ["genie"]
        sets, ids_all = [], []
        for s in p["signal"]:
            ids = _dataset_ids(s.get("l3", []))
            ids_all += ids
            sets.append("%s (%s)" % (" + ".join(ids) or "dataset not recorded",
                                     FLAVOUR.get(s["name"], s["name"])))
        who = "" if len(signals) == 1 else "For the %s model: " % tag
        parts.append("%sThe signal is %s%s sets %s, trained as one neutrino "
                     "class." % (who, "GENIE " if genie else "", p["production"],
                                 _join(sets)))
        ident = [(ds, _note_set(ds)) for ds in ids_all]
        said = ["%s (1x%s) is %s (%s)" % (ds, ds[2:], d, ref)
                for ds, (d, ref) in [(a, b) for a, b in ident if b]]
        unknown = [ds for ds, d in ident if not d]
        if said:
            parts.append("According to the note's tables, %s." % _join(said))
        if unknown:
            parts.append("%s %s not identified in the technical note."
                         % (_join(unknown), "is" if len(unknown) == 1 else "are"))
        if any(not d or d[0] != "a nominal set" for _, d in ident):
            parts.append("The production used the nominal sets (1X9002, later "
                         "1X0000).")
        w = p.get("signal_weighting", {})
        if genie and "norm" in w and "gamma" in w:
            parts.append(
                "Signal is weighted with a single unoscillated power law, "
                "%s·E^%s, identical for all flavours. The production trained "
                "with a different, physical weight (the power law was only "
                "added at L3 v01.03, after the L4 classifiers were trained, "
                "note Appendix F.1). So our classifiers see a different energy "
                "spectrum and flavour mix."
                % (("%g" % w["norm"]), ("%g" % w["gamma"]).replace("-", "−")))
        else:
            parts.append("Signal weighting scheme(s): %s (oscnext_l4/data.py); "
                         "not described further here." % ", ".join(schemes))
    missing = [t for t, p in provs if not p]
    if missing:
        parts.append("(The %s model predates provenance; not recorded for it.)"
                     % _join(missing))
    return " ".join(parts)


def text_muon_background(models):
    if "muon" not in models:
        return ("Not recorded: no muon model was shipped with this release.")
    p = _provenance(models, "muon")
    if not p:
        return ("Not recorded: the shipped muon model predates provenance, so "
                "which background it was trained against, and whether the "
                "noise cut was applied first, is not known from the files.")
    what = []
    for s in p.get("background", []):
        ids = _dataset_ids(s.get("l3", []))
        runs = sorted(set(re.findall(r"Run(\d{8})", " ".join(s.get("l3", [])))))
        kind = {"data": ("detector data, %d runs (the %s `%s` sample)"
                         % (len(runs), p["production"], s["name"]) if runs else
                         "detector data (the %s `%s` sample, %d L3 path "
                         "pattern(s), runs not recorded in them)"
                         % (p["production"], s["name"], len(s.get("l3", [])))),
                "muongun": "MuonGun simulation (`%s`, dataset %s)"
                           % (s["name"], "+".join(ids) or "not recorded"),
                "corsika": "CORSIKA simulation (`%s`, dataset %s)"
                           % (s["name"], "+".join(ids) or "not recorded")
                }.get(s.get("weight"), "`%s` (weight scheme %s)"
                      % (s["name"], s.get("weight")))
        what.append(kind)
    nc = p.get("noise_cut") or {}
    if nc.get("applied"):
        sha = nc.get("model_sha256") or ""
        shipped = (models.get("noise") or {}).get("sha256")
        same = ("the noise model shipped here" if sha and sha == shipped else
                "[!] NOT the noise model shipped here" if shipped else
                "a noise model not shipped here")
        cut = ("after the noise cut at P ≥ %.2f with noise model sha256 "
               "%s…, %s" % (nc["threshold"], sha[:12], same))
    elif nc.get("applied") is False:
        cut = "WITHOUT the noise cut the production applies first"
    else:
        cut = "with no record of whether the noise cut was applied"
    return "Our muon model was trained against %s, %s." % (_join(what or ["an unrecorded background"]), cut)


def text_models(models, missing):
    if not models:
        return ("No model is shipped with this release: `models/` holds only "
                "a README saying which files go there.  Until they are added, "
                "`--apply-cut` cannot run; everything else can.")
    rows = ["| | " + " | ".join(t for t in TAGS if t in models) + " |",
            "|---|" + "---|" * len(models)]

    def row(label, fn):
        rows.append("| %s | %s |" % (label, " | ".join(
            fn(models[t]) for t in TAGS if t in models)))

    def adc(m):
        a = (m["meta"].get("metrics") or {}).get("at_default_cut")
        return ("not recorded (predates the record)" if not a else
                "%s signal kept, %s background rejected at %.2f"
                % (_pct(a["eff"]), _pct(a["rej"]), a["cut"]))

    def prov(m):
        p = m["meta"].get("provenance")
        if not p:
            return "not recorded: this model predates provenance"
        return "%s; signal %s; background %s" % (
            p["production"], "+".join(s["name"] for s in p["signal"]),
            "+".join(s["name"] for s in p["background"]))

    row("files", lambda m: "`%s` + `.json`" % os.path.basename(m["txt"]))
    row("trained", lambda m: str(m["meta"].get("trained_at", "not recorded")))
    row("LightGBM", lambda m: str(m["meta"].get("lightgbm_version", "not recorded")))
    row("trees", lambda m: str(m["meta"].get("n_trees", "not recorded")))
    row("inputs", lambda m: ", ".join("`%s`" % f for f in m["meta"]["features"]))
    row("train / test events",
        lambda m: "%s / %s signal, %s / %s background" % tuple(
            m["meta"].get(k, "?") for k in ("n_train_sig", "n_test_sig",
                                            "n_train_bg", "n_test_bg")))
    row("at the default cut", adc)
    row("trained on", prov)
    row("sha256 (.txt)", lambda m: "`%s…`" % m["sha256"][:16])
    out = "\n".join(rows)
    notes = []
    for t in TAGS:
        for w in (models.get(t) or {}).get("warnings", []):
            notes.append("- **%s**: %s" % (t, w))
    if missing:
        notes.append("- **The %s model is NOT shipped.**  `--apply-cut` runs "
                     "both classifiers, so it cannot run until it is added "
                     "(`models/README.md`)." % _join(missing))
    return out + ("\n\n" + "\n".join(notes) if notes else "")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _fill(template, fields):
    for k, v in fields.items():
        template = template.replace("{{%s}}" % k, v)
    template = re.sub(r"<!-- Template:.*?-->\n*", "", template, flags=re.S)
    left = re.findall(r"\{\{[A-Z_]+\}\}|<fill[^>]*>|<model-dependent[^>]*>",
                      template)
    if left:
        raise ReleaseError("README fields left unfilled: %s" % sorted(set(left)))
    return template


def _git(*args):
    try:
        p = subprocess.run(["git", "-C", ROOT] + list(args),
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def _contents(files, external, models, build):
    lines = ["```"]
    for d in sorted({os.path.dirname(f) or "." for f in files}):
        lines.append("%s/" % d if d != "." else "./")
        for f in sorted(x for x in files if (os.path.dirname(x) or ".") == d):
            lines.append("    %s" % os.path.basename(f))
    lines.append("```")
    lines.append("")
    third = sorted(e for e in external
                   if e not in getattr(sys, "stdlib_module_names", ()))
    lines.append("Outside this directory and the standard library, the code "
                 "imports: %s.  (`I3Tray` is the pre-v1.5 location of "
                 "`icecube.icetray.I3Tray`, tried second; `h5py`, `simweights` "
                 "and `matplotlib` are optional.)"
                 % ", ".join("`%s`" % e for e in third))
    lines.append("")
    lines.append("Built %s from commit `%s`%s."
                 % (build["date"], build["commit"] or "unknown",
                    " **with uncommitted changes**" if build["dirty"] else ""))
    return "\n".join(lines)


def build(out, model_dir, partial):
    files, external, errors = closure(ROOT, ENTRY_SCRIPTS)
    if errors:
        raise ReleaseError("import closure:\n  " + "\n  ".join(errors))
    for name in ("README.md", "physics_caveats.md", "models_README.md"):
        if not os.path.exists(os.path.join(TEMPLATE_DIR, name)):
            raise ReleaseError("template release/%s is missing" % name)

    models, missing = ({}, list(TAGS))
    if model_dir:
        models, missing = check_models(model_dir, partial)

    dirty = _git("status", "--porcelain")
    info = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(dirty), "models": model_dir and os.path.abspath(model_dir)}

    out = os.path.abspath(out)
    if os.path.exists(out) and os.listdir(out) and \
            not os.path.exists(os.path.join(out, MARKER)):
        raise ReleaseError("%s exists and is not an earlier release (no %s); "
                           "refusing to replace it" % (out, MARKER))
    tmp = out + ".tmp-build"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        for rel in sorted(files):
            os.makedirs(os.path.join(tmp, os.path.dirname(rel)), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, rel), os.path.join(tmp, rel))

        os.makedirs(os.path.join(tmp, "models"))
        shipped = []
        for tag, m in models.items():
            for src in (m["txt"], m["json"]):
                shutil.copy2(src, os.path.join(tmp, "models", os.path.basename(src)))
                shipped.append(os.path.join("models", os.path.basename(src)))
        if missing:
            status = ("**Not shipped in this release: the %s model%s.**  Add "
                      "%s here before running `--apply-cut`."
                      % (_join(missing), "s" if len(missing) > 1 else "",
                         _join("`L4_%s_model.txt` + `.json`" % t for t in missing)))
            if models:
                status += ("  Shipped: %s."
                           % ", ".join("`%s`" % os.path.basename(f) for f in shipped))
            with open(os.path.join(TEMPLATE_DIR, "models_README.md")) as fh:
                txt = _fill(fh.read(), {"STATUS": status})
            with open(os.path.join(tmp, "models", "README.md"), "w") as fh:
                fh.write(txt)
            shipped.append(os.path.join("models", "README.md"))

        with open(os.path.join(TEMPLATE_DIR, "physics_caveats.md")) as fh:
            caveats = _fill(fh.read(), {
                "AT_DEFAULT_CUT": text_at_default_cut(models),
                "SIGNAL": text_signal(models),
                "MUON_BACKGROUND": text_muon_background(models)})
        with open(os.path.join(TEMPLATE_DIR, "README.md")) as fh:
            readme = fh.read()
        all_files = sorted(files) + shipped + ["README.md"]
        readme = _fill(readme, {
            "MODELS": text_models(models, missing),
            "PHYSICS_CAVEATS": caveats.strip(),
            "CONTENTS": _contents(all_files, external, models, info)})
        with open(os.path.join(tmp, "README.md"), "w") as fh:
            fh.write(readme)

        verify(tmp, files)
        with open(os.path.join(tmp, MARKER), "w") as fh:
            json.dump(dict(info, files=all_files), fh, indent=1)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if os.path.exists(out):
        shutil.rmtree(out)
    os.rename(tmp, out)
    return out, sorted(files), external, models, missing, info


def verify(tree, files):
    """
    On the COPY: the closure must reproduce itself inside the release (every
    import resolves there, no script or config file points outside it), and
    every file must compile.
    """
    entry = [f for f in ENTRY_SCRIPTS]
    again, _, errors = closure(tree, entry)
    if errors:
        raise ReleaseError("inside the release:\n  " + "\n  ".join(errors))
    extra = sorted(set(files) - again)
    lost = sorted(again - set(files))
    if extra or lost:
        raise ReleaseError("the release's own closure differs from the "
                           "source's: extra %s, missing %s" % (extra, lost))
    for rel in files:
        if rel.endswith(".py"):
            path = os.path.join(tree, rel)
            with open(path) as fh:
                try:
                    compile(fh.read(), path, "exec")
                except SyntaxError as e:
                    raise ReleaseError("%s does not compile: %s" % (rel, e))
    stray = [os.path.join(d, x) for d, dirs, fs in os.walk(tree)
             for x in dirs + fs if x == "__pycache__" or x.endswith(".pyc")]
    if stray:
        raise ReleaseError("byte code in the release: %s" % stray)


def main():
    ap = argparse.ArgumentParser(description="assemble the runnable release")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist",
                                                  "oscnext_l4_release"))
    ap.add_argument("--models", default=None,
                    help="directory holding L4_noise_model.txt/.json and "
                         "L4_muon_model.txt/.json (models never enter git)")
    ap.add_argument("--partial-models", action="store_true",
                    help="ship the models that exist even if one of the two "
                         "is missing; models/README.md says which")
    ap.add_argument("--list", action="store_true",
                    help="print the import closure and exit; builds nothing")
    args = ap.parse_args()

    try:
        if args.list:
            files, external, errors = closure(ROOT, ENTRY_SCRIPTS)
            for f in sorted(files):
                print(f)
            print("external:", ", ".join(sorted(external)))
            if errors:
                raise ReleaseError("import closure:\n  " + "\n  ".join(errors))
            return
        out, files, external, models, missing, info = build(
            args.out, args.models, args.partial_models)
    except ReleaseError as e:
        sys.exit("make_release: FAILED -- nothing was written.\n%s" % e)

    print("release: %s" % out)
    for f in files:
        print("    %s" % f)
    print("models : %s" % (", ".join(sorted(models)) or "none"))
    if missing:
        print("  [!] NOT shipped: the %s model%s -- --apply-cut cannot run "
              "until added (models/README.md says so)."
              % (_join(missing), "s" if len(missing) > 1 else ""))
    print("external imports (beyond the standard library): %s"
          % ", ".join(sorted(e for e in external if e not in
                             getattr(sys, "stdlib_module_names", ()))))
    if info["dirty"]:
        print("  [!] built from a checkout with uncommitted changes (%s)"
              % info["commit"])


if __name__ == "__main__":
    main()
