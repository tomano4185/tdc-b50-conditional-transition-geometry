#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PUB-001 — frozen B50 operator-contract extractor

Read-only audit tool. It does not modify the B50 source or scientific outputs.
It extracts exact function bodies needed to close the manuscript's source-lock.
"""

from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

REQUIRED_FUNCTIONS = [
    "load_full_range_primes",
    "primes_to_gaps",
    "build_markov_operator",
    "pca_decompose",
    "safe_corr",
]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp,path)

def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True))

def get_function_node(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name:
            return node
    return None

def source_segment(lines: List[str], node: ast.AST) -> str:
    start=int(node.lineno); end=int(getattr(node,"end_lineno",node.lineno))
    return "".join(lines[start-1:end])

class FactVisitor(ast.NodeVisitor):
    def __init__(self): self.facts=[]
    def add(self,kind,node,detail):
        self.facts.append({"kind":kind,"line":int(getattr(node,"lineno",-1)),"detail":detail})
    def visit_BinOp(self,node):
        if isinstance(node.op,ast.Mod): self.add("modulo",node,ast.unparse(node))
        if isinstance(node.op,ast.FloorDiv): self.add("floor_division",node,ast.unparse(node))
        self.generic_visit(node)
    def visit_Call(self,node):
        try: name=ast.unparse(node.func)
        except Exception: name=""
        if any(tok in name for tok in ("digitize","histogram","bincount","searchsorted","clip","minimum","maximum","min","max","sum","reshape","unique","sort","diff","zeros","ones")):
            self.add("call",node,ast.unparse(node))
        self.generic_visit(node)
    def visit_Assign(self,node):
        txt=ast.unparse(node); low=txt.lower()
        if any(tok in low for tok in ("row","norm","count","bin","transition","matrix")):
            self.add("assignment",node,txt)
        self.generic_visit(node)
    def visit_AugAssign(self,node):
        self.add("augassign",node,ast.unparse(node)); self.generic_visit(node)

def import_module(path: Path):
    spec=importlib.util.spec_from_file_location("pub001_frozen_b50",str(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def dynamic_probe(mod) -> Dict[str,Any]:
    out={"available":False,"warning":"Descriptive only; exact source is authoritative."}
    if not all(hasattr(mod,x) for x in ("primes_to_gaps","build_markov_operator")):
        out["reason"]="required functions unavailable after import"; return out
    try:
        primes=np.array([2,3,5,7,11,13,17,19,23,29,31,37],dtype=np.int64)
        gaps=np.asarray(mod.primes_to_gaps(primes))
        P=np.asarray(mod.build_markov_operator(gaps,n_bins=64),dtype=float)
        rs=P.sum(axis=1) if P.ndim==2 else np.array([])
        nz=rs[np.abs(rs)>1e-15]
        out.update({
            "available":True,"probe_primes":primes.tolist(),"probe_gaps":gaps.tolist(),
            "operator_shape":list(P.shape),"operator_dtype":str(P.dtype),
            "finite":bool(np.isfinite(P).all()),
            "row_sum_min_nonzero":float(np.min(nz)) if len(nz) else None,
            "row_sum_max_nonzero":float(np.max(nz)) if len(nz) else None,
            "nonzero_entry_count":int(np.count_nonzero(P)),
            "operator_preview_nonzero":[
                {"row":int(i),"col":int(j),"value":float(P[i,j])}
                for i,j in np.argwhere(np.abs(P)>1e-15)[:30]
            ] if P.ndim==2 else [],
        })
    except Exception as exc:
        out["reason"]=repr(exc)
    return out

def main()->int:
    ap=argparse.ArgumentParser(description="PUB-001 frozen B50 operator-contract extractor")
    ap.add_argument("--b50-script",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--no-import-probe",action="store_true")
    args=ap.parse_args()
    src=args.b50_script.resolve(); out=args.output_dir.resolve()
    if not src.exists(): raise FileNotFoundError(src)
    out.mkdir(parents=True,exist_ok=True)
    text=src.read_text(encoding="utf-8"); lines=text.splitlines(keepends=True)
    tree=ast.parse(text,filename=str(src))
    extracted={}; snippets=[]
    for name in REQUIRED_FUNCTIONS:
        node=get_function_node(tree,name)
        if node is None:
            extracted[name]={"found":False,"source":None,"facts":[]}; continue
        seg=source_segment(lines,node); fv=FactVisitor(); fv.visit(node)
        rec={"found":True,"lineno":int(node.lineno),"end_lineno":int(getattr(node,"end_lineno",node.lineno)),"source":seg,"facts":fv.facts}
        extracted[name]=rec
        snippets.append(f"\n{'='*80}\n{name} [lines {rec['lineno']}-{rec['end_lineno']}]\n{'='*80}\n{seg}\n")
    missing=[k for k,v in extracted.items() if not v["found"]]
    if args.no_import_probe:
        probe={"available":False,"skipped":True}
    else:
        try: probe=dynamic_probe(import_module(src)); probe["skipped"]=False
        except Exception as exc: probe={"available":False,"skipped":False,"reason":repr(exc),"warning":"Static extraction remains authoritative."}
    contract={
        "tool":"PUB-001 frozen B50 operator-contract extractor",
        "b50_script":str(src),"b50_script_sha256":sha256_file(src),
        "file_bytes":src.stat().st_size,"file_lines":len(lines),
        "required_functions":REQUIRED_FUNCTIONS,"missing_functions":missing,
        "source_contract_complete":len(missing)==0,
        "functions":extracted,"dynamic_probe":probe,
        "publication_rule":"Use exact extracted source to state b_64 and normalization R; never substitute behavior inferred only from the probe."
    }
    atomic_json(out/"pub001_b50_operator_contract.json",contract)
    atomic_text(out/"pub001_b50_source_snippets.txt","".join(snippets))
    md=["# PUB-001 — Frozen B50 operator contract","",f"- Source: `{src}`",f"- SHA-256: `{contract['b50_script_sha256']}`",f"- Required functions found: {len(REQUIRED_FUNCTIONS)-len(missing)}/{len(REQUIRED_FUNCTIONS)}",f"- Source contract complete: **{contract['source_contract_complete']}**","","## Function locations"]
    for name in REQUIRED_FUNCTIONS:
        rec=extracted[name]
        md.append(f"- `{name}`: lines {rec['lineno']}-{rec['end_lineno']}" if rec["found"] else f"- `{name}`: **NOT FOUND**")
    md += ["","## Static structural facts for build_markov_operator"]
    bm=extracted.get("build_markov_operator",{})
    if bm.get("found") and bm.get("facts"):
        md += [f"- line {f['line']}: `{f['detail']}`" for f in bm["facts"]]
    else:
        md.append("- No selected structural facts available; inspect verbatim source snippet.")
    md += ["","## Dynamic probe","Descriptive only; source text is authoritative.","```json",json.dumps(probe,ensure_ascii=False,indent=2,allow_nan=True),"```","","## Publication closure rule","Replace the manuscript SOURCE-LOCK only after reviewing the exact extracted `build_markov_operator` body. State the mapping and normalization exactly as implemented; do not retune or simplify the frozen pipeline."]
    atomic_text(out/"pub001_b50_operator_contract.md","\n".join(md)+"\n")
    print("PUB-001 source contract extraction complete")
    print("source:",src); print("sha256:",contract["b50_script_sha256"])
    print("functions found:",len(REQUIRED_FUNCTIONS)-len(missing),"/",len(REQUIRED_FUNCTIONS))
    print("contract complete:",contract["source_contract_complete"]); print("outputs:",out)
    return 0 if not missing else 2

if __name__=="__main__":
    raise SystemExit(main())
