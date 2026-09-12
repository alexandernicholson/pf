#!/usr/bin/env python3
"""Build an offline benchmark explorer and a lossless numeric-leaf CSV.

The report uses Python's standard library and has no browser dependencies.
Raw result objects are embedded so new metrics remain available without editing
the report. JSON pointers in the CSV identify the exact original numeric leaf.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import gzip
import json
import math
from pathlib import Path
import sys


FIELDS = [
    "record", "iteration", "timestamp", "decision", "case", "threads",
    "configuration", "workload", "source", "metric", "pointer", "value",
]


def stable(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def first(obj, keys, default=""):
    for key in keys:
        value = obj.get(key)
        if value is not None and value != "":
            return value
    return default


def pointer_part(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def numeric_leaves(value, path=()):
    """Do not discard unknown metrics, zeroes, or array element identities."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, (*path, index))


def path_metric(path):
    return "/" + "/".join("[]" if isinstance(x, int) else pointer_part(x) for x in path)


def read_records(root):
    """Read canonical run results, avoiding large raw artifacts and snapshots."""
    files = [root] if root.is_file() else sorted(root.rglob("manifest.json"))
    if not files and root.is_dir():
        files = sorted(root.rglob("result.json")) + sorted(root.rglob("results.json")) + sorted(root.rglob("run.json"))
    records, warnings = [], []
    for path in files:
        relative_parts = (path.name,) if root.is_file() else path.relative_to(root).parts
        if any(part.lower().startswith("model-") or "-model-" in part.lower() for part in relative_parts):
            continue
        try:
            source_text = path.read_text()
            payload = json.loads(source_text)
        except (ValueError, OSError) as exc:
            warnings.append(f"{path}: {exc}")
            continue
        if not isinstance(payload, dict):
            warnings.append(f"{path}: skipped non-object result")
            continue
        raw_files = {path.name: source_text}
        if path.name == "manifest.json":
            payload = {"manifest": payload}
            # Keep numerical kernel evidence. Model corpus files are outside
            # this report's scope; every unknown field within these evidence
            # records is still preserved.
            for companion in sorted(path.parent.glob("*.json")):
                if companion == path:
                    continue
                if companion.stem not in ("records", "summary", "host-before", "host-after", "decision"):
                    continue
                try:
                    companion_text = companion.read_text()
                    payload[companion.stem] = json.loads(companion_text)
                    raw_files[companion.name] = companion_text
                except (ValueError, OSError) as exc:
                    warnings.append(f"{companion}: {exc}")
        relative = path.name if root.is_file() else str(path.relative_to(root))
        records.append({"file": relative, "payload": payload, "raw_files": raw_files})
    return records, warnings


def context_for(record):
    payload = record["payload"]
    obj = payload.get("manifest", payload)
    build = obj.get("build", {})
    if not isinstance(build, dict):
        build = {"build": build}
    config = {
        key: build[key] for key in ("compiler", "compiler_version", "flags", "cflags")
        if key in build
    }
    for key in ("flags", "cflags", "compiler"):
        if key in obj:
            config[key] = obj[key]
    arguments = obj.get("arguments", {})
    for key in ("cc", "cflags", "ldflags"):
        if key in arguments:
            config[key] = arguments[key]
    compiler = obj.get("compiler_version", {})
    if isinstance(compiler, dict) and compiler.get("stdout"):
        config["compiler_version"] = compiler["stdout"].splitlines()[0]
    host = payload.get("host-before", {})
    if host:
        uname = host.get("uname", [])
        if len(uname) >= 5:
            config["host"] = [uname[0], uname[1], uname[2], uname[4]]
        config["affinity"] = host.get("affinity")
        config["cpu_quota"] = host.get("raw_files", {}).get("/sys/fs/cgroup/cpu.max")
    source = first(obj, ("source_sha256", "source_hash", "revision", "commit", "git_commit"))
    if not source:
        source = first(build, ("source_sha256", "source_hash", "revision", "commit"))
    if not source:
        source = obj.get("sources", {}).get("pf.c", {}).get("sha256", "")
    decision = payload.get("decision", first(obj, ("decision", "status"), "unclassified"))
    if isinstance(decision, dict):
        decision = first(decision, ("status", "decision", "outcome"), stable(decision))
    threads = first(obj, ("threads", "pf_threads"), "n/a")
    if "parallel" in arguments:
        threads = "1" if not arguments["parallel"] else host.get("environment", {}).get("PF_THREADS", "auto")
    for entry in payload.get("records", []):
        if isinstance(entry, dict) and entry.get("type") == "metadata":
            threads = first(entry, ("threads", "effective_threads", "pool_threads"), threads)
    return {
        "record": record["file"],
        "iteration": stable(first(obj, ("iteration", "label", "name", "run_id", "id"), Path(record["file"]).parent.name)),
        "timestamp": stable(first(obj, ("timestamp", "started_at", "started_utc", "created_at", "utc"))),
        "decision": stable(decision),
        "case": "run metadata",
        "threads": stable(threads),
        "configuration": stable(config) if config else "unspecified",
        "workload": "{}",
        "source": stable(source) if source else "unspecified",
    }


def rows_for(record):
    """Attach contextual dimensions while preserving every finite numeric leaf."""
    base = context_for(record)
    payload = record["payload"]
    raw_records = payload.get("records", [])
    suite = {}
    workloads = {}
    for entry in raw_records if isinstance(raw_records, list) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "metadata":
            suite = {k: v for k, v in entry.items() if k not in ("type", "reps", "target_ms", "profile", "threads", "effective_threads", "pool_threads")}
        if entry.get("type") == "workload" and "case" in entry:
            workloads[entry["case"]] = {k: v for k, v in entry.items() if k not in ("type", "case", "loops_per_rep", "calibration_two_calls_ns")}
    rows = []
    for path, value in numeric_leaves(payload):
        context = dict(base)
        obj = payload
        workload = dict(suite)
        metric_path = path
        if len(path) >= 2 and path[0] == "records" and isinstance(path[1], int):
            entry = raw_records[path[1]]
            if isinstance(entry, dict):
                context["case"] = str(entry.get("case", entry.get("type", "run metadata")))
                metric_path = ("records", entry.get("type", "unknown"), *path[2:])
        elif len(path) >= 2 and path[0] == "summary":
            context["case"] = str(path[1])
            metric_path = ("summary", "{case}", *path[2:])
        elif len(path) >= 3 and path[:2] == ("manifest", "summary"):
            context["case"] = str(path[2])
            metric_path = ("manifest", "summary", "{case}", *path[3:])
        for part in path:
            if isinstance(obj, dict):
                case = first(obj, ("case", "case_name", "kernel", "benchmark"))
                if case and not isinstance(case, (dict, list)):
                    context["case"] = stable(case)
                threads = first(obj, ("threads", "pf_threads", "thread_count"))
                if threads != "":
                    context["threads"] = stable(threads)
                # Workload dimensions are conservative: unlike timestamps or
                # measured throughput, these values affect comparability.
                for key in ("shape", "dimensions", "batch", "batch_size", "B", "M", "N", "K", "n_in", "n_out", "tokens", "seed", "mode", "dtype", "implementation"):
                    if key in obj and isinstance(obj[key], (str, int, float, list, dict)):
                        workload[key] = obj[key]
            obj = obj[part]
        workload.update(workloads.get(context["case"], {}))
        context["workload"] = stable(workload)
        rows.append({
            **context,
            "metric": path_metric(metric_path),
            "pointer": "/" + "/".join(pointer_part(x) for x in path),
            "value": value,
            "value_exact": str(value),
        })
    return rows


HTML = r'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PF cloud hillclimb — benchmark explorer</title>
<style>
:root{color-scheme:light dark;--bg:#f5f7fa;--panel:#fff;--ink:#15243b;--muted:#52627b;--line:#dce3ed;--accent:#005eb8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:30px}h1{font-size:27px;letter-spacing:-.7px;margin:0 0 7px}p{margin:0 0 16px;color:var(--muted)}.panel{padding:20px;background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:18px}.controls{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:12px}label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:12px;font-weight:600}select,button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:5px;padding:8px;max-width:100%}select{width:100%}button{cursor:pointer}button:hover{border-color:var(--accent)}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:15px}.facts{display:flex;gap:25px;flex-wrap:wrap;padding-top:12px;color:var(--muted)}.facts b{color:var(--ink)}#chart{width:100%;height:auto;display:block;min-height:240px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}td.wrap{white-space:normal;min-width:260px;max-width:480px}code{font:12px ui-monospace,monospace}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}#legend{display:flex;gap:8px 18px;flex-wrap:wrap;margin:8px 0 12px;font-size:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.small{font-size:12px;color:var(--muted)}.warn{color:#a94e00}.metric-key{overflow-wrap:anywhere}.subcontrol{display:flex;flex-direction:row;align-items:center;font-size:13px;gap:6px;margin-top:12px}input{accent-color:var(--accent)}h2{font-size:18px;margin:0 0 12px}@media(max-width:800px){main{padding:15px}.controls{grid-template-columns:1fr 1fr}}@media(prefers-color-scheme:dark){:root{--bg:#101722;--panel:#182231;--ink:#e9eef8;--muted:#a9b9d0;--line:#33435a;--accent:#70b5ff}.warn{color:#ffbb70}}
</style>
<main>
<h1>PF cloud hillclimb</h1>
<p>Deterministic synthetic kernel measurements. These runs do not establish full-model throughput or PII-label accuracy.</p>
<section class="panel">
<div class="controls">
<label>Numeric metric<select id="metric"></select></label>
<label>Case<select id="case"></select></label>
<label>Threads<select id="threads"></select></label>
<label>Decision / run status<select id="decision"></select></label>
</div>
<label class="subcontrol"><input type="checkbox" id="connect">Connect iterations within the same case, thread count, compiler flags and workload</label>
<div class="actions"><button id="raw-download">Download complete structured results</button><button id="csv-download">Download all numeric leaves (CSV)</button><button id="selection-download">Download selected metric (CSV)</button></div>
<div class="facts" id="facts"></div>
</section>
<section class="panel">
<div id="metric-name" class="metric-key"></div>
<p class="small">Each point is the median of matching numeric leaves in one run; whiskers show the observed minimum and maximum, not a confidence interval. Source fingerprints remain attached to each point. Array indexes are normalized only in metric names; exact JSON pointers remain in the CSV. Lines are disabled by default. Wall/CPU totals and resource counters are per calibrated sample; use ns_per_call to compare latency when loop counts change.</p>
<svg id="chart" viewBox="0 0 1320 460" role="img" aria-label="Selected metric across benchmark runs"></svg>
<div id="legend"></div><div id="chart-note" class="small"></div>
</section>
<section class="panel"><h2>Selected metric</h2><div class="scroll"><table><thead><tr><th>Run</th><th>Case / threads</th><th>Median</th><th>Min – max</th><th>n</th><th>Status</th><th>Source</th><th>Configuration / workload</th></tr></thead><tbody id="table"></tbody></table></div></section>
<section class="panel"><details open><summary>Iteration decisions and experiment context</summary><pre id="log"></pre></details></section>
<section class="panel"><details><summary>Full structured result records</summary><p class="small">Compiler logs, source snapshots and raw process streams remain alongside their result files in the benchmark archive. This view embeds the structured JSON records exactly as loaded.</p><pre id="raw"></pre></details></section>
<p id="footer" class="small">Loading embedded benchmark data. This report requires a browser with DecompressionStream support.</p>
</main>
<script id="benchmark-data" type="application/gzip">__DATA__</script>
<script>
"use strict";
(async()=>{
const compressed=Uint8Array.from(atob(document.getElementById('benchmark-data').textContent.trim()),c=>c.charCodeAt(0));
const stream=new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
const DATA=JSON.parse(await new Response(stream).text());
const $=id=>document.getElementById(id), rows=DATA.rows, records=DATA.records;
const colors=['#2374c9','#d56a13','#179879','#9257c2','#d14e6a','#608c0d','#188fab','#946340','#717787','#ba4996'];
const unique=a=>[...new Set(a)].sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true}));
const format=x=>Number.isInteger(x)&&Math.abs(x)<1e7?String(x):Number(x).toPrecision(6).replace(/(\.\d*?)0+(e|$)/,'$1$2').replace(/\.(e|$)/,'$1');
const fmtLong=x=>String(x);
function option(select,value,label=value){const e=document.createElement('option');e.value=value;e.textContent=label;select.append(e)}
const metrics=unique(rows.map(x=>x.metric));
metrics.forEach(x=>option($('metric'),x));
const preferred=metrics.find(x=>x==='/records/sample/ns_per_call')||metrics.find(x=>/sample.*(gflops|gflop_s|gb_per_s|tokens_per_s)/i.test(x))||metrics.find(x=>/(gflops|gflop_s|ns_per|elapsed|wall|seconds)/i.test(x));
if(preferred)$('metric').value=preferred;
for(const key of ['case','threads','decision']){option($(key),'','All');unique(rows.map(x=>x[key])).forEach(x=>option($(key),x))}
if(rows.some(r=>r.case==='layer_packed_t256_s32'))$('case').value='layer_packed_t256_s32';
if(rows.some(r=>r.threads==='8'))$('threads').value='8';
const order=new Map(records.map((r,i)=>[r.file,i]));
const recordName=new Map(records.map(r=>[r.file,(rows.find(x=>x.record===r.file)||{}).iteration||r.file]));
function selected(){return rows.filter(r=>r.metric===$('metric').value&&['case','threads','decision'].every(k=>!$(k).value||r[k]===$(k).value))}
function seriesKey(r){return JSON.stringify([r.case,r.threads,r.configuration,r.workload])}
function summarize(filtered){const buckets=new Map();for(const row of filtered){const key=JSON.stringify([row.record,seriesKey(row)]);if(!buckets.has(key))buckets.set(key,{...row,values:[]});buckets.get(key).values.push(row.value)}return [...buckets.values()].map(r=>{r.values.sort((a,b)=>a-b);const n=r.values.length;return {...r,n,min:r.values[0],max:r.values[n-1],median:(r.values[Math.floor((n-1)/2)]+r.values[Math.floor(n/2)])/2}}).sort((a,b)=>order.get(a.record)-order.get(b.record)||seriesKey(a).localeCompare(seriesKey(b)))}
function el(name,attrs={},parent=$('chart')){const e=document.createElementNS('http://www.w3.org/2000/svg',name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e}
function text(label,x,y,attrs={}){const e=el('text',{x,y,fill:'var(--muted)','font-size':12,...attrs});e.textContent=label;return e}
function td(tr,value,cls){const e=document.createElement('td');e.textContent=value;if(cls)e.className=cls;tr.append(e);return e}
function render(){const filtered=selected(),groups=summarize(filtered),series=unique(groups.map(seriesKey)),visible=unique(groups.map(x=>x.record)).sort((a,b)=>order.get(a)-order.get(b));
$('facts').replaceChildren();for(const [value,label]of [[records.length,'archived runs'],[metrics.length,'numeric metric paths'],[rows.length,'numeric leaves'],[groups.length,'selected points']]){const span=document.createElement('span'),b=document.createElement('b');b.textContent=value.toLocaleString();span.append(b,` ${label}`);$('facts').append(span)}
$('metric-name').textContent=$('metric').value||'No numeric metrics are available';$('chart').replaceChildren();$('table').replaceChildren();$('legend').replaceChildren();$('chart-note').textContent='';
if(!groups.length){text('No matching numeric measurements.',70,210);return}
const left=112,right=1280,top=25,bottom=350,width=right-left,height=bottom-top;
let lo=Math.min(...groups.map(x=>x.min)),hi=Math.max(...groups.map(x=>x.max));if(lo===hi){const d=Math.abs(lo)*.05||1;lo-=d;hi+=d}else{const p=(hi-lo)*.07;lo-=p;hi+=p}
const x=i=>visible.length===1?(left+right)/2:left+i*width/(visible.length-1),y=v=>bottom-(v-lo)/(hi-lo)*height;
for(let i=0;i<=5;i++){const value=lo+(hi-lo)*i/5,py=y(value);el('line',{x1:left,y1:py,x2:right,y2:py,stroke:'var(--line)'});text(format(value),left-12,py+4,{'text-anchor':'end'})}
el('line',{x1:left,y1:top,x2:left,y2:bottom,stroke:'var(--line)'});
visible.forEach((name,i)=>{const label=recordName.get(name);text(label.length>33?label.slice(0,30)+'…':label,x(i),bottom+20,{'text-anchor':'end',transform:`rotate(-30 ${x(i)} ${bottom+20})`})});
for(let si=0;si<series.length;si++){const key=series[si],points=groups.filter(r=>seriesKey(r)===key),color=colors[si%colors.length];const legend=document.createElement('span'),dot=document.createElement('i');dot.className='dot';dot.style.background=color;legend.append(dot,`${si+1}: ${points[0].case} · ${points[0].threads} threads`);legend.title=points[0].configuration+' / '+points[0].workload;$('legend').append(legend);
if($('connect').checked&&points.length>1){el('polyline',{points:points.map(r=>`${x(visible.indexOf(r.record))},${y(r.median)}`).join(' '),fill:'none',stroke:color,'stroke-width':1.5,opacity:.6})}
for(const r of points){const px=x(visible.indexOf(r.record)),py=y(r.median);el('line',{x1:px,y1:y(r.min),x2:px,y2:y(r.max),stroke:color,'stroke-width':2});for(const v of [r.min,r.max])el('line',{x1:px-4,y1:y(v),x2:px+4,y2:y(v),stroke:color});const point=el('circle',{cx:px,cy:py,r:4.8,fill:color,stroke:'var(--panel)','stroke-width':1.3,tabindex:0});const title=el('title',{},point);title.textContent=`${r.iteration}\n${r.case} / ${r.threads} threads\nMedian: ${fmtLong(r.median)}\nRange: ${fmtLong(r.min)} – ${fmtLong(r.max)} (n=${r.n})\nDecision/status: ${r.decision}\nSource: ${r.source}\n${r.configuration}\n${r.workload}`}}
for(const r of groups){const tr=document.createElement('tr');td(tr,r.iteration).title=r.record+'\n'+r.timestamp;td(tr,`${r.case} / ${r.threads}`);td(tr,format(r.median)).title=fmtLong(r.median);td(tr,`${format(r.min)} – ${format(r.max)}`).title=`${fmtLong(r.min)} – ${fmtLong(r.max)}`;td(tr,r.n);td(tr,r.decision);td(tr,r.source.length>24?r.source.slice(0,20)+'…':r.source).title=r.source;td(tr,r.configuration+'\n'+r.workload,'wrap');$('table').append(tr)}
$('chart-note').textContent=`${series.length} distinct configuration/workload series. Values retain the unit expressed in the selected JSON metric key. Medians summarize the selected leaves only; aggregate fields and raw sample fields are separate metrics. Hover or focus a point for exact values and source provenance.`}
function download(name,type,content){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function csv(data){const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';return [DATA.csv_fields,...data.map(r=>DATA.csv_fields.map(k=>k==='value'?r.value_exact:r[k]))].map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n'}
$('raw-download').onclick=()=>download('pf-benchmark-results.json','application/json',JSON.stringify({generated_at:DATA.generated_at,records:records.map(({file,raw_files})=>({file,raw_files})),iteration_log:DATA.iteration_log,warnings:DATA.warnings},null,2));
$('csv-download').onclick=()=>download('pf-benchmark-metrics.csv','text/csv',csv(rows));$('selection-download').onclick=()=>download('pf-selected-metric.csv','text/csv',csv(selected()));
for(const key of ['metric','case','threads','decision','connect'])$(key).addEventListener('change',render);
$('log').textContent=DATA.iteration_log||'No iteration log was found.';$('raw').textContent=JSON.stringify(records,null,2);$('footer').textContent=`Generated ${DATA.generated_at}. No external scripts or network requests. ${DATA.warnings.length?('Warnings: '+DATA.warnings.join('; ')):''}`;render();
})().catch(error=>{document.getElementById('footer').textContent='The report could not load: '+String(error);});
</script>
</html>
'''


def main(argv=None):
    bench = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=bench / "results", help="Run archive directory or one result JSON")
    parser.add_argument("--output", type=Path, default=bench / "report.html")
    parser.add_argument("--csv", type=Path, default=bench / "metrics.csv")
    parser.add_argument("--log", type=Path, default=bench / "ITERATIONS.md")
    args = parser.parse_args(argv)
    records, warnings = read_records(args.data)
    rows = [row for record in records for row in rows_for(record)]
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records,
        "rows": rows,
        "csv_fields": FIELDS,
        "iteration_log": args.log.read_text() if args.log.exists() else "",
        "warnings": warnings,
    }
    # Compact repeated host/context evidence without dropping unknown fields.
    # Base64 cannot terminate a script element, even when raw logs contain HTML.
    raw_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    encoded = base64.b64encode(gzip.compress(raw_json, compresslevel=6, mtime=0)).decode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", encoded))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    csv_gz = Path(str(args.csv) + ".gz")
    csv_gz.write_bytes(gzip.compress(args.csv.read_bytes(), compresslevel=6, mtime=0))
    print(f"{len(records)} runs, {len(rows)} numeric leaves, {len({r['metric'] for r in rows})} metric paths")
    print(args.output)
    print(args.csv)
    print(csv_gz)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
