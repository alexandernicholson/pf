#!/usr/bin/env python3
"""Load generator and checker for `pf --serve` (Python stdlib only).

Subcommands
  run     closed-loop load: C concurrent clients (threads spread over worker
          processes), each sending POST /redact {"text": doc} (or {"texts": [B docs]})
          back to back for warmup+duration seconds. Requests completing inside the
          measured window count. Every response is checked against a reference
          masked file (one line per corpus line, e.g. `pf_cuda --lines -f corpus`).
  exact   one POST /redact {"texts": [docs of a subset]} to an idle server; every
          element must equal the reference for that subset byte for byte.
  sample  write a subset of corpus lines (every k-th) plus its indices.
  report  merge run/exact JSONs and server stats into results.json.

pf's server answers with `Connection: close`, so each request opens its own TCP
connection; the latency figure includes connect.
"""
import argparse, json, multiprocessing as mp, os, random, re, socket, sys, threading, time


def load_lines(path):
    with open(path, 'rb') as f:
        data = f.read()
    lines = data.split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    return lines


def build_request(port, route, payload, keepalive=False):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8', 'surrogateescape')
    head = ('POST %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n'
            'Content-Length: %d\r\nConnection: %s\r\n\r\n'
            % (route, port, len(body), 'keep-alive' if keepalive else 'close')).encode()
    return head + body


class KeepAlive:
    """one persistent connection; reconnects when the server closes it"""
    def __init__(self, port, timeout):
        self.port, self.timeout, self.s, self.buf = port, timeout, None, b''

    def roundtrip(self, req):
        if self.s is None:
            self.s = socket.create_connection(('127.0.0.1', self.port), timeout=self.timeout)
            self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.buf = b''
        try:
            self.s.sendall(req)
            while b'\r\n\r\n' not in self.buf:
                b = self.s.recv(262144)
                if not b:
                    raise ConnectionError('closed before response header')
                self.buf += b
            he = self.buf.index(b'\r\n\r\n')
            head = self.buf[:he].decode('latin-1').lower()
            m = re.search(r'content-length:\s*(\d+)', head)
            n = int(m.group(1)) if m else 0
            while len(self.buf) < he + 4 + n:
                b = self.s.recv(262144)
                if not b:
                    raise ConnectionError('closed mid-body')
                self.buf += b
            body = self.buf[he + 4:he + 4 + n]
            self.buf = self.buf[he + 4 + n:]
            status = int(head[9:12]) if head.startswith('http/') else 0
            if 'connection: close' in head:
                self.close()
            return status, body
        except Exception:
            self.close()
            raise

    def close(self):
        if self.s is not None:
            self.s.close()
        self.s, self.buf = None, b''


def http_roundtrip(port, req, timeout):
    s = socket.create_connection(('127.0.0.1', port), timeout=timeout)
    try:
        s.sendall(req)
        chunks = []
        while True:
            b = s.recv(262144)
            if not b:
                break
            chunks.append(b)
    finally:
        s.close()
    resp = b''.join(chunks)
    he = resp.find(b'\r\n\r\n')
    if he < 0:
        return 0, b''
    status = int(resp[9:12]) if resp[:5] == b'HTTP/' else 0
    return status, resp[he + 4:]


def doc_text(line):
    return line.decode('utf-8', 'surrogateescape')


def compare(body, idxs, ref, batched):
    """-> (n_verified, n_match, mismatched indices)"""
    try:
        obj = json.loads(body.decode('utf-8', 'surrogateescape'))
        outs = obj['texts'] if batched else [obj['text']]
    except Exception:
        return len(idxs), 0, list(idxs)
    if len(outs) != len(idxs):
        return len(idxs), 0, list(idxs)
    nm, bad = 0, []
    for o, i in zip(outs, idxs):
        if ref is None:
            continue
        if o.encode('utf-8', 'surrogateescape') == ref[i]:
            nm += 1
        else:
            bad.append(i)
    return (len(idxs) if ref is not None else 0), nm, bad


def worker(a, gids, order, t0, q):
    corpus = load_lines(a.corpus)
    ntok = [int(x) for x in load_lines(a.ntok)]
    ref = load_lines(a.ref) if a.ref else None
    C, B = a.conc, a.batch_docs
    w0, w1 = t0 + a.warmup, t0 + a.warmup + a.duration
    lock = threading.Lock()
    recs, stats = [], {'errors': 0, 'err_samples': [], 'verified': 0, 'match': 0,
                       'mismatch_idx': [], 'sent': 0, 'late': 0}

    def client(g):
        r = 0
        ka = KeepAlive(a.port, a.timeout) if a.keepalive else None
        while True:
            now = time.time()
            if now >= w1:
                break
            pos = (g + r * C) * B
            idxs = [order[(pos + j) % len(order)] for j in range(B)]
            r += 1
            if a.batch_docs > 1 or a.texts:
                payload = {'texts': [doc_text(corpus[i]) for i in idxs]}
            else:
                payload = {'text': doc_text(corpus[idxs[0]])}
            req = build_request(a.port, a.route, payload, a.keepalive)
            ts = time.time()
            try:
                st, body = ka.roundtrip(req) if ka else http_roundtrip(a.port, req, a.timeout)
                err = None if st == 200 else 'status %d' % st
            except Exception as e:  # noqa: BLE001
                st, body, err = 0, b'', repr(e)
            te = time.time()
            if err:
                with lock:
                    stats['errors'] += 1
                    if len(stats['err_samples']) < 5:
                        stats['err_samples'].append(err)
                continue
            nv, nm, bad = compare(body, idxs, ref, B > 1 or a.texts)
            inwin = w0 <= te <= w1
            with lock:
                stats['sent'] += 1
                stats['verified'] += nv
                stats['match'] += nm
                if len(stats['mismatch_idx']) < 200:
                    stats['mismatch_idx'] += bad[:200 - len(stats['mismatch_idx'])]
                if inwin:
                    recs.append((te - ts, len(idxs), sum(ntok[i] for i in idxs)))
                elif te > w1:
                    stats['late'] += 1
        if ka:
            ka.close()

    while time.time() < t0:
        time.sleep(0.001)
    ths = [threading.Thread(target=client, args=(g,), daemon=True) for g in gids]
    for t in ths:
        t.start()
    for t in ths:
        t.join(a.duration + a.warmup + a.timeout + 30)
    q.put((recs, stats))


def server_stats(port):
    try:
        st, body = http_roundtrip(port, b'GET /stats HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n', 5)
        d = json.loads(body)
        return d if 'forwards' in d else None
    except Exception:  # noqa: BLE001
        return None


def stats_delta(s0, s1):
    if not s0 or not s1:
        return None
    d = {k: s1[k] - s0[k] for k in ('forwards', 'jobs', 'docs', 'tokens', 'run_docs_ms', 'linger_ms', 't_ms')}
    f = d['forwards']
    d.update({'mean_tok_per_forward': d['tokens'] / f if f else None,
              'mean_jobs_per_forward': d['jobs'] / f if f else None,
              'mean_run_docs_ms': d['run_docs_ms'] / f if f else None,
              'run_docs_busy_pct': 100.0 * d['run_docs_ms'] / d['t_ms'] if d['t_ms'] else None,
              'linger_pct': 100.0 * d['linger_ms'] / d['t_ms'] if d['t_ms'] else None})
    return d


def pct(xs, p):
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def cmd_run(a):
    corpus = load_lines(a.corpus)
    ntok = [int(x) for x in load_lines(a.ntok)]
    if len(ntok) != len(corpus):
        sys.exit('ntok file has %d lines, corpus %d' % (len(ntok), len(corpus)))
    if a.ref:
        nref = len(load_lines(a.ref))
        if nref != len(corpus):
            sys.exit('ref file has %d lines, corpus %d' % (nref, len(corpus)))
    order = [i for i in range(len(corpus)) if not a.max_tok or ntok[i] <= a.max_tok]
    random.Random(a.seed).shuffle(order)
    nproc = max(1, min(a.procs, a.conc))
    groups = [list(range(p, a.conc, nproc)) for p in range(nproc)]
    q = mp.get_context('fork').Queue()
    t0 = time.time() + 1.0
    ps = [mp.get_context('fork').Process(target=worker, args=(a, g, order, t0, q)) for g in groups]
    for p in ps:
        p.start()
    snap = {}

    def sampler():   # batcher counters at the edges of the measured window
        for key, t in (('s0', t0 + a.warmup), ('s1', t0 + a.warmup + a.duration)):
            time.sleep(max(0.0, t - time.time()))
            snap[key] = server_stats(a.port)
    sth = threading.Thread(target=sampler, daemon=True)
    sth.start()
    recs, agg = [], {'errors': 0, 'err_samples': [], 'verified': 0, 'match': 0,
                     'mismatch_idx': [], 'sent': 0, 'late': 0}
    for _ in ps:
        r, s = q.get(timeout=a.duration + a.warmup + a.timeout + 120)
        recs += r
        for k in ('errors', 'verified', 'match', 'sent', 'late'):
            agg[k] += s[k]
        agg['err_samples'] += s['err_samples']
        agg['mismatch_idx'] += s['mismatch_idx']
    for p in ps:
        p.join(10)
    sth.join(10)
    lat = sorted(x[0] * 1e3 for x in recs)
    ndocs = sum(x[1] for x in recs)
    ntoks = sum(x[2] for x in recs)
    out = {
        'name': a.name, 'concurrency': a.conc, 'docs_per_request': a.batch_docs,
        'route': a.route, 'max_tok': a.max_tok, 'procs': nproc, 'keepalive': a.keepalive,
        'warmup_s': a.warmup, 'duration_s': a.duration,
        'requests': len(recs), 'docs': ndocs, 'tokens': ntoks,
        'req_s': len(recs) / a.duration, 'docs_s': ndocs / a.duration, 'tok_s': ntoks / a.duration,
        'mean_tok_per_doc': ntoks / ndocs if ndocs else None,
        'lat_ms': {'mean': sum(lat) / len(lat) if lat else None, 'p50': pct(lat, 50),
                   'p90': pct(lat, 90), 'p99': pct(lat, 99), 'max': lat[-1] if lat else None},
        'errors': agg['errors'], 'error_samples': agg['err_samples'][:5],
        'requests_completed_total': agg['sent'],
        'server': stats_delta(snap.get('s0'), snap.get('s1')),
        'verify': {'reference': a.ref, 'docs_verified': agg['verified'], 'docs_identical': agg['match'],
                   'agree_pct': 100.0 * agg['match'] / agg['verified'] if agg['verified'] else None,
                   'mismatch_idx_sample': sorted(set(agg['mismatch_idx']))[:50]},
    }
    js = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(js + '\n')
    print('%-12s C=%-4d ka=%d req/s %8.1f docs/s %9.1f tok/s %11.1f p50 %8.2fms p90 %8.2fms p99 %8.2fms '
          'err %d agree %s' % (a.name, a.conc, a.keepalive, out['req_s'], out['docs_s'], out['tok_s'],
                               out['lat_ms']['p50'] or -1, out['lat_ms']['p90'] or -1,
                               out['lat_ms']['p99'] or -1, out['errors'],
                               '%.2f%%' % out['verify']['agree_pct'] if out['verify']['agree_pct'] is not None else 'n/a'))
    return 0 if out['requests'] > 0 else 1


def cmd_sample(a):
    corpus = load_lines(a.corpus)
    step = max(1, len(corpus) // a.n)
    idx = [i * step for i in range(a.n) if i * step < len(corpus)]
    with open(a.out_txt, 'wb') as f:
        f.write(b''.join(corpus[i] + b'\n' for i in idx))
    with open(a.out_idx, 'w') as f:
        f.write(''.join('%d\n' % i for i in idx))
    print('sample: %d docs (every %d-th line) -> %s' % (len(idx), step, a.out_txt))


def cmd_exact(a):
    docs = load_lines(a.docs)
    ref = load_lines(a.ref)
    if len(ref) != len(docs):
        sys.exit('ref has %d lines, docs %d' % (len(ref), len(docs)))
    req = build_request(a.port, a.route, {'texts': [doc_text(d) for d in docs]})
    ts = time.time()
    st, body = http_roundtrip(a.port, req, a.timeout)
    ms = (time.time() - ts) * 1e3
    nv, nm, bad = compare(body, list(range(len(docs))), ref, True) if st == 200 else (len(docs), 0, [])
    out = {'name': a.name, 'status': st, 'docs': len(docs), 'docs_identical': nm,
           'identical_pct': 100.0 * nm / len(docs), 'byte_identical': st == 200 and nm == len(docs),
           'mismatch_idx': bad[:50], 'latency_ms': ms, 'reference': a.ref}
    with open(a.out, 'w') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    print('%s: HTTP %d, %d/%d docs byte-identical to the batch path (%.1f ms)' % (a.name, st, nm, len(docs), ms))
    return 0 if out['byte_identical'] else 1


def cmd_report(a):
    res = {'runs': {}, 'exact': {}, 'meta': {}}
    for p in a.inputs:
        with open(p) as f:
            d = json.load(f)
        if 'concurrency' in d:
            res['runs'][d['name']] = d
        else:
            res['exact'][d['name']] = d
    if a.meta:
        with open(a.meta) as f:
            res['meta'] = json.load(f)
    summary = {}
    for name, r in res['runs'].items():
        summary[name] = {'concurrency': r['concurrency'], 'docs_per_request': r['docs_per_request'],
                         'keepalive': r.get('keepalive', False),
                         'docs_s': round(r['docs_s'], 1), 'tok_s': round(r['tok_s'], 1),
                         'req_s': round(r['req_s'], 1),
                         'p50_ms': r['lat_ms']['p50'], 'p90_ms': r['lat_ms']['p90'], 'p99_ms': r['lat_ms']['p99'],
                         'errors': r['errors'], 'agree_pct_vs_batch_path': r['verify']['agree_pct'],
                         'docs_verified': r['verify']['docs_verified'],
                         'server_tok_per_forward': (r.get('server') or {}).get('mean_tok_per_forward'),
                         'server_jobs_per_forward': (r.get('server') or {}).get('mean_jobs_per_forward'),
                         'server_ms_per_forward': (r.get('server') or {}).get('mean_run_docs_ms'),
                         'server_busy_pct': (r.get('server') or {}).get('run_docs_busy_pct')}
    res['summary'] = summary
    tv = sum(r['verify']['docs_verified'] for r in res['runs'].values())
    tm = sum(r['verify']['docs_identical'] for r in res['runs'].values())
    res['agreement'] = {
        'all_runs_docs_verified': tv, 'all_runs_docs_identical': tm,
        'all_runs_agree_pct': 100.0 * tm / tv if tv else None,
        'exact_same_batch': {k: {'docs': v['docs'], 'identical': v['docs_identical'],
                                 'byte_identical': v['byte_identical']} for k, v in res['exact'].items()},
    }
    with open(a.out, 'w') as f:
        f.write(json.dumps(res, indent=1) + '\n')
    print('%-12s %5s %4s %3s %10s %12s %9s %9s %9s %6s %8s %8s %6s' % (
        'run', 'C', 'B', 'ka', 'docs/s', 'tok/s', 'p50ms', 'p90ms', 'p99ms', 'err', 'agree%', 'tok/fwd', 'busy%'))
    for name, s in summary.items():
        f = lambda v: '%9.2f' % v if v is not None else '      n/a'
        print('%-12s %5d %4d %3s %10.1f %12.1f %s %s %s %6d %8s %8s %6s' % (
            name, s['concurrency'], s['docs_per_request'], 'y' if s['keepalive'] else 'n', s['docs_s'], s['tok_s'], f(s['p50_ms']),
            f(s['p90_ms']), f(s['p99_ms']), s['errors'],
            '%.2f' % s['agree_pct_vs_batch_path'] if s['agree_pct_vs_batch_path'] is not None else 'n/a',
            '%.0f' % s['server_tok_per_forward'] if s['server_tok_per_forward'] is not None else 'n/a',
            '%.1f' % s['server_busy_pct'] if s['server_busy_pct'] is not None else 'n/a'))
    for k, v in res['exact'].items():
        print('exact %s: %d/%d byte-identical' % (k, v['docs_identical'], v['docs']))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--port', type=int, required=True)
    r.add_argument('--corpus', required=True)
    r.add_argument('--ntok', required=True, help='tokens per corpus line (pf --lines --dump-ntok)')
    r.add_argument('--ref', help='reference masked output, one line per corpus line')
    r.add_argument('--name', default='run')
    r.add_argument('--route', default='/redact')
    r.add_argument('--conc', type=int, default=1)
    r.add_argument('--batch-docs', type=int, default=1, help='docs per request (>1: {"texts": [...]})')
    r.add_argument('--texts', action='store_true', help='use {"texts": [...]} even for 1 doc')
    r.add_argument('--keepalive', action='store_true', help='persistent connection per client')
    r.add_argument('--max-tok', type=int, default=0, help='only docs with at most this many tokens')
    r.add_argument('--warmup', type=float, default=3)
    r.add_argument('--duration', type=float, default=20)
    r.add_argument('--procs', type=int, default=4)
    r.add_argument('--timeout', type=float, default=60)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--out')
    s = sp.add_parser('sample')
    s.add_argument('--corpus', required=True)
    s.add_argument('--n', type=int, default=500)
    s.add_argument('--out-txt', required=True)
    s.add_argument('--out-idx', required=True)
    e = sp.add_parser('exact')
    e.add_argument('--port', type=int, required=True)
    e.add_argument('--docs', required=True)
    e.add_argument('--ref', required=True)
    e.add_argument('--name', default='exact')
    e.add_argument('--route', default='/redact')
    e.add_argument('--timeout', type=float, default=120)
    e.add_argument('--out', required=True)
    p = sp.add_parser('report')
    p.add_argument('inputs', nargs='+')
    p.add_argument('--meta')
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    return {'run': cmd_run, 'sample': cmd_sample, 'exact': cmd_exact, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
