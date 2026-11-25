"""Check the built page in a headless browser: numbers against the saved results, the network and parser against
Python, and every control against the view it should change.

  python notes/child-feature-enabling/explorer/check_site.py [--url URL] [--shots DIR] [--chromium PATH]

Without --url it serves explorer/site/ on a local port. With --url it checks a published copy instead (for example the
public Artifact link; the page then sits inside an iframe, which this script finds). Needs Playwright with a Chromium
build; --chromium points it at an existing binary. Exits non-zero on any failed check.
"""
import argparse
import functools
import http.server
import json
import statistics
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / 'studies/rhm/child_feature_enabling/results'
FAILS = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    if not ok:
        FAILS.append(name)


def serve():
    class Handler(http.server.SimpleHTTPRequestHandler):
        # The Artifact skeleton declares UTF-8; a bare local server has to say so itself.
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                          '.html': 'text/html; charset=utf-8', '.json': 'application/json; charset=utf-8'}

        def log_message(self, *a):
            pass
    handler = functools.partial(Handler, directory=str(HERE / 'site'))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{srv.server_port}/index.html'


def page_frame(page):
    """The frame that holds the study page (the page itself, or the artifact iframe)."""
    for _ in range(120):
        for fr in page.frames:
            try:
                if fr.evaluate('() => !!(window.CFE && window.CFE.STUDY)'):
                    return fr
            except Exception:
                pass
        page.wait_for_timeout(500)
    raise RuntimeError('study page did not load')


def saved_pairs(folder):
    out = {}
    for f in sorted((RESULTS / folder / 'pairs').glob('*.json')):
        p = json.loads(f.read_text())
        out[(p['grammar'], p['seed'])] = (p['arms']['intact']['crossing']['T'], p['arms']['detached']['crossing']['T'])
    return out


def run(fr, page, shots):
    js = fr.evaluate
    js('() => window.CFE.loadWeights(0)')
    fr.wait_for_function('() => window.CFE.WEIGHTS[0] !== undefined')

    # ---- planted integrator against the saved Radau event times (replaces check.js, which needed Node)
    worst = js('''() => { let w = 0; for (const a of window.CFE.STUDY.planted) {
        const e = integrate(a.tied, a.s, a.delta, { keep: false }).events;
        for (const [k, s] of [['left', a.left], ['right', a.right], ['hard', a.hard]]) w = Math.max(w, Math.abs(e[k] - s) / s); }
        return w; }''')
    check('planted integrator matches 60 saved event times', worst < 1e-6, f'max rel diff {worst:.2e}')

    # ---- parser and network against Python on fixed trees
    vecs = json.loads((HERE / 'check_vectors.json').read_text())
    for v in vecs:
        g = v['grammar']
        js(f'() => window.CFE.loadWeights({g})')
        fr.wait_for_function(f'() => window.CFE.WEIGHTS[{g}] !== undefined')
        res = js('''([g, leaves]) => leaves.map((x) => { const W = window.CFE.WEIGHTS[g];
            return { root: window.CFE.parse(g, x)[0][0], logits: window.CFE.forward(W.init, x).logits,
                     intact: window.CFE.forward(W.intact, x), detached: window.CFE.forward(W.detached, x) }; })''',
                 [g, v['leaves']])
        check(f'grammar {g}: parser recovers every root', [r['root'] for r in res] == v['root'])
        dl = max(abs(a - b) for r, ref in zip(res, v['init_logits']) for a, b in zip(r['logits'], ref))
        check(f'grammar {g}: step-0 logits match Python', dl < 2e-3, f'max |diff| {dl:.1e}')
        for arm in ('intact', 'detached'):
            d = max(max(abs(r[arm]['hard'] - h), abs(r[arm]['easy'] - e))
                    for r, h, e in zip(res, v[arm]['hard'], v[arm]['easy']))
            check(f'grammar {g}: {arm} head predictions match Python', d < 0.01, f'max |diff| {d:.1e}')
        for arm in ('intact', 'detached'):
            fit = js(f"() => window.CFE.sampleFit({g}, '{arm}')")
            saved = js(f"() => window.CFE.WEIGHTS[{g}]['{arm}'].saved.D_h")
            check(f'grammar {g}: {arm} sample hard fit near saved exact fit', abs(fit['D_h'] - saved) < 0.03,
                  f"{fit['D_h']:.4f} vs {saved:.4f}")

    # ---- displayed numbers against the saved results
    am = saved_pairs('intervention-extended')
    ratios = [d / i for i, d in am.values()]
    txt = lambda sel: fr.inner_text(sel)  # noqa: E731
    check('answer tile: median ratio', txt('#ans-ratio') == f'{statistics.median(ratios):.2f}×', txt('#ans-ratio'))
    check('answer tile: range', txt('#ans-range') == f'{min(ratios):.2f}–{max(ratios):.2f}', txt('#ans-range'))
    hyp = txt('#hyp-body')
    for s in ('1.0571 at every δ', '1.4408, 1.4352, 1.4329, 1.4321', '0.2503 and 0.5004', 'intact first in 9 of 9',
              'ratio 2.41 (2.09 to 2.51); 0 censored', '18 of 18 arms censored', '54 of 54 checkpoints',
              '5.8–8.2× at update 100', '2.12, 2.45, 2.33 against 2.12, 2.51, 2.35'):
        check(f'hypothesis table shows "{s}"', s in hyp)

    def table_rows():
        return fr.eval_on_selector_all('#pair-rows tr', 'rs => rs.map(r => [...r.children].map(c => c.innerText))')

    for key, folder in (('amended', 'intervention-extended'), ('written', 'intervention'),
                        ('rate01', 'intervention-extended-lr0.1')):
        fr.click(f'#proto button[data-v="{key}"]')
        saved = saved_pairs(folder)
        rows = table_rows()
        ok = len(rows) == len(saved)
        for r in rows:
            g, s = int(r[0].split()[1].rstrip(',')), int(r[0].split()[-1])
            ti, td = saved[(g, s)]
            want = [f'> {4000 if key == "written" else 40000:,}' if t is None else f'{t:,}' for t in (ti, td)]
            ok &= r[1] == want[0] and r[2] == want[1]
        check(f'{key}: pair table matches saved half-fit times', ok)
        title = txt('#pair-title')
        check(f'{key}: chart title follows the protocol', key_label(key) in title, title)
    fr.click('#proto button[data-v="written"]')
    check('as written: caption reports censoring', 'Censored' in txt('#pair-cap'))
    check('as written: aggregate caption', 'All 18 arms are censored' in txt('#agg-cap'), txt('#agg-cap'))
    fr.click('#proto button[data-v="amended"]')
    check('amended: aggregate caption', 'ratio 2.41 at the median (2.09 to 2.51). 0 of 18 arms censored' in txt('#agg-cap'), txt('#agg-cap'))

    # ---- pair and contrast controls
    fr.click('#pg button[data-v="1"]'); fr.click('#ps button[data-v="2"]')
    check('pair buttons select grammar 1, seed 2', 'grammar 1, init seed 2' in txt('#pair-title'), txt('#pair-title'))
    check('pair caption follows the pair', '4,150 updates intact and 10,050 detached' in txt('#pair-cap'), txt('#pair-cap'))
    fr.click('#pair-rows tr[data-g="2"][data-s="1"]')
    check('clicking a table row selects the pair', 'grammar 2, init seed 1' in txt('#pair-title'))
    fr.click('#ct button[data-v="e"]')
    check('contrast switch', txt('#pair-title').startswith('Easy'))
    fr.click('#ct button[data-v="h"]')
    fr.click('#proto button[data-v="rate01"]')
    disabled = fr.eval_on_selector_all('#ps button', 'bs => bs.map(b => b.disabled)')
    check('rate 0.1 offers only init seed 0', disabled == [False, True, True], str(disabled))
    fr.click('#proto button[data-v="amended"]')

    # ---- grammar explorer: synonym switch, resample, grammar change
    js("() => window.CFE.setGrammar(0, false)")
    fr.wait_for_timeout(300)
    st = js('''() => { const S = window.CFE.S; return { prods: S.prods.map(r => r.slice()), g: S.g, root: S.root }; }''')
    leaves0 = js('() => window.CFE.symbols(window.CFE.S.g, window.CFE.S.root, window.CFE.S.prods)[4]')
    js('() => { window.CFE.S.sel = { l: 2, p: 1 }; window.CFE.update("sel"); }')
    other = (st['prods'][2][1] + 1) % 4
    fr.click(f'#prod-buttons button[data-k="{other}"]')
    leaves1 = js('() => window.CFE.symbols(window.CFE.S.g, window.CFE.S.root, window.CFE.S.prods)[4]')
    diff = [i for i in range(16) if leaves0[i] != leaves1[i]]
    check('synonym switch changes only leaves inside the subtree (4-7)', diff and all(4 <= i <= 7 for i in diff), str(diff))
    checks = fr.eval_on_selector_all('#parse-card .checks li > span:first-child', 'ss => ss.map(s => s.className)')
    check('parser still recovers node, parent and root after the switch', checks == ['ok'] * 3, str(checks))
    hl = fr.eval_on_selector_all('#tree1 .node.leaf.changed', 'ns => ns.map(n => +n.dataset.p)')
    check('changed leaves are highlighted in the tree', sorted(hl) == diff, f'{hl} vs {diff}')
    out_txt = txt('#out-ttn')
    check('network panel reports the prediction before the switch', 'Before the last synonym switch' in out_txt)
    fr.click('#resample')
    leaves2 = js('() => window.CFE.symbols(window.CFE.S.g, window.CFE.S.root, window.CFE.S.prods)[4]')
    root2 = js('() => window.CFE.parse(window.CFE.S.g, window.CFE.symbols(window.CFE.S.g, window.CFE.S.root, window.CFE.S.prods)[4])[0][0]')
    check('resample keeps the chosen root', root2 == st['root'] and leaves2 != leaves1)
    prods_before = js('() => window.CFE.S.prods.map(r => r.slice())')
    fr.click('#g-next')
    fr.wait_for_function('() => window.CFE.WEIGHTS[1] !== undefined')
    fr.wait_for_timeout(300)
    after = js('() => ({ g: window.CFE.S.g, prods: window.CFE.S.prods })')
    leaves3 = js('() => window.CFE.symbols(window.CFE.S.g, window.CFE.S.root, window.CFE.S.prods)[4]')
    check('change grammar swaps the rule table and keeps the production choices',
          after['g'] == 1 and after['prods'] == prods_before and leaves3 != leaves2)
    check('network weights follow the grammar', 'grammar 1' in txt('#learner-id'), txt('#learner-id')[:80])
    check('parse card reports the grammar change', 'grammar 0 to grammar 1' in txt('#parse-card'))
    fr.select_option('#g-seed', '5')
    fr.wait_for_timeout(200)
    check('grammar without a network says so', 'No trained network' in txt('#learner-id'))
    fr.select_option('#g-seed', '0')
    fr.wait_for_function('() => window.CFE.S.g === 0 && window.CFE.WEIGHTS[0] !== undefined')
    fr.wait_for_timeout(300)

    # ---- network panel controls
    fr.click('#learner button[data-v="init"]')
    check('step-0 weights: accuracy near chance', 'chance is 0.0625' in txt('#out-ttn'))
    check('learner id names the checkpoint', 'ckpt_0.pt' in txt('#learner-id'))
    fr.click('#learner button[data-v="detached"]')
    check('detached weights identified', 'g0_s0_detached_final.pt' in txt('#learner-id'))
    fr.click('#learner button[data-v="intact"]')
    fr.click('#tree2 .node[data-l="4"][data-p="3"]')
    check('selecting a leaf in the network tree shows its embedding', 'Leaf 3: embedding' in txt('#ttn-card'))
    check('selection is shared with section 1', 'leaf 3' in txt('#node-card'))
    fr.click('#tree2 .node[data-l="2"][data-p="1"]')
    check('core inspector shows dimensions and tying', '16 out × 16 left × 16 right' in txt('#ttn-card')
          and 'Weight tying' in txt('#ttn-card'))
    fr.click('#stages button:nth-child(3)')
    dims = fr.eval_on_selector_all('#tree2 .node.dim', 'ns => ns.length')
    check('pass stepper dims nodes not yet reached', dims > 0, f'{dims} dimmed')
    fr.click('#stages button:nth-child(10)')

    # ---- theory controls
    fr.fill('#pl-rl', '1'); fr.dispatch_event('#pl-rl', 'input')
    fr.fill('#pl-rr', '0.5'); fr.dispatch_event('#pl-rr', 'input')
    loss = txt('#pl-loss')
    check('ceiling at r_L = 1, r_R = 0.5 is 0.25 by enumeration and closed form', loss.count('0.250000') >= 2, loss[-400:])
    rows = fr.eval_on_selector_all('#pl-loss tbody tr', 'rs => rs.map(r => [...r.children].slice(2).map(c => c.innerText))')
    check('enumeration equals closed form for loss, gradients and fits', all(a == b for a, b in rows), str(rows))
    fr.click('#pl-best')
    check('best w attains the ceiling', txt('#pl-loss').count('0.250000') >= 4)
    fr.select_option('#pl-x0', '3')
    check('leaf picker updates the joint-law card', 'bottom nodes' in txt('#pl-joint'))
    before = txt('#planted-out')
    fr.fill('#delta', '-2.3'); fr.dispatch_event('#delta', 'input')
    check('δ slider updates the event times', txt('#planted-out') != before and txt('#delta-val') == '0.00501')
    fr.click('#tie button[data-v="tied"]')
    check('tied law text', 'shared table' in txt('#planted-law'))

    if shots:
        Path(shots).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=f'{shots}/full.png', full_page=True)


def key_label(key):
    return {'amended': 'amended, rate 0.3', 'written': 'as written', 'rate01': 'amended, rate 0.1'}[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url')
    ap.add_argument('--shots')
    ap.add_argument('--chromium')
    a = ap.parse_args()
    url = a.url or serve()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chromium) if a.chromium else p.chromium.launch()
        ctx = browser.new_context(viewport=dict(width=1280, height=900))   # fresh context: no cookies, signed out
        page = ctx.new_page()
        errors = []
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(url, wait_until='load', timeout=120_000)
        fr = page_frame(page)
        run(fr, page, a.shots)
        check('no script errors in the page', not [e for e in errors if 'favicon' not in e], '; '.join(errors[:3]))
        browser.close()
    print(f'{len(FAILS)} failed' if FAILS else 'all checks passed')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
