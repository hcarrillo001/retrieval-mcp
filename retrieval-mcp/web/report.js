/* retriEVAL report renderer, shared by the sandbox (index.html), the report
 * page (report.html) and the MCP widget inside Claude (server.py inlines this
 * file). One renderer, so the three views cannot drift apart.
 *
 *   RetrievalReport.mount(el, data, opts)
 *
 * data: a sandbox result, a run_eval result, or a saved run record. All carry
 *   threshold, judge_model, aggregate {metric: {mean_score, pass_rate, n, verdict?}}
 *   and per_case [{index, input, actual_output, retrieval_context,
 *                  scores {metric: {score, success, reason, details}}}].
 * opts (all optional):
 *   metric       which metric to show first
 *   side(data, metric)  HTML for the right half of the score card
 *   footer       HTML appended at the bottom of the card
 *   onRender(el, data, metric)  called after every render (wire buttons here)
 *   openLink(url)               how to open a link (the widget asks its host)
 * Plain ES5, no dependencies.
 */
(function (root) {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function fm(n) { return n == null || isNaN(n) ? "n/a" : Number(n).toFixed(2); }
  // green at or above the threshold, yellow in the middle, red below 0.4
  function tone(v, thr) { if (v == null) return "mid"; return v >= thr ? "ok" : (v >= 0.4 ? "mid" : "bad"); }

  // Jev claim rows share p / bad / label; older rows only had p_supported / supported
  function normRow(r) {
    var p = r.p != null ? r.p : r.p_supported, bad = r.bad != null ? r.bad : !r.supported;
    var t = bad ? "bad" : ((r.verdict === "says_nothing" || p < 0.7) ? "mid" : "ok");
    return { claim: r.claim, quote: r.quote, p: p, t: t, why: r.why || "", label: r.label || (bad ? "unsupported" : "supported") };
  }

  function chips(d) {
    if (!d || typeof d !== "object") return "";
    var out = [], skip = { claims: 1, probabilities: 1, legend: 1, kind: 1, judge: 1 };
    Object.keys(d).forEach(function (k) {
      var v = d[k];
      if (skip[k]) return;
      if ((d.judge === "jev" || d.claims) && (k === "unsupported_claims" || k === "contradictions" || k === "level_score" || k === "top_level")) return;
      var name = esc(k.replace(/_/g, " "));
      if (Array.isArray(v)) {
        if (!v.length) return;
        out.push('<span class="rr-chip bad"><b>' + v.length + "</b> " + name + "</span>");
        v.slice(0, 6).forEach(function (it) { out.push('<span class="rr-chip bad">' + esc(String(it)).slice(0, 110) + "</span>"); });
      } else if (typeof v === "number") out.push('<span class="rr-chip">' + name + " <b>" + v + "</b></span>");
    });
    return out.length ? '<div class="rr-chips">' + out.join("") + "</div>" : "";
  }

  function claimTable(d) {
    var claims = (d || {}).claims;
    if (!Array.isArray(claims) || !claims.length) return "";
    // worst first: failing, then borderline / not in context, then supported
    var rank = { bad: 0, mid: 1, ok: 2 };
    var rows = claims.map(normRow).sort(function (a, b) { return (rank[a.t] - rank[b.t]) || (a.p - b.p); });
    var n = { ok: 0, mid: 0, bad: 0 };
    rows.forEach(function (r) { n[r.t]++; });
    var contra = d.kind === "contradiction", jev = d.judge === "jev";
    // Jev gives a probability per claim; an LLM judge gives a verdict and, for
    // each failing claim, a one-line reason
    var mark = function (r) { return r.t === "ok" ? "✓" : (r.t === "bad" ? "✕" : "!"); };
    var row = function (r) {
      return '<div class="rr-cl t-' + r.t + (jev ? "" : " llm") + '"><span class="p">' + (jev ? fm(r.p) : mark(r)) + "</span>" +
        '<span class="c">' + esc(r.claim) + (r.why && r.t !== "ok" ? '<span class="why">' + esc(r.why) + "</span>" : "") + "</span>" +
        '<span class="bar"><span style="width:' + Math.max(3, Math.round(r.p * 100)) + '%"></span></span>' +
        '<span class="st">' + esc(r.label) + "</span></div>";
    };
    var first = rows.slice(0, 13), rest = rows.slice(13);
    var what = jev ? "Jev’s probability that the context " + (contra ? "does not contradict" : "supports") + " each claim, lowest first"
      : "The judge’s verdict on each claim, with a reason for every one that fails";
    return '<div class="rr-claims"><div class="hd"><b>Claims</b><span class="what">' + what + "</span><span class=\"sp\"></span>" +
      (n.bad ? '<span class="key"><i class="k-bad"></i>' + (contra ? "contradicts" : "unsupported") + " " + n.bad + "</span>" : "") +
      (n.mid ? '<span class="key"><i class="k-mid"></i>' + (contra ? "not in context" : "weak") + " " + n.mid + "</span>" : "") +
      '<span class="key"><i class="k-ok"></i>supported ' + n.ok + "</span></div>" +
      first.map(row).join("") +
      (rest.length ? '<div class="rr-hidden">' + rest.map(row).join("") + "</div>" +
        '<div class="more"><button type="button" class="rr-btn" data-rr="more">Show all ' + rows.length + " claims</button></div>" : "") +
      "</div>";
  }

  // mark the passages of the answer a claim came from (Jev quotes), or the LLM
  // judge's own unsupported / contradicted strings when they appear verbatim
  function highlight(text, d) {
    text = String(text || "");
    var spans = [], low = text.toLowerCase();
    function add(q, t, title) {
      if (!q) return;
      var i = low.indexOf(String(q).toLowerCase());
      if (i >= 0) spans.push({ s: i, e: i + String(q).length, t: t, title: title });
    }
    d = d || {};
    (d.claims || []).forEach(function (r) { var x = normRow(r); if (x.t !== "ok") add(x.quote, x.t, x.label + " · p=" + fm(x.p)); });
    if (!d.claims) {
      (d.unsupported_claims || []).forEach(function (q) { add(q, "bad", "unsupported"); });
      (d.contradictions || []).forEach(function (q) { add(q, "bad", "contradicts the context"); });
    }
    spans.sort(function (a, b) { return a.s - b.s; });
    var out = "", at = 0;
    spans.forEach(function (sp) {
      if (sp.s < at) return;
      out += esc(text.slice(at, sp.s)) + '<mark class="rr-hl t-' + sp.t + '" title="' + esc(sp.title) + '">' + esc(text.slice(sp.s, sp.e)) + "</mark>";
      at = sp.e;
    });
    return { html: out + esc(text.slice(at)), n: spans.length };
  }

  function chunkTitle(c, i) {
    // first sentence, e.g. "Travel policy, section 1: booking and approval"
    var s = String(c || "").trim(), m = s.match(/^(.{6,80}?)\.(\s|$)/), t = m ? m[1] : s;
    return (t.length > 72 ? t.slice(0, 70) + "…" : t) || ("Chunk " + (i + 1));
  }

  function detail(c, sc, noWhy) {
    var ctx = Array.isArray(c.retrieval_context) ? c.retrieval_context : (c.retrieval_context ? [c.retrieval_context] : []);
    var hl = highlight(c.actual_output, sc.details);
    var wt = sc.success === false ? (sc.score >= 0.4 ? "mid" : "bad") : "ok";
    // one case: the run verdict above already says this, so don't repeat it
    return (noWhy ? "" : '<div class="rr-why t-' + wt + '">' + esc(sc.reason || "(no explanation)") + "</div>") +
      chips(sc.details) + claimTable(sc.details) +
      '<div class="rr-io"><div class="rr-box"><div class="l">Question</div><p>' + esc(c.input || "not provided") + "</p>" +
      '<div class="l" style="margin-top:14px">Retrieved context</div>' +
      (ctx.length ? ctx.map(function (x, i) {
        return "<details" + (ctx.length === 1 && String(x).length < 400 ? " open" : "") + "><summary>" + esc(chunkTitle(x, i)) +
          "</summary><p>" + esc(x) + "</p></details>";
      }).join("") : '<p class="none">not provided</p>') +
      '</div><div class="rr-box"><div class="l">Model output' + (hl.n ? ' <span class="hl-key">problem claims highlighted</span>' : "") +
      '</div><p class="out">' + (hl.html || "not provided") + "</p></div></div>";
  }

  // the run-level answer to "why did this pass or fail"
  function verdictHtml(v, t) {
    if (!v || !v.headline) return "";
    return '<div class="rr-verdict t-' + t + '"><div class="h">' + esc(v.headline) + "</div>" +
      (v.drivers && v.drivers.length ? "<ul>" + v.drivers.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("") + "</ul>" : "") +
      "</div>";
  }

  // fallback for saved runs from before verdicts were stored
  function fallbackVerdict(m, mean, thr, cases) {
    var ok = mean >= thr, fails = cases.filter(function (c) { var s = (c.scores || {})[m]; return s && s.success === false; });
    return { headline: (ok ? "Passes" : "Fails") + ": " + m + " averaged " + fm(mean) + ", " + (ok ? "at or above" : "below") +
      " the " + fm(thr) + " threshold" + (cases.length > 1 ? " (" + (cases.length - fails.length) + " of " + cases.length + " cases pass)." : "."),
      drivers: fails.slice(0, 3).map(function (c) { var s = c.scores[m]; return "“" + String(c.input || "").slice(0, 70) + "” scored " + fm(s.score) + ": " + String(s.reason || "").slice(0, 180); }) };
  }

  function html(data, opts) {
    opts = opts || {};
    var agg = data.aggregate || {}, metrics = Object.keys(agg), cases = data.per_case || [], thr = data.threshold != null ? data.threshold : 0.7;
    var m = opts.metric && agg[opts.metric] ? opts.metric : metrics[0];
    if (!m) return '<div class="rr"><div class="rr-card"><div class="rr-empty">' + esc(data.message || data.error || "No scores in this result.") + "</div></div></div>";
    var a = agg[m] || {}, mean = a.mean_score, t = tone(mean, thr);
    var anyFail = cases.some(function (c) { var s = (c.scores || {})[m]; return s && s.success === false; });
    var withM = cases.filter(function (c) { return (c.scores || {})[m]; });
    var d0 = withM.length === 1 ? ((withM[0].scores[m] || {}).details || {}) : {};
    var sub = d0.total_claims != null
      ? (d0.kind === "contradiction" ? (d0.total_claims - (d0.contradictions || []).length) + " of " + d0.total_claims + " claims consistent with the context"
        : (d0.supported_claims != null ? d0.supported_claims + " of " + d0.total_claims + " claims supported" : ""))
      : (withM.length > 1 ? Math.round((a.pass_rate || 0) * withM.length) + " of " + withM.length + " cases pass" : "");
    var v = a.verdict || fallbackVerdict(m, mean, thr, withM);
    var shown = data.total_cases != null && data.total_cases > withM.length
      ? '<div class="rr-note">Showing the ' + withM.length + " lowest-scoring of " + data.total_cases + " cases.</div>" : "";
    var tabs = metrics.length > 1 ? '<div class="rr-tabs" role="tablist">' + metrics.map(function (x) {
      var xt = tone((agg[x] || {}).mean_score, thr);
      return '<button type="button" role="tab" class="rr-tab' + (x === m ? " on" : "") + '" data-rr="tab" data-m="' + esc(x) + '" aria-selected="' + (x === m) + '">' +
        esc(x) + ' <span class="t-' + xt + '">' + fm((agg[x] || {}).mean_score) + "</span></button>";
    }).join("") + "</div>" : "";
    var body = withM.length === 1 ? '<div class="rr-one">' + detail(withM[0], withM[0].scores[m], !!(v && v.headline)) + "</div>"
      : withM.map(function (c, i) {
        var sc = c.scores[m], ct = tone(sc.score, thr);
        return '<div class="rr-case' + (sc.success === false ? " open" : "") + '"><button type="button" class="rr-ch" data-rr="case">' +
          '<span class="rr-pill t-' + ct + '">' + fm(sc.score) + " " + (sc.success === false ? "fail" : "pass") + "</span>" +
          '<span class="q">' + esc(c.input || "case " + (i + 1)) + '</span><span class="caret">▶</span></button>' +
          '<div class="rr-cbody">' + detail(c, sc) + "</div></div>";
      }).join("");
    var side = opts.side ? opts.side(data, m) : "";
    return '<div class="rr">' + (data.notice ? '<div class="rr-notice">' + esc(data.notice) + "</div>" : "") +
      '<div class="rr-card">' + tabs +
      '<div class="rr-top' + (side ? "" : " solo") + '"><div class="rr-score">' +
      '<div class="r1"><span class="mono">' + esc(m) + '</span><span class="rr-pill t-' + (anyFail ? (t === "ok" ? "mid" : t) : "ok") + '">' + (anyFail ? "Needs review" : "All pass") + "</span></div>" +
      '<div class="r2"><span class="big t-' + t + '">' + fm(mean) + '</span><span class="sub">' + esc(sub) + "</span></div>" +
      '<div class="rr-trk"><div class="fill t-' + t + '" style="width:' + Math.max(2, Math.round((mean || 0) * 100)) + '%"></div><div class="thr" style="left:' + Math.round(thr * 100) + '%"></div></div>' +
      '<div class="scale"><span>0.00</span><span>threshold ' + fm(thr) + "</span><span>1.00</span></div>" +
      '<div class="meta">' + esc(data.judge_model || "") + (typeof data.elapsed_ms === "number" ? " · scored in " + (data.elapsed_ms / 1000).toFixed(1) + " s" : "") +
      (data.run_id ? " · " + esc(data.run_id) : "") + "</div>" +
      "</div>" + side + "</div>" +
      verdictHtml(v, anyFail ? (t === "ok" ? "mid" : t) : "ok") + shown + body + (opts.footer || "") + "</div></div>";
  }

  // compact card for chat (the MCP widget): the headline score and verdict,
  // the other metrics as chips, then a checklist of what failed and why.
  // The full report (every claim, highlighted answer) is the link in the footer.
  function compactHtml(data, opts) {
    opts = opts || {};
    var agg = data.aggregate || {}, metrics = Object.keys(agg), thr = data.threshold != null ? data.threshold : 0.7;
    var cases = data.per_case || [];
    if (!metrics.length) return '<div class="rr"><div class="rr-card"><div class="rr-empty">' + esc(data.message || data.error || "No scores in this result.") + "</div></div></div>";
    // lead with the worst metric: that is the one to explain
    var m = metrics.slice().sort(function (x, y) { return (agg[x].mean_score || 0) - (agg[y].mean_score || 0); })[0];
    var a = agg[m] || {}, t = tone(a.mean_score, thr), ok = a.mean_score >= thr;
    var withM = cases.filter(function (c) { return (c.scores || {})[m]; }).sort(function (x, y) { return (x.index || 0) - (y.index || 0); });
    var v = a.verdict || fallbackVerdict(m, a.mean_score, thr, withM);
    var n = data.total_cases != null ? data.total_cases : cases.length;
    var chipsHtml = metrics.length > 1 ? '<div class="rr-cb-chips">' + metrics.map(function (x) {
      return '<span class="rr-cb-chip t-' + tone(agg[x].mean_score, thr) + '">' + esc(x) + " " + fm(agg[x].mean_score) + "</span>";
    }).join("") + "</div>" : "";
    var row = function (ic, rt, title, why, score) {
      return '<div class="rr-cb-row t-' + rt + '"><span class="ic">' + ic + '</span><div class="tx"><div class="q">' + esc(title) + "</div>" +
        (why ? '<div class="r">' + esc(why) + "</div>" : "") + '</div><span class="s">' + score + "</span></div>";
    };
    var rows = [], more = 0, MAX = 6;
    if (withM.length > 1) {
      // one line per case; failing cases carry their reason
      var sorted = withM.slice().sort(function (x, y) { return x.scores[m].score - y.scores[m].score; });
      sorted.slice(0, MAX).sort(function (x, y) { return (x.index || 0) - (y.index || 0); }).forEach(function (c) {
        var sc = c.scores[m], pass = sc.success !== false;
        rows.push(row(pass ? "\u2713" : "\u2715", pass ? "ok" : tone(sc.score, thr), c.input || ("case " + ((c.index || 0) + 1)), pass ? "" : sc.reason, fm(sc.score)));
      });
      more = Math.max(0, n - Math.min(MAX, sorted.length));
    } else if (withM.length === 1) {
      // one case: its problem claims are the checklist
      var sc = withM[0].scores[m], d = sc.details || {}, claims = (d.claims || []).map(normRow);
      if (claims.length) {
        var bad = claims.filter(function (c) { return c.t !== "ok"; }).sort(function (x, y) { return x.p - y.p; });
        // the cross already says "unsupported"; spell out only the yellow cases
        // Jev: its probability; an LLM judge: its one-line reason instead
        var jevRows = d.judge === "jev";
        bad.slice(0, MAX).forEach(function (c) {
          rows.push(row(c.t === "bad" ? "\u2715" : "!", c.t, c.claim, c.why || (c.t === "mid" ? c.label : ""), jevRows ? fm(c.p) : ""));
        });
        if (bad.length > MAX) rows.push('<div class="rr-cb-more">+' + (bad.length - MAX) + " more in the full report</div>");
        var good = claims.length - bad.length;
        if (good) rows.push(row("\u2713", "ok", good + " other claim" + (good === 1 ? "" : "s") + " supported by the context", "", ""));
      } else {
        rows.push(row(sc.success === false ? "\u2715" : "\u2713", sc.success === false ? tone(sc.score, thr) : "ok",
          withM[0].input || m, sc.reason, fm(sc.score)));
        var listed = d.unsupported_claims || d.contradictions || [];
        listed.slice(0, 3).forEach(function (x) { rows.push(row("\u2715", "bad", x, "", "")); });
      }
    }
    return '<div class="rr rr-compact"><div class="rr-card">' +
      '<div class="rr-cb-top t-' + t + '"><span class="big">' + fm(a.mean_score) + '</span><div class="hd">' +
      '<div class="lbl">' + esc(m) + " \u00b7 threshold " + fm(thr) + " \u00b7 " + n + " case" + (n === 1 ? "" : "s") + "</div>" +
      '<div class="h">' + esc(v.headline || "") + "</div></div></div>" + chipsHtml +
      (rows.length ? '<div class="rr-cb-list">' + rows.join("") +
        (more ? '<div class="rr-cb-more">+' + more + " more in the full report</div>" : "") + "</div>" : "") +
      (opts.footer || "") + "</div></div>";
  }

  function mount(el, data, opts) {
    opts = opts || {};
    var metric = opts.metric;
    function draw() {
      el.innerHTML = (opts.compact ? compactHtml : html)(data, Object.assign({}, opts, { metric: metric }));
      if (opts.onRender) opts.onRender(el, data, metric || Object.keys(data.aggregate || {})[0]);
    }
    if (!el.__rrWired) {
      el.__rrWired = true;
      el.addEventListener("click", function (e) {
        var b = e.target.closest ? e.target.closest("[data-rr]") : null;
        if (b && el.contains(b)) {
          var k = b.getAttribute("data-rr");
          if (k === "case") b.parentNode.classList.toggle("open");
          else if (k === "more") { var h = b.parentNode.previousElementSibling; if (h) h.classList.remove("rr-hidden"); b.parentNode.remove(); }
          else if (k === "tab") { metric = b.getAttribute("data-m"); draw(); }
          if (opts.onChange) opts.onChange(el);
          return;
        }
        var a = e.target.closest ? e.target.closest("a[href]") : null;
        if (a && opts.openLink && el.contains(a)) { e.preventDefault(); opts.openLink(a.href); }
      });
      el.addEventListener("toggle", function () { if (opts.onChange) opts.onChange(el); }, true);
    }
    draw();
    return el;
  }

  root.RetrievalReport = { mount: mount, html: html, compactHtml: compactHtml, esc: esc, fm: fm, tone: tone };
})(typeof window !== "undefined" ? window : this);
