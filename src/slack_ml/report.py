"""週次レポート（1枚の HTML）を作る。"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from slack_ml.scoring import FEATURE_TEXT, format_value, reason_text

TEMPLATE_STYLE = """
:root {
  color-scheme: light;
  --paper: #ffffff;
  --ink: #000000;
  --graphite: #666666;
  --smoke: #8f8f8f;
  --ash: #f1f1f1;
  --hairline: rgba(0, 0, 0, 0.12);
  --grid: rgba(0, 0, 0, 0.07);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --paper: #0b0b0b;
    --ink: #f5f5f5;
    --graphite: #a3a3a3;
    --smoke: #7c7c7c;
    --ash: #1c1c1c;
    --hairline: rgba(255, 255, 255, 0.14);
    --grid: rgba(255, 255, 255, 0.08);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #0b0b0b;
  --ink: #f5f5f5;
  --graphite: #a3a3a3;
  --smoke: #7c7c7c;
  --ash: #1c1c1c;
  --hairline: rgba(255, 255, 255, 0.14);
  --grid: rgba(255, 255, 255, 0.08);
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif;
  font-feature-settings: "calt" on, "liga" on;
  font-size: 17px;
  line-height: 1.65;
  letter-spacing: -0.01em;
}
main { max-width: 1200px; margin: 0 auto; padding: 64px 24px 96px; }
section { margin-top: 64px; }
p { margin: 0; }

.eyebrow { font-size: 14px; font-weight: 500; color: var(--graphite); letter-spacing: 0; }
h1 {
  font-size: 48px; line-height: 1.16; font-weight: 500; letter-spacing: -0.03em;
  margin: 12px 0 20px; max-width: 18em;
}
h2 { font-size: 28px; line-height: 1.21; font-weight: 600; letter-spacing: 0.011em; margin: 0 0 8px; }
.lede { font-size: 18px; line-height: 1.5; color: var(--graphite); max-width: 46em; margin-bottom: 20px; }
.caption { font-size: 13px; line-height: 1.5; color: var(--graphite); letter-spacing: 0; }

.pills { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
  display: inline-block; border: 1px solid var(--hairline); border-radius: 9999px;
  padding: 4px 14px; font-size: 13px; font-weight: 500; color: var(--ink); letter-spacing: 0; white-space: nowrap;
}

.split { display: grid; gap: 32px; grid-template-columns: minmax(0, 1.85fr) minmax(0, 1fr); align-items: start; }
.split .statement { font-size: 22px; line-height: 1.45; letter-spacing: -0.01em; font-weight: 400; }
.notes { display: grid; gap: 20px; }
.note h3 { font-size: 14px; font-weight: 500; color: var(--graphite); margin: 0 0 2px; letter-spacing: 0; }
.note p { font-size: 15px; line-height: 1.6; }

.card { border: 1px solid var(--hairline); border-radius: 6px; padding: 16px; }
.grid-2 { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }

.tile .label { font-size: 14px; font-weight: 500; color: var(--graphite); letter-spacing: 0; }
.tile .value { font-size: 48px; line-height: 1.16; font-weight: 500; letter-spacing: -0.03em; margin: 8px 0 4px; }
.tile .delta { font-size: 13px; color: var(--graphite); letter-spacing: 0; }

.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 15px; letter-spacing: 0; }
th, td { text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--hairline); vertical-align: top; }
thead th { font-size: 13px; font-weight: 500; color: var(--graphite); padding-top: 4px; }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.rank { font-size: 22px; font-weight: 500; letter-spacing: -0.01em; width: 3em; }
.who { font-weight: 500; }
.id { display: block; font-size: 12px; color: var(--smoke); font-variant-numeric: tabular-nums; }
.reasons { margin: 0; padding: 0; list-style: none; display: grid; gap: 4px; }
.reasons li { font-size: 14px; line-height: 1.55; padding-left: 14px; position: relative; }
.reasons li::before { content: ""; position: absolute; left: 0; top: 0.72em; width: 6px; height: 1px; background: var(--ink); }
.table-wrap.members table { min-width: 760px; }
.table-wrap.members td:last-child { min-width: 380px; }
.table-wrap.members.connect table { min-width: 980px; }
.table-wrap.members.connect td:nth-child(5) { min-width: 360px; }
.table-wrap.members.connect td:last-child { min-width: 200px; }
.partners { margin: 0; padding: 0; list-style: none; display: grid; gap: 6px; font-size: 14px; }
.table-wrap.wide table { min-width: 620px; }
tr.selected td { font-weight: 500; }
.tag { display: inline-block; margin-left: 8px; border: 1px solid var(--hairline); border-radius: 9999px;
  padding: 0 10px; font-size: 12px; font-weight: 500; vertical-align: 1px; }

.chart-title { font-size: 15px; font-weight: 500; margin-bottom: 2px; }
.chart { position: relative; margin-top: 8px; }
.chart svg { display: block; width: 100%; height: 220px; }
.chart svg:focus { outline: 1px solid var(--ink); outline-offset: 4px; border-radius: 6px; }
.tooltip {
  position: absolute; pointer-events: none; display: none; min-width: 120px;
  background: var(--paper); border: 1px solid var(--hairline); border-radius: 6px;
  padding: 8px 12px; font-size: 13px; letter-spacing: 0;
}
.tooltip .date { color: var(--graphite); }
.tooltip strong { font-size: 18px; font-weight: 500; font-variant-numeric: tabular-nums; }

details { margin-top: 16px; }
summary {
  display: inline-block; cursor: pointer; list-style: none; border: 1px solid var(--hairline); border-radius: 9999px;
  padding: 6px 18px; font-size: 14px; font-weight: 500; letter-spacing: 0;
}
summary::-webkit-details-marker { display: none; }
summary:hover { background: var(--ash); }
details[open] summary { margin-bottom: 12px; }

.cautions { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--hairline); }
.cautions li { padding: 14px 0; border-bottom: 1px solid var(--hairline); font-size: 16px; }
footer { margin-top: 64px; }
.sub-heading { font-size: 18px; font-weight: 500; margin: 32px 0 4px; letter-spacing: -0.01em; }
.sub-heading + .caption { margin-bottom: 12px; }
.state { display: inline-block; border: 1px solid var(--hairline); border-radius: 9999px;
  padding: 0 10px; font-size: 12px; font-weight: 500; white-space: nowrap; }
.state.alert { background: var(--ink); border-color: var(--ink); color: var(--paper); }
.state.expected, .state.stable { color: var(--graphite); }

@media (max-width: 760px) {
  main { padding: 40px 16px 64px; }
  section { margin-top: 48px; }
  h1 { font-size: 34px; }
  h2 { font-size: 24px; }
  .split, .grid-2 { grid-template-columns: minmax(0, 1fr); }
  .split .statement { font-size: 19px; }
  .tile .value { font-size: 40px; }
}
"""

TEMPLATE_SCRIPT = """
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const NS = 'http://www.w3.org/2000/svg';

function el(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceStep(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function lineChart(root, spec) {
  const svg = root.querySelector('svg');
  const tip = root.querySelector('.tooltip');
  const rows = spec.rows;
  const draw = () => {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const W = svg.clientWidth || 500, H = 220;
    const m = { l: 40, r: 12, t: 10, b: 26 };
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const values = rows.map((r) => r[spec.key]).filter((v) => v !== null);
    const top = spec.top || Math.max(...values);
    const step = niceStep(top / 4);
    const ticks = Math.max(1, Math.ceil(top / step));
    const yMax = step * ticks;
    const x = (i) => m.l + (rows.length <= 1 ? 0 : (i * (W - m.l - m.r)) / (rows.length - 1));
    const y = (v) => m.t + (1 - v / yMax) * (H - m.t - m.b);
    for (let k = 0; k <= ticks; k++) {
      const v = step * k;
      el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: k === 0 ? css('--hairline') : css('--grid'), 'stroke-width': 1 }, svg);
      const t = el('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: css('--smoke') }, svg);
      t.textContent = spec.format(v);
    }
    let lastYear = '';
    rows.forEach((r, i) => {
      const d = r.week_start;
      if (d.slice(5, 7) % 3 === 1 && d.slice(8, 10) <= '07') {
        const t = el('text', { x: x(i), y: H - 6, 'text-anchor': 'middle', 'font-size': 11, fill: css('--smoke') }, svg);
        t.textContent = d.slice(0, 4) === lastYear ? Number(d.slice(5, 7)) + '月' : d.slice(0, 4) + '年' + Number(d.slice(5, 7)) + '月';
        lastYear = d.slice(0, 4);
      }
    });
    let path = '';
    rows.forEach((r, i) => {
      if (r[spec.key] === null) return;
      path += (path === '' ? 'M' : 'L') + x(i).toFixed(1) + ',' + y(r[spec.key]).toFixed(1);
    });
    el('path', { d: path, fill: 'none', stroke: css('--ink'), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
    const lastIndex = rows.map((r) => r[spec.key]).lastIndexOf(values[values.length - 1]);
    el('circle', { cx: x(lastIndex), cy: y(values[values.length - 1]), r: 4, fill: css('--ink'), stroke: css('--paper'), 'stroke-width': 2 }, svg);

    const cross = el('line', { y1: m.t, y2: H - m.b, stroke: css('--hairline'), 'stroke-width': 1, visibility: 'hidden' }, svg);
    const dot = el('circle', { r: 4, fill: css('--ink'), stroke: css('--paper'), 'stroke-width': 2, visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    const show = (i) => {
      const r = rows[i];
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
      if (r[spec.key] === null) dot.setAttribute('visibility', 'hidden');
      else { dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(r[spec.key])); dot.setAttribute('visibility', 'visible'); }
      tip.replaceChildren();
      const val = document.createElement('strong');
      val.textContent = r[spec.key] === null ? '—' : spec.format(r[spec.key], true);
      const date = document.createElement('div'); date.className = 'date'; date.textContent = r.week_start + ' の週';
      tip.append(val, date);
      tip.style.display = 'block';
      const left = Math.min(Math.max(x(i) + 12, 0), W - tip.offsetWidth - 4);
      tip.style.left = left + 'px'; tip.style.top = '4px';
    };
    const hide = () => { cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; };
    let current = rows.length - 1;
    hit.addEventListener('pointermove', (e) => {
      const box = svg.getBoundingClientRect();
      const px = ((e.clientX - box.left) / box.width) * W;
      current = Math.max(0, Math.min(rows.length - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (rows.length - 1))));
      show(current);
    });
    hit.addEventListener('pointerleave', hide);
    svg.onfocus = () => show(current);
    svg.onblur = hide;
    svg.onkeydown = (e) => {
      if (e.key === 'ArrowLeft') current = Math.max(0, current - 1);
      else if (e.key === 'ArrowRight') current = Math.min(rows.length - 1, current + 1);
      else return;
      e.preventDefault(); show(current);
    };
  };
  draw();
  window.addEventListener('resize', draw);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
}

const people = (v, tip) => Math.round(v).toLocaleString('ja-JP') + (tip ? '人' : '');
const percent = (v) => Math.round(v * 100) + '%';
const membersTop = Math.max(...DATA.weeks.flatMap((r) => [r.active, r.interacting]).filter((v) => v !== null));
lineChart(document.getElementById('chart-active'), { rows: DATA.weeks, key: 'active', format: people, top: membersTop });
lineChart(document.getElementById('chart-interacting'), { rows: DATA.weeks, key: 'interacting', format: people, top: membersTop });
lineChart(document.getElementById('chart-reply'), { rows: DATA.weeks, key: 'reply_rate', format: percent });
"""


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_risk(risk: float, model: str) -> str:
    return f"{risk:.0%}" if model in ("logistic", "gbdt") else f"{risk:,.0f}"


def _member_rows(frame: pd.DataFrame, model: str, partners: dict | None = None) -> str:
    columns = 5 if partners is None else 6
    if frame.empty:
        return f'<tr><td colspan="{columns}" class="caption">該当するメンバーはいません</td></tr>'
    out = []
    for i, row in enumerate(frame.itertuples(index=False), start=1):
        items = "".join(f"<li>{_esc(reason_text(r))}</li>" for r in json.loads(row.reasons)) or "<li>—</li>"
        cells = (
            f'<tr><td class="num rank">{i}</td>'
            f'<td><span class="who">{_esc(row.display)}</span><span class="id">{_esc(row.user_id)}</span></td>'
            f'<td class="num">{_esc(_fmt_risk(row.risk, model))}</td>'
            f'<td class="num">{row.days_since_last_action:.0f}日前</td>'
            f'<td><ul class="reasons">{items}</ul></td>'
        )
        if partners is not None:
            people = partners.get(row.user_id, [])
            listed = "".join(
                f'<li><span class="who">{_esc(name)}</span>'
                f'<span class="id">{_esc(partner_id)}・話題の近さ {similarity:.2f}</span></li>'
                for partner_id, name, similarity in people
            )
            cells += f'<td><ul class="partners">{listed or "<li>—</li>"}</ul></td>'
        out.append(cells + "</tr>")
    return "\n".join(out)


def _tile(label: str, value: str, delta: str) -> str:
    return (
        f'<div class="card tile"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div><div class="delta">{_esc(delta)}</div></div>'
    )


def _signed(value: float, unit: str) -> str:
    if value is None or pd.isna(value):
        return "比べる値がありません"
    return f"4週前より {value:+,.0f}{unit}"


def _members_table(frame: pd.DataFrame, model: str, partners: dict | None = None) -> str:
    extra = "" if partners is None else "<th>つなぐ相手</th>"
    wide = "" if partners is None else " connect"
    return (
        f'<div class="card table-wrap members{wide}"><table>'
        '<thead><tr><th class="num">順位</th><th>メンバー</th><th class="num">リスク</th>'
        f'<th class="num">最後の行動</th><th>理由</th>{extra}</tr></thead>'
        f"<tbody>{_member_rows(frame, model, partners)}</tbody></table></div>"
    )


def _table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    [(n,)] = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
        [schema, table],
    ).fetchall()
    return n > 0


def _expertise_block(result) -> str:
    """詳しい人探しの評価（返信した人が上位に入った割合）の表。"""
    if result is None:
        return ""
    n = int(result["top_n"])
    return f"""
  <h3 class="sub-heading">詳しい人探し（つなぐ相手の選び方）の評価</h3>
  <p class="caption">「投稿に返信した人は、その話題に詳しい」とみなし、最後の8週の投稿を質問文にして、\
実際に返信した人が上位 {n} 人に1人でも入った割合を測っています\
（{int(result["queries"]):,} 件の投稿、{int(result["members"]):,} 人の中から検索）。</p>
  <div class="card table-wrap"><table>
    <thead><tr><th>選び方</th><th class="num">上位 {n} 人に返信した人が入った割合</th></tr></thead>
    <tbody>
      <tr class="selected"><td>話題の近さ（このレポートで使っている方法）</td><td class="num">{result["hit_model"]:.1%}</td></tr>
      <tr><td>発言の多い順</td><td class="num">{result["hit_activity"]:.1%}</td></tr>
      <tr><td>ランダム（期待値）</td><td class="num">{result["hit_random"]:.1%}</td></tr>
    </tbody></table></div>
  <p class="caption" style="margin-top: 12px">返信する人が話題よりも活発さで決まるコミュニティでは、\
発言の多い順の方が高く出ます。その場合、つなぐ相手は「話題の近さ」の参考として扱ってください。</p>"""


DRIFT_STATE = {
    "alert": "大きく変化",
    "warning": "注意",
    "stable": "安定",
    "expected": "想定どおり（時間とともに増える）",
    "unknown": "測れない",
}


def _drift_rows(frame: pd.DataFrame) -> str:
    return "\n".join(
        f"<tr><td>{_esc(FEATURE_TEXT.get(r.feature, (r.feature,))[0])}"
        f'<span class="id">{_esc(r.feature)}</span></td>'
        f'<td class="num">{"—" if pd.isna(r.psi) else f"{r.psi:.3f}"}</td>'
        f'<td><span class="state {_esc(r.status)}">{_esc(DRIFT_STATE.get(r.status, r.status))}</span></td>'
        f'<td class="num">{_esc(format_value(r.feature, r.train_median))}</td>'
        f'<td class="num">{_esc(format_value(r.feature, r.current_median))}</td></tr>'
        for r in frame.itertuples(index=False)
    )


def _drift_table(frame: pd.DataFrame) -> str:
    return (
        '<div class="card table-wrap wide"><table><thead><tr><th>特徴量</th><th class="num">PSI</th>'
        '<th>状態</th><th class="num">中央値（学習）</th><th class="num">中央値（今週）</th></tr></thead>'
        f"<tbody>{_drift_rows(frame)}</tbody></table></div>"
    )


def _monitoring_section(con: duckdb.DuckDBPyConnection, selected: str, snapshot) -> str:
    """モデルの見守りの節: 特徴量の分布の変化（PSI）と、確率のずれ（予測の平均と実際の割合）。"""
    drift = pd.DataFrame()
    if snapshot is not None and _table_exists(con, "ops", "feature_drift"):
        drift = con.execute(
            "SELECT * FROM ops.feature_drift WHERE snapshot_date = ? ORDER BY psi DESC NULLS LAST", [snapshot]
        ).df()
    calib = pd.DataFrame()
    if _table_exists(con, "ops", "calibration"):
        calib = con.execute(
            """
            SELECT * FROM ops.calibration
            WHERE population = 'early' AND model = ?
              AND evaluated_at = (
                  SELECT MAX(evaluated_at) FROM ops.calibration WHERE population = 'early' AND model = ?
              )
            ORDER BY grp
            """,
            [selected, selected],
        ).df()
    if drift.empty and calib.empty:
        return ""

    counts = drift["status"].value_counts() if len(drift) else pd.Series(dtype=int)
    drift_tile = _tile(
        "分布が大きく変わった特徴量",
        f"{counts.get('alert', 0)}個",
        f"注意 {counts.get('warning', 0)}個・安定 {counts.get('stable', 0)}個"
        f"（時間とともに増える {counts.get('expected', 0)}個は除く）",
    )
    if len(calib):
        first = calib.iloc[0]
        calib_tile = _tile(
            "確率のずれ（早期発見、ECE）",
            f"{first['ece'] * 100:.1f}ポイント",
            f"Brier {first['brier']:.3f}・全員に平均の離脱率を言ったとき {first['baseline_brier']:.3f}",
        )
    else:
        calib_tile = _tile("確率のずれ", "—", "ルールは確率を出さないので測りません")

    if drift.empty:
        drift_block = ""
    else:
        flagged = drift[drift["status"] != "stable"]
        stable = drift[drift["status"] == "stable"]
        main = (
            _drift_table(flagged)
            if len(flagged)
            else '<p class="caption">すべての特徴量が安定しています。</p>'
        )
        rest = (
            f"<details><summary>安定している {len(stable)} 個を見る</summary>{_drift_table(stable)}</details>"
            if len(stable) and len(flagged)
            else ""
        )
        advice = (
            '<p class="caption" style="margin-top: 12px">大きく変化した特徴量が多いときは、学習に使った過去と今とで、'
            "メンバーの様子が違います。上のモデルの評価が下がっていないかを確かめ、学習に使う期間を直近に絞ることも考えてください。</p>"
            if counts.get("alert", 0)
            else ""
        )
        drift_block = f"""
  <h3 class="sub-heading">特徴量の分布の変化（PSI）</h3>
  <p class="caption">学習に使ったデータ（ラベルのある全期間）と今週の対象者とで、特徴量の分布を比べています。\
学習データの十分位で区間を切り、区間ごとの割合の差を足し合わせた値（PSI）です。0.1 未満は安定、0.25 を超えると大きく変化。</p>
  {main}
  {rest}
  {advice}"""

    if calib.empty:
        calib_block = ""
    else:
        rows = "\n".join(
            f'<tr><td class="num">{int(r.grp)}</td><td class="num">{int(r.n):,}</td>'
            f'<td class="num">{r.mean_predicted:.1%}</td><td class="num">{r.observed_rate:.1%}</td>'
            f'<td class="num">{(r.mean_predicted - r.observed_rate) * 100:+.1f}</td></tr>'
            for r in calib.itertuples(index=False)
        )
        top = calib.iloc[-1]
        if top["mean_predicted"] - top["observed_rate"] > 0.05:
            reading = (
                "リスクの最も高い区間では、予測が実際より高めに出ています。"
                "リスクの数字は「順番」として読み、確率そのものは割り引いてください。"
            )
        elif top["observed_rate"] - top["mean_predicted"] > 0.05:
            reading = "リスクの最も高い区間では、実際の方が予測より高くなっています。リスクの数字は控えめに出ています。"
        else:
            reading = "リスクの最も高い区間でも、予測と実際の差は5ポイント以内です。"
        calib_block = f"""
  <h3 class="sub-heading">確率のずれ（{_esc(selected)}、早期発見）</h3>
  <p class="caption">ウォークフォワード評価の予測（学習に使っていない期間の予測）を、リスクの小さい順に10の区間に分け、\
区間ごとに予測の平均と、実際に次の4週間来なかった割合を比べています。</p>
  <div class="card table-wrap wide"><table>
    <thead><tr><th class="num">区間</th><th class="num">人数</th><th class="num">予測の平均</th>\
<th class="num">実際に来なかった割合</th><th class="num">差（ポイント）</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="caption" style="margin-top: 12px">{_esc(reading)}</p>"""

    return f"""<section>
  <h2>モデルの見守り</h2>
  <p class="lede">モデルは過去のデータで学習しています。学習したときからメンバーの様子が変わっていないか（分布の変化）と、\
リスクの確率がどれくらい当たっているか（確率のずれ）を、毎週確かめます。</p>
  <div class="grid-2">{drift_tile}{calib_tile}</div>
  {drift_block}
  {calib_block}
</section>"""


def build_report(
    con: duckdb.DuckDBPyConnection,
    path: Path | str,
    k: int,
    comparison: str | None = None,
) -> Path:
    """DWH の gold と ops から週次レポートを作り、path に書き出す。"""
    weeks = con.execute(
        """
        SELECT strftime(week_start, '%Y-%m-%d') AS week_start, active_members AS active,
               interacting_members AS interacting, reply_rate_7d AS reply_rate, new_members
        FROM gold.weekly_kpi
        WHERE NOT is_partial_week
        ORDER BY week_start
        """
    ).df()
    weeks = weeks.astype(object).where(weeks.notna(), None)
    evaluation = con.execute(
        """
        SELECT * FROM ops.model_evaluation
        WHERE evaluated_at = (SELECT MAX(evaluated_at) FROM ops.model_evaluation)
        ORDER BY population DESC, model
        """
    ).df()
    selected = evaluation.loc[evaluation["selected"], "model"].iloc[0]
    scores = con.execute(
        """
        SELECT
            s.*,
            f.days_since_last_action,
            COALESCE(NULLIF(u.display_name, ''), NULLIF(u.real_name, ''), u.name, s.user_id) AS display
        FROM ops.risk_scores AS s
        LEFT JOIN gold.feature_snapshots AS f USING (snapshot_date, user_id)
        LEFT JOIN silver.users AS u USING (user_id)
        WHERE s.model = ?
          AND s.snapshot_date = (SELECT MAX(snapshot_date) FROM ops.risk_scores WHERE model = ?)
        ORDER BY s.rank_all
        """,
        [selected, selected],
    ).df()
    [(last_date,)] = con.execute("SELECT MAX(posted_date) FROM silver.messages WHERE is_activity").fetchall()
    snapshot = pd.Timestamp(scores["snapshot_date"].iloc[0]).date() if len(scores) else None

    partners = None
    if snapshot is not None and _table_exists(con, "ops", "connect_suggestions"):
        rows = con.execute(
            """
            SELECT c.user_id, c.partner_id,
                   COALESCE(NULLIF(u.display_name, ''), NULLIF(u.real_name, ''), u.name, c.partner_id),
                   c.similarity
            FROM ops.connect_suggestions AS c
            LEFT JOIN silver.users AS u ON u.user_id = c.partner_id
            WHERE c.snapshot_date = ?
            ORDER BY c.user_id, c.partner_rank
            """,
            [snapshot],
        ).fetchall()
        partners = {}
        for user_id, partner_id, name, similarity in rows:
            partners.setdefault(user_id, []).append((partner_id, name, similarity))
    expert_eval = None
    if _table_exists(con, "ops", "expertise_evaluation"):
        found = con.execute("SELECT * FROM ops.expertise_evaluation ORDER BY evaluated_at DESC LIMIT 1").df()
        expert_eval = found.iloc[0] if len(found) else None

    early = scores[scores["early"]].sort_values("rank_early").head(k)
    away = scores[~scores["early"]].sort_values("days_since_last_action", ascending=False).head(k)

    latest = weeks.iloc[-1] if len(weeks) else None
    before = weeks.iloc[-5] if len(weeks) >= 5 else None
    reply = weeks.dropna(subset=["reply_rate"]) if len(weeks) else weeks
    reply_latest = reply.iloc[-1] if len(reply) else None
    reply_before = reply.iloc[-5] if len(reply) >= 5 else None
    tiles = "".join(
        [
            _tile(
                f"活動したメンバー（{latest['week_start']} の週）"
                if latest is not None
                else "活動したメンバー",
                f"{latest['active']:,}人" if latest is not None else "—",
                _signed(latest["active"] - before["active"], "人") if before is not None else "",
            ),
            _tile(
                "他のメンバーとやり取りしたメンバー（同じ週）",
                f"{latest['interacting']:,}人" if latest is not None else "—",
                _signed(latest["interacting"] - before["interacting"], "人") if before is not None else "",
            ),
            _tile(
                f"7日以内の返信率（{reply_latest['week_start']} の週）"
                if reply_latest is not None
                else "7日以内の返信率",
                f"{reply_latest['reply_rate']:.0%}" if reply_latest is not None else "—",
                f"4週前は {reply_before['reply_rate']:.0%}" if reply_before is not None else "",
            ),
            _tile(
                "今週の予測の対象者",
                f"{len(scores):,}人",
                f"うち直近7日に行動 {int(scores['early'].sum()):,}人",
            ),
        ]
    )

    eval_rows = "\n".join(
        f'<tr class="{"selected" if r.selected else ""}"><td>{"早期発見" if r.population == "early" else "全員"}</td>'
        f"<td>{_esc(r.model)}{'<span class=tag>採用</span>' if r.selected else ''}</td>"
        f'<td class="num">{r.positives:,}</td><td class="num">{r.pr_auc:.3f}</td>'
        f'<td class="num">{r.precision_at_k:.3f}</td><td class="num">{r.recall_at_k:.3f}</td>'
        f'<td class="num">{r.lift:.1f}倍</td></tr>'
        for r in evaluation.itertuples(index=False)
    )
    week_rows = "\n".join(
        f'<tr><td>{_esc(r.week_start)}</td><td class="num">{r.active}</td><td class="num">{r.interacting}</td>'
        f'<td class="num">{"—" if r.reply_rate is None else f"{r.reply_rate:.0%}"}</td>'
        f'<td class="num">{r.new_members}</td></tr>'
        for r in weeks.itertuples(index=False)
    )
    data = json.dumps({"weeks": weeks.to_dict(orient="records")}, ensure_ascii=False, default=float)
    data = data.replace("<", "\\u003c")
    headline = (
        f"今週、声をかけたいメンバーは {len(early)} 人です"
        if len(early)
        else "今週、声をかけたいメンバーはいません"
    )

    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>メンバー活性化 週次レポート</title>
<style>{TEMPLATE_STYLE}</style>
</head>
<body>
<main>
<header>
  <p class="eyebrow">メンバー活性化 週次レポート</p>
  <h1>{_esc(headline)}</h1>
  <div class="pills">
    <span class="pill">予測時点 {_esc(snapshot)}</span>
    <span class="pill">データは {_esc(last_date)} まで</span>
    <span class="pill">モデル {_esc(selected)}</span>
    <span class="pill">K = {k}</span>
  </div>
</header>

<section class="split">
  <p class="statement">まだ来ているうちに、これから4週間来なくなりそうなメンバーに気付き、声をかけるきっかけにするためのレポートです。</p>
  <div class="notes">
    <div class="note"><h3>背景</h3><p>メンバーは退会せずに、静かに来なくなります。気付いたときには、声をかけても間に合いません。</p></div>
    <div class="note"><h3>読む人</h3><p>コミュニティマネージャー（CM）。メンバーの名前が入るので、CM の外には共有しません。</p></div>
    <div class="note"><h3>使い方</h3><p>毎週、上位の人の理由とつなぐ相手を読み、声のかけ方を CM が決めます。本人に自動で通知はしません。</p></div>
  </div>
</section>

<section>
  <h2>今週の数字</h2>
  <p class="lede">直近の、7日分そろった週の値です。</p>
  <div class="grid-2">{tiles}</div>
</section>

<section>
  <h2>今週声をかけたいメンバー</h2>
  <p class="lede">直近7日に行動したメンバーの中で、次の4週間に来なくなる見込みが高い順に {k} 人です。理由は、その値を「継続している人の中央値」に置き換えたときに、予測が大きく下がる特徴量です。</p>
  {_members_table(early, selected, partners)}
</section>

<section>
  <h2>1週間以上来ていないメンバー</h2>
  <p class="lede">直近28日には行動したが、この1週間は来ていないメンバーです。最後の行動が古い順に {k} 人を並べています。</p>
  {_members_table(away, selected)}
</section>

<section>
  <h2>週次の推移</h2>
  <p class="lede">左右のグラフは縦軸をそろえています。線の上にカーソルを置くか、グラフを選んで左右の矢印キーを押すと、その週の値が出ます。</p>
  <div class="grid-2">
    <div class="card">
      <p class="chart-title">活動したメンバー</p>
      <p class="caption">投稿、返信、リアクションのどれかをした人数</p>
      <div class="chart" id="chart-active"><svg role="img" tabindex="0" aria-label="週ごとの、活動したメンバー数の推移"></svg><div class="tooltip"></div></div>
    </div>
    <div class="card">
      <p class="chart-title">他のメンバーとやり取りしたメンバー</p>
      <p class="caption">他のメンバーに返信かリアクションをした人数</p>
      <div class="chart" id="chart-interacting"><svg role="img" tabindex="0" aria-label="週ごとの、他のメンバーとやり取りしたメンバー数の推移"></svg><div class="tooltip"></div></div>
    </div>
  </div>
  <div class="card" style="margin-top: 16px">
    <p class="chart-title">7日以内に、他のメンバーから返信が付いた投稿の割合</p>
    <p class="caption">7日後までのデータが無い直近の週は表示しません</p>
    <div class="chart" id="chart-reply"><svg role="img" tabindex="0" aria-label="週ごとの、7日以内に返信が付いた投稿の割合の推移"></svg><div class="tooltip"></div></div>
  </div>
  <details><summary>表で見る</summary><div class="card table-wrap wide"><table>
    <thead><tr><th>週</th><th class="num">活動</th><th class="num">やり取り</th><th class="num">7日以内の返信率</th><th class="num">新しいメンバー</th></tr></thead>
    <tbody>{week_rows}</tbody></table></div></details>
  <p class="caption" style="margin-top: 12px">7日分そろっていない週（データの最初と最後）は除いています。</p>
</section>

<section>
  <h2>モデルの評価</h2>
  <p class="lede">過去で学習して次の期間を当てる評価（ウォークフォワード、学習と評価のあいだは28日空ける）の結果です。採用するモデルは、早期発見の PR-AUC で選びます。</p>
  <div class="card table-wrap wide"><table>
    <thead><tr><th>対象</th><th>モデル</th><th class="num">陽性</th><th class="num">PR-AUC</th><th class="num">上位 {k} 人の適合率</th><th class="num">上位 {k} 人の再現率</th><th class="num">リフト</th></tr></thead>
    <tbody>{eval_rows}</tbody></table></div>
  {f'<p class="caption" style="margin-top: 12px">{_esc(comparison)}</p>' if comparison else ""}
  {_expertise_block(expert_eval)}
</section>

{_monitoring_section(con, selected, snapshot)}

<section>
  <h2>使い方の注意</h2>
  <ul class="cautions">
    <li>予測は相関に基づきます。上位に入ったことは、その人の評価や選別には使いません。</li>
    <li>本人への自動通知はしません。声をかけるかどうかは CM が判断します。</li>
    <li>リアクションを付けた時刻は Slack から取れないので、投稿の時刻で近似しています。</li>
    <li>「来なくなる」は「次の4週間に投稿も返信もリアクションも無い」ことです。その後に戻ってくる人もいます。</li>
    <li>読むだけのメンバーはデータに現れません。</li>
  </ul>
</section>

<footer><p class="caption">作成 {_esc(datetime.now().strftime("%Y-%m-%d %H:%M"))}・このレポートにはメンバーの名前が含まれます。CM の外に共有しないでください。</p></footer>
</main>
<script type="application/json" id="report-data">{data}</script>
<script>{TEMPLATE_SCRIPT}</script>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
