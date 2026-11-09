"""대시보드 데모(HTML) 만들기: outputs/dashboard_week.json을 넣어 한 장짜리 페이지로.

사용: .venv/Scripts/python scripts/build_prototype.py
출력: reports/dashboard_prototype.html  (Artifact로 게시. 문서 뼈대 태그는 게시할 때 붙는다)
디자인은 팀 앱(https://bookids-nine.vercel.app, /dashboard)을 따른다:
크림 바탕 #fffbeb, Jua 글꼴, 흰 카드(둥근 24px, 옅은 그림자), 하늘색 선택·막대, 주황 버튼, 가운데 한 줄 배치.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "dashboard_week.json"
OUT = ROOT / "reports" / "dashboard_prototype.html"
REPORT_URL = "https://claude.ai/artifact/NR4idEaXtbXZaXaxgLZG8e"

TEMPLATE = r"""<title>햇살반 대시보드 데모</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Jua&display=swap" rel="stylesheet">
<style>
:root {
  /* 팀 앱(Bookids)과 같은 세계: 크림 바탕 + 흰 카드, 하늘색은 선택·자료, 주황은 행동. 가운데 한 줄로 쌓는 보호자·선생님 화면.
     팀 앱이 밝은 화면 하나만 쓰므로 여기도 밝은 화면 하나로 고정한다. */
  --bg: #fffbeb; --card: #ffffff; --soft: #fafafa; --ink: #27272a; --ink-2: #52525c; --ink-3: #71717b; --ink-4: #9f9fa9;
  --line: #f4f4f5; --line-2: #d4d4d8;
  --sky: #00a5ef; --sky-2: #00bcfe; --sky-soft: #dff2fe; --sky-ink: #0084cc;
  --orange: #ff8b1a; --orange-2: #fe6e00; --orange-soft: #ffd7a8;
  --green: #00bb7f; --green-2: #00d294; --green-soft: #ecfdf5;
  --amber: #fcbb00; --amber-soft: #fef3c6; --amber-ink: #b75000; --amber-bg: #fffbeb;
  --rose: #ff667f; --rose-soft: #fff1f2;
  --font: "Jua", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
  --shadow: 0 1px 3px 0 #0000001a, 0 1px 2px -1px #0000001a;
  --shadow-md: 0 4px 6px -1px #0000001a, 0 2px 4px -2px #0000001a;
  color-scheme: light;
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body { background: var(--bg); color: var(--ink); font-family: var(--font); font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
h1, h2, h3 { font-weight: 400; margin: 0; text-wrap: balance; }
p { margin: 0; }
button { font: inherit; color: inherit; }
.num { font-variant-numeric: tabular-nums; }
:focus-visible { outline: 3px solid var(--sky-2); outline-offset: 2px; }

.page { max-width: 1000px; margin: 0 auto; padding-block: 24px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: flex; flex-direction: column; gap: 24px; }

/* 머리 */
.head { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.home { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; background: var(--card); border-radius: 999px; box-shadow: var(--shadow-md);
  color: var(--ink-2); text-decoration: none; font-size: 15px; }
.head h1 { font-size: 30px; display: flex; align-items: center; gap: 10px; }
.head .report { margin-left: auto; color: var(--ink-3); font-size: 14px; text-decoration: underline; text-underline-offset: 3px; }
.notice { background: var(--amber-bg); border: 1px solid var(--amber-soft); color: var(--amber-ink); border-radius: 16px; padding: 10px 16px; font-size: 15px; }
.notice.inside { border: 0; background: var(--amber-soft); }

/* 아이 고르기 */
.pills { display: flex; flex-wrap: wrap; gap: 8px; }
.pill { border: 0; background: var(--card); box-shadow: var(--shadow-md); border-radius: 999px; padding: 9px 18px; font-size: 17px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 6px; }
.pill[aria-pressed="true"] { background: var(--sky); color: #fff; }
.pill .mk { font-size: 12px; }
.pill .up { color: var(--green); } .pill .down { color: var(--orange-2); }
.pill[aria-pressed="true"] .up, .pill[aria-pressed="true"] .down { color: #fff; }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--amber); display: inline-block; }
.switch { display: inline-flex; background: var(--card); box-shadow: var(--shadow-md); border-radius: 999px; padding: 4px; gap: 4px; }
.switch a { text-decoration: none; color: var(--ink-2); padding: 7px 18px; border-radius: 999px; font-size: 16px; }
.switch a.on { background: var(--orange); color: #fff; }
.choose { display: flex; flex-direction: column; gap: 12px; }
.choose .row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.choose .legend { font-size: 13px; color: var(--ink-3); display: flex; gap: 14px; flex-wrap: wrap; }

/* 카드 */
.card { background: var(--card); border-radius: 24px; box-shadow: var(--shadow); padding: 22px 22px 20px; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.card h2 { font-size: 22px; }
.card .cap { font-size: 13.5px; color: var(--ink-3); }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
.sub { font-size: 15px; color: var(--ink-2); }

/* 숫자 상자 */
.boxes { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.box { background: var(--soft); border-radius: 16px; padding: 14px 14px 12px; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.box .k { font-size: 13.5px; color: var(--ink-2); }
.box .v { font-size: 32px; line-height: 1.15; display: flex; align-items: center; gap: 8px; }
.box .v small { font-size: 17px; }
.box .s { font-size: 12.5px; color: var(--ink-4); }

/* 6주 지표 */
.metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
.metric { background: var(--soft); border-radius: 16px; padding: 12px 12px 8px; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.metric .k { font-size: 13.5px; color: var(--ink-2); display: flex; align-items: center; gap: 4px; }
.metric .v { font-size: 26px; line-height: 1.2; }
.metric .v small { font-size: 14px; color: var(--ink-3); }
.metric .d { font-size: 12.5px; color: var(--ink-4); }
.tag { display: inline-block; width: fit-content; font-size: 13px; padding: 2px 10px; border-radius: 999px; }
.tag.up { background: var(--green-soft); color: var(--green); }
.tag.down { background: var(--rose-soft); color: var(--orange-2); }
.metric svg { width: 100%; height: auto; display: block; margin-top: 4px; }
.qmark { display: inline-grid; place-items: center; width: 17px; height: 17px; border-radius: 50%; background: var(--line); color: var(--ink-3); font-size: 11px; cursor: help; }
svg text { font-family: var(--font); fill: var(--ink-3); font-size: 11px; }

/* 말풍선과 질문 */
.talk { display: flex; flex-direction: column; gap: 10px; }
.say { display: flex; gap: 10px; align-items: flex-end; }
.say .when { font-size: 12.5px; color: var(--ink-4); width: 60px; flex: none; line-height: 1.3; }
.bubble { background: var(--line); padding: 10px 16px; border-radius: 20px 20px 20px 6px; font-size: 17px; }
.bubble.now { background: var(--sky-soft); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { background: var(--line); border-radius: 999px; padding: 5px 13px; font-size: 16px; }
.chip .c { color: var(--ink-4); font-size: 14px; margin-left: 3px; }
.chip.char { background: var(--card); box-shadow: inset 0 0 0 1.5px var(--line-2); color: var(--ink-2); }
.chip.hot { background: var(--sky-soft); color: var(--sky-ink); }
.news { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.news li { display: grid; grid-template-columns: 12px minmax(0, 1fr); gap: 10px; }
.news .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--amber); margin-top: 6px; }
.news q { display: block; color: var(--ink-3); font-size: 14px; }
.qs { display: grid; gap: 10px; }
.q { background: var(--amber-bg); border: 2px dashed var(--orange-soft); border-radius: 18px; padding: 12px 16px; display: grid; gap: 6px; }
.q .for { font-size: 13px; color: var(--orange-2); }
.q .ask { font-size: 20px; }
.q .why { font-size: 13.5px; color: var(--ink-3); }
.btn { border: 0; cursor: pointer; border-radius: 999px; padding: 8px 18px; font-size: 16px; background: var(--card); box-shadow: var(--shadow-md); }
.btn.sky { background: var(--sky); color: #fff; }
.btn.orange { background: var(--orange); color: #fff; }
.btn[aria-pressed="true"] { background: var(--green-2); color: #fff; }
.q .btn { justify-self: start; font-size: 14.5px; padding: 6px 14px; }
.q.stretch { background: #f0f9ff; border-color: #7dd3fc; }
.q.stretch .for { color: var(--sky-ink); }

/* 이야기 구조 */
.path { display: block; width: 100%; max-width: 720px; margin-inline: auto; }
.stack { display: flex; height: 18px; border-radius: 999px; overflow: hidden; gap: 3px; background: var(--line); }
.stack span { display: block; height: 100%; }
.lrow { display: grid; grid-template-columns: 68px minmax(0, 1fr); gap: 10px; align-items: center; font-size: 14px; color: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13.5px; color: var(--ink-2); }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }

/* 막대 */
.bars { display: grid; grid-template-columns: auto minmax(0, 1fr) 30px; gap: 8px 12px; align-items: center; font-size: 15px; }
.bars .track { display: block; height: 10px; border-radius: 999px; background: var(--line); overflow: hidden; }
.bars .fill { display: block; height: 100%; border-radius: 999px; background: var(--sky-2); }
.bars .n { text-align: right; color: var(--ink-2); }
.dumb { display: grid; gap: 12px; }
.dumb .lab { display: flex; justify-content: space-between; gap: 10px; font-size: 15px; }
.dumb .lab span:last-child { color: var(--ink-3); white-space: nowrap; }

/* 우리 반 */
.heat-wrap { overflow-x: auto; }
table.heat { border-collapse: separate; border-spacing: 4px; width: 100%; font-size: 15px; }
.heat th { font-weight: 400; color: var(--ink-3); font-size: 13.5px; padding: 2px 6px; text-align: center; white-space: nowrap; }
.heat th.name, .heat td.name { text-align: left; white-space: nowrap; }
.heat td { height: 34px; text-align: center; border-radius: 10px; }
.heat td.name { padding-right: 8px; }
.heat td.chg { white-space: nowrap; font-size: 13px; color: var(--ink-2); text-align: left; padding-left: 6px; }
.heat tbody tr { cursor: pointer; }
.heat tbody tr:hover td.name { color: var(--sky-ink); }
.foot { font-size: 13px; color: var(--ink-4); }

@media (max-width: 900px) { .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 760px) {
  .grid2 { grid-template-columns: 1fr; }
  .boxes { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .head h1 { font-size: 24px; }
  .head .report { margin-left: 0; width: 100%; }
}
@media (max-width: 480px) { .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .card { padding: 18px 16px; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="page" id="page"></div>

<script>
const DATA = __DATA__;
const REPORT = "__REPORT__";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (v, d = 1) => (v === null || v === undefined) ? "–" : (Math.round(v * 10 ** d) / 10 ** d).toLocaleString("ko-KR");
const sd = (iso) => { const [, m, d] = iso.split("-").map(Number); return `${m}/${d}`; };
const KEYS = ["mlc_m", "conn_rate", "sg_score", "ist_rate", "self_ratio"];
const UNITS = ["배경", "등장인물", "시작 사건", "목표", "시도", "결과", "반응"];
const ageText = (m) => m == null ? "" : `만 ${Math.floor(m / 12)}세 ${m % 12}개월`;
let current = (DATA.children.find((k) => k.growth_up.length) || DATA.children[0]).child_id;

function weekLabel() { const s = new Date(DATA.class.week_start), e = new Date(s); e.setDate(e.getDate() + 6);
  return `${s.getMonth() + 1}월 ${s.getDate()}일 ~ ${e.getMonth() + 1}월 ${e.getDate()}일`; }

/* ---------- 그래프 조각 ---------- */
function spark(vals) {
  const W = 170, H = 50, P = 4, xs = (i) => P + i * (W - 2 * P) / 5;
  const ok = vals.filter((v) => v !== null); if (!ok.length) return "";
  let lo = Math.min(...ok), hi = Math.max(...ok); if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const y = (v) => H - P - (v - lo) * (H - 2 * P) / (hi - lo);
  const pts = vals.map((v, i) => v === null ? null : [xs(i), y(v)]).filter(Boolean);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
  const area = `M${pts[0][0]},${H - P}` + pts.map((p) => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("") + `L${pts[pts.length - 1][0]},${H - P}Z`;
  const mid = (xs(2) + xs(3)) / 2, last = pts[pts.length - 1];
  return `<svg viewBox="0 0 ${W} ${H + 14}" role="img" aria-label="최근 6주 주별 값">
    <rect x="${mid}" y="0" width="${W - mid}" height="${H}" rx="8" fill="var(--sky-soft)" opacity="0.6"/>
    <path d="${area}" fill="var(--sky-2)" opacity="0.14"/>
    <path d="${line}" fill="none" stroke="var(--sky)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${last[0]}" cy="${last[1]}" r="4" fill="var(--sky)" stroke="#fff" stroke-width="2"/>
    <text x="${P}" y="${H + 12}">6주 전</text><text x="${mid + 4}" y="${H + 12}">최근 3주</text></svg>`;
}

function ring(pct) {
  const r = 13, c = 2 * Math.PI * r, v = Math.max(0, Math.min(100, pct || 0));
  return `<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><circle cx="17" cy="17" r="${r}" fill="none" stroke="var(--line)" stroke-width="5"/>
    <circle cx="17" cy="17" r="${r}" fill="none" stroke="var(--green-2)" stroke-width="5" stroke-linecap="round"
      stroke-dasharray="${(c * v / 100).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 17 17)"/></svg>`;
}

function storyPath(units, ref) {
  const W = 720, H = ref ? 122 : 104, P = 46, step = (W - 2 * P) / (UNITS.length - 1);
  let s = `<svg class="path" viewBox="0 0 ${W} ${H}" role="img" aria-label="이야기문법 단위별로 그 단위가 들어간 이야기 비율">
    <line x1="${P}" x2="${W - P}" y1="36" y2="36" stroke="var(--line)" stroke-width="6" stroke-linecap="round"/>`;
  UNITS.forEach((u, i) => {
    const v = units[u] ?? 0, x = P + i * step;
    s += `<circle cx="${x}" cy="36" r="20" fill="#fff" stroke="var(--sky-soft)" stroke-width="4"/>
      <circle cx="${x}" cy="36" r="${(5 + 12 * v / 100).toFixed(1)}" fill="var(--sky)" opacity="${(0.3 + 0.7 * v / 100).toFixed(2)}"/>
      <text x="${x}" y="78" text-anchor="middle" style="fill:var(--ink);font-size:14px">${u}</text>
      <text x="${x}" y="96" text-anchor="middle" style="font-size:12.5px">${v}%</text>`
      + (ref ? `<text x="${x}" y="114" text-anchor="middle" style="font-size:11.5px;fill:var(--orange-2)">또래 ${ref[u]}%</text>` : "");
  });
  return s + "</svg>";
}

const LEVEL_COLORS = ["var(--line-2)", "#7dd3fc", "var(--sky)", "#0369a1"];
function ladder(a, b) {
  const lv = Object.keys(b.episode_levels);
  const bar = (x) => { const t = lv.reduce((s, k) => s + x.episode_levels[k], 0) || 1;
    return `<div class="stack">${lv.map((k, i) => x.episode_levels[k] ? `<span style="width:${100 * x.episode_levels[k] / t}%;background:${LEVEL_COLORS[i]}" title="${k} ${x.episode_levels[k]}편"></span>` : "").join("")}</div>`; };
  return `<div class="lrow"><span>그 전 3주</span>${bar(a)}</div><div class="lrow"><span>최근 3주</span>${bar(b)}</div>
    <div class="legend">${lv.map((k, i) => `<span><i style="background:${LEVEL_COLORS[i]}"></i>${k}</span>`).join("")}</div>`;
}

function bars(obj) {
  const max = Math.max(1, ...Object.values(obj));
  return `<div class="bars">${Object.entries(obj).map(([k, v]) => `<span>${esc(k)}</span>
    <span class="track"><span class="fill" style="width:${100 * v / max}%;opacity:${v ? 1 : 0}"></span></span><span class="n num">${v}</span>`).join("")}</div>`;
}

function dumbbell(label, a, b, max, unit, betterLow) {
  const W = 300, H = 24, P = 10, x = (v) => P + (W - 2 * P) * Math.min(v ?? 0, max) / max;
  const good = ((b ?? 0) >= (a ?? 0)) !== !!betterLow, col = good ? "var(--sky)" : "var(--orange)";
  return `<div><div class="lab"><span>${label}</span><span class="num">${fmt(a)} → <b style="color:var(--ink)">${fmt(b)}</b> ${unit}</span></div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" aria-hidden="true"><line x1="${P}" x2="${W - P}" y1="12" y2="12" stroke="var(--line)" stroke-width="8" stroke-linecap="round"/>
    <line x1="${x(a)}" x2="${x(b)}" y1="12" y2="12" stroke="${col}" stroke-width="4" stroke-linecap="round"/>
    <circle cx="${x(a)}" cy="12" r="6" fill="var(--line-2)"/><circle cx="${x(b)}" cy="12" r="7" fill="${col}" stroke="#fff" stroke-width="2"/></svg></div>`;
}

/* ---------- 아이별 ---------- */
function childView() {
  const r = DATA.reports[current], d = r.development, c = r.cards;
  const g = Object.fromEntries(d.six_week_growth.map((m) => [m.key, m]));
  const a = d.refined.before_3w, b = d.refined.recent_3w, ev = d.growth_evidence;
  const metric = (k) => { const m = g[k]; if (!m) return "";
    const diff = (m.value ?? 0) - (m.baseline ?? 0);
    const tag = m.change === "up" ? '<span class="tag up">▲ 확실히 늘었어요</span>' : m.change === "down" ? '<span class="tag down">▼ 확실히 줄었어요</span>'
      : `<span class="d">그 전 3주 ${fmt(m.baseline)} (${diff >= 0 ? "+" : ""}${fmt(diff)})</span>`;
    return `<div class="metric"><span class="k">${esc(m.label)} <span class="qmark" tabindex="0" title="${esc(m.help)}">?</span></span>
      <span class="v num">${fmt(m.value)} <small>${esc(m.unit)}</small></span>${tag}${spark(d.weekly[k])}</div>`; };
  const news = d.milestones.length ? `<ul class="news">${d.milestones.map((m) => `<li><span class="dot"></span><span>${esc(m.skill)} <span class="foot">${sd(m.date)}부터 · 그림책 ${m.stories}권</span><q>${esc(m.text)}</q></span></li>`).join("")}</ul>`
    : '<p class="foot">최근 3주에 새로 쓰기 시작한 표현은 없어요.</p>';
  const qs = d.next_steps.length ? d.next_steps.map((s) => `<div class="q${s.tier === "한 단계 더" ? " stretch" : ""}"><span class="for">${s.tier === "한 단계 더" ? "한 단계 더 · " : "먼저 해 볼 질문 · "}${esc(s.target)}</span><span class="ask">“${esc(s.question)}”</span>
      <span class="why">${esc(s.reason)}</span><button class="btn" aria-pressed="false" data-toggle="토리 질문에 넣었어요">토리 질문에 넣기</button></div>`).join("")
    : '<p class="foot">최근 그림책에 이야기 요소가 고루 들어 있어요.</p>';
  const themes = r.themes.map((t) => `<span class="chip${t.character ? " char" : ""}${t.increased ? " hot" : ""}">${esc(t.word)}<span class="c">${t.count}</span>${t.increased ? " ↑" : ""}</span>`).join("");
  const n3 = d.recent_3w.sentences;
  return `
  <section class="card"><h2>한눈에 보기 <span class="cap">${esc(current.replace("아이 ", "햇살반 "))} · ${ageText(r.age_months)}</span></h2>
    ${n3 < 50 ? `<p class="notice inside">참고치: 최근 3주 문장이 ${n3}개예요. 50개가 넘으면 숫자가 더 믿을 만해져요.</p>` : ""}
    <div class="boxes">
      <div class="box"><span class="k">이번 주 그림책</span><span class="v num">${r.books}<small>권</small></span><span class="s">그림책 한 권에 문장 ${fmt(d.recent_3w.sentences_per_story)}개${d.age_ref ? ` · 또래 ${d.age_ref.sentences_per_story}개` : ""}</span></div>
      <div class="box"><span class="k">스스로 이어 말한 장면</span><span class="v num">${ring(100 * c.self_scenes.value / Math.max(1, c.self_scenes.total))}${c.self_scenes.value}<small>/${c.self_scenes.total}</small></span><span class="s">선택지 도움 ${c.self_scenes.choice_help}번</span></div>
      <div class="box"><span class="k">평균 문장 길이</span><span class="v num">${fmt(c.mean_eojeol.value)}<small>어절</small></span><span class="s">형태소로는 ${fmt(c.mlc_m.value)}개 · 지난주 ${fmt(c.mlc_m.last_week)}개</span></div>
      <div class="box"><span class="k">처음 쓴 낱말</span><span class="v num">${c.new_words.count}<small>개</small></span><span class="s">${esc(c.new_words.examples.join(", ")) || "–"}</span></div>
    </div></section>

  <section class="card"><h2>6주 동안 달라진 점</h2>
    <p class="cap">큰 숫자는 최근 3주 평균이에요. ‘확실히 늘었어요/줄었어요’는 그림책마다 생기는 오르내림보다 확실히 클 때만 붙어요.</p>
    <div class="metrics">${KEYS.map(metric).join("")}</div></section>

  <div class="grid2">
    <section class="card"><h2>이야기가 이렇게 달라졌어요</h2>
      <div class="talk">
        ${ev.before ? `<div class="say"><span class="when">그 전 3주<br>${sd(ev.before.date)}</span><span class="bubble">${esc(ev.before.text)}</span></div>` : ""}
        ${ev.recent ? `<div class="say"><span class="when">최근 3주<br>${sd(ev.recent.date)}</span><span class="bubble now">${esc(ev.recent.text)}</span></div>` : ""}
      </div>
      <p class="cap">각 기간의 보통 장면이에요. 가장 잘한 장면을 고르지 않았어요.</p>
      <span class="sub">새로 쓰기 시작한 표현</span>${news}</section>
    <section class="card"><h2>다음 이야기 시간에 물어볼까요?</h2>
      <div class="qs">${qs}</div>
      <p class="cap">‘먼저 해 볼 질문’은 4~5세 대부분이 쓰는 요소(배경·등장인물·사건·시도·결과)가 또래보다 적을 때 나와요. 이 요소들이 갖춰지면 ‘한 단계 더’ 질문이 하나 나와요.</p></section>
  </div>

  <section class="card"><h2>이야기 구조</h2>
      <p class="cap">최근 3주 그림책 중 그 요소가 들어간 비율</p>
      ${storyPath(d.recent_3w.units, d.age_ref)}
      ${d.age_ref ? `<p class="cap">주황 글씨는 ${d.age_ref.age}세 또래가 새 이야기를 지어 말할 때 그 요소를 쓴 비율이에요(한유진·유안진, 2001, 20명). ‘목표’와 ‘반응’은 5세도 대부분 쓰지 않는 것이 정상이에요.</p>` : ""}
      <span class="sub">이야기 짜임 <span class="qmark" tabindex="0" title="목표 → 시도 → 결과를 다 갖춘 것이 완전한 에피소드예요">?</span></span>
      ${a && b ? ladder(a, b) : '<p class="foot">기록이 더 필요해요.</p>'}</section>

  <div class="grid2">
    <section class="card"><h2>말하는 방식</h2>
      <p class="cap">그 전 3주 → 최근 3주</p>
      ${a && b ? `<div class="dumb">
        ${dumbbell("인물을 이름으로 소개", a.intro_explicit_pct, b.intro_explicit_pct, 100, "%")}
        ${dumbbell("이어진 문장", a.complex_pct, b.complex_pct, 60, "%")}
        ${dumbbell("인물의 말 옮기기", a.quotes_per_story, b.quotes_per_story, 1, "회/권")}
        ${dumbbell("‘걔·그거’로 가리키기", a.pronouns_per_10_sent, b.pronouns_per_10_sent, 2, "회/10문장", true)}
        ${dumbbell("서로 다른 낱말", a.ndw50, b.ndw50, 45, "개/50낱말")}</div>` : '<p class="foot">기록이 더 필요해요.</p>'}</section>
    <section class="card"><h2>잇는 말과 마음을 나타내는 말</h2>
      <span class="sub">잇는 말 <span class="foot">최근 3주</span></span>${bars(d.recent_3w.conn)}
      ${d.connectives.example ? `<p class="cap">예: “${esc(d.connectives.example)}”</p>` : ""}
      <span class="sub">마음을 나타내는 말 <span class="foot">최근 3주</span></span>${bars(d.recent_3w.ist)}</section>
  </div>
  <section class="card"><h2>자주 나온 낱말과 소재</h2>
      <div class="chips">${themes}</div>
      <p class="cap">테두리만 있는 칩은 등장인물이에요. 하늘색은 지난 4주보다 늘어난 소재예요.</p></section>`;
}

/* ---------- 우리 반 ---------- */
function classView() {
  const heat = (v) => v === null ? "var(--line)" : `color-mix(in srgb, var(--sky) ${Math.round(8 + 0.8 * v)}%, #ffffff)`;
  const ink = (v) => v !== null && v >= 65 ? "#ffffff" : "var(--ink)";
  const rows = DATA.children.map((k) => { const u = DATA.reports[k.child_id].development.recent_3w.units;
    const chg = (k.growth_up.length ? `<span style="color:var(--green)">▲</span> ${esc(k.growth_up.join(", "))} ` : "") + (k.growth_down.length ? `<span style="color:var(--orange-2)">▼</span> ${esc(k.growth_down.join(", "))}` : "");
    return `<tr data-id="${esc(k.child_id)}"><td class="name">${esc(k.child_id)}</td>
      ${UNITS.map((x) => `<td class="num" style="background:${heat(u[x])};color:${ink(u[x])}">${u[x] ?? "–"}</td>`).join("")}<td class="chg">${chg}</td></tr>`; }).join("");
  const cnt = {}; for (const k of DATA.children) for (const s of DATA.reports[k.child_id].development.next_steps) cnt[s.target] = (cnt[s.target] || 0) + 1;
  const order = ["배경", "등장인물", "시작 사건", "결과", "시도", "목표", "반응", "원인 이음말", "인용", "생각을 나타내는 말"].filter((t) => cnt[t]);
  const avg = (u) => Math.round(DATA.children.reduce((s, k) => s + (DATA.reports[k.child_id].development.recent_3w.units[u] || 0), 0) / DATA.children.length);
  return `
  <section class="card"><h2>우리 반 이야기 구조</h2><p class="cap">최근 3주 · 아이별 비율의 평균</p>
    ${storyPath(Object.fromEntries(UNITS.map((u) => [u, avg(u)])))}</section>
  <section class="card"><h2>아이별 이야기 구조</h2>
    <p class="cap">최근 3주 그림책 중 그 요소가 들어간 비율(%) · 이름순이에요, 순위가 아니에요 · 줄을 누르면 그 아이 화면으로 가요</p>
    <div class="heat-wrap"><table class="heat"><thead><tr><th class="name">이름</th>${UNITS.map((u) => `<th>${u}</th>`).join("")}<th class="name">6주 변화</th></tr></thead><tbody>${rows}</tbody></table></div></section>
  <section class="card"><h2>함께 연습하면 좋은 이야기 요소</h2><p class="cap">이 질문을 제안받은 아이 수</p>
    ${bars(Object.fromEntries(order.map((t) => [t, cnt[t]])))}
    <p class="cap">여러 아이가 같은 요소에서 질문을 받았다면, 반 전체 이야기 시간에 그 요소를 함께 연습해 볼 수 있어요. 예: 목표 → “주인공은 무엇을 하고 싶었을까?”</p></section>`;
}

function render() {
  const view = location.hash === "#class" ? "class" : "child";
  const pills = DATA.children.map((k) => {
    const m = (k.growth_up.length ? '<span class="mk up">▲</span>' : "") + (k.growth_down.length ? '<span class="mk down">▼</span>' : "") + (k.milestones ? '<span class="dot"></span>' : "");
    return `<button class="pill" data-id="${esc(k.child_id)}" aria-pressed="${view === "child" && k.child_id === current}">${esc(k.child_id.replace("아이 ", "햇살반 "))}${m}</button>`; }).join("");
  document.getElementById("page").innerHTML = `
  <div class="head"><a class="home" href="#child">🏠 처음으로</a><h1><span aria-hidden="true">👩‍🏫</span>보호자·선생님 화면</h1>
    <a class="report" href="${REPORT}">분석 리포트 보기</a></div>
  <p class="notice">데모 화면이에요. 가상 반 18명의 6주 기록(${weekLabel()}까지)으로 계산했고 실제 아이의 말이 아니에요.</p>
  <div class="choose">
    <div class="row"><div class="switch" role="tablist"><a href="#child" class="${view === "child" ? "on" : ""}">아이별</a><a href="#class" class="${view === "class" ? "on" : ""}">우리 반</a></div>
      <span class="legend"><span><span style="color:var(--green)">▲</span> 확실히 늘어난 지표</span><span><span style="color:var(--orange-2)">▼</span> 확실히 줄어든 지표</span><span><span class="pill-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--amber)"></span> 새로 쓰기 시작한 표현</span></span></div>
    ${view === "child" ? `<div class="pills">${pills}</div>` : ""}
  </div>
  ${view === "class" ? classView() : childView()}
  <p class="foot">보호자 동의 ${DATA.class.consent.given}/${DATA.class.consent.total} · 담임 선생님만 볼 수 있어요</p>`;
  for (const el of document.querySelectorAll(".pill")) el.addEventListener("click", () => { current = el.dataset.id; render(); });
  for (const tr of document.querySelectorAll(".heat tbody tr")) tr.addEventListener("click", () => { current = tr.dataset.id; location.hash = "child"; });
  for (const btn of document.querySelectorAll("[data-toggle]")) { const label = btn.textContent;
    btn.addEventListener("click", () => { const on = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", on); btn.textContent = on ? `✓ ${btn.dataset.toggle}` : label; }); }
}
window.addEventListener("hashchange", render);
render();
</script>
"""


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.replace("__DATA__", payload).replace("__REPORT__", REPORT_URL), encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
