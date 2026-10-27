# -*- coding: utf-8 -*-
import json, io, re, os
D = os.path.dirname(os.path.abspath(__file__))
dl = os.path.join(D, "raw")
k12 = json.load(io.open(os.path.join(dl, "k12_poems.json"), encoding="utf-8"))
old = json.load(io.open(os.path.join(dl, "poem300.json"), encoding="utf-8"))

primary = []
for p in k12["poems"]:
    g = p.get("grade")
    stage = str(p.get("stage") or "")
    if not (stage.startswith("小学") or (isinstance(g, int) and 1 <= g <= 6)):
        continue
    lines = p.get("linesPunct") or []
    if not lines:
        lines = p.get("lines") or []
    lines = [s.strip() for s in lines if s and s.strip()]
    if not lines:
        continue
    primary.append({
        "title": (p.get("title") or "").strip(),
        "author": (p.get("author") or "").strip(),
        "type": (p.get("form") or "").strip(),
        "grade": g if isinstance(g, int) else 0,
        "lines": lines,
    })
print("k12小学", len(primary))

# 老诗库 319 首里挑短诗（<=8 句）做扩展
old_short = []
for p in old:
    raw = (p.get("contents") or "").split("\n")
    lines = [s.strip() for s in raw if s.strip()]
    if not lines or len(lines) > 8:
        continue
    old_short.append({
        "title": (p.get("title") or "").strip(),
        "author": (p.get("author") or "").strip(),
        "type": (p.get("type") or "").strip(),
        "grade": 0,
        "lines": lines,
    })
print("老库短诗", len(old_short))

titles = {p["title"] for p in primary}
merged = list(primary)
added = 0
for p in old_short:
    if p["title"] in titles:
        continue
    titles.add(p["title"])
    merged.append(p)
    added += 1
print("扩展补充", added, "合计", len(merged))

# 覆盖率抽查
probe = ["静夜思","咏鹅","春晓","悯农","江雪","小池","敕勒歌","赠汪伦","望庐山瀑布","绝句","村居","画","风","池上","所见","山行","九月九日忆山东兄弟","游子吟","题西林壁","饮湖上初晴后雨"]
have = {p["title"] for p in merged}
print("命中", [t for t in probe if t in have])
print("缺失", [t for t in probe if t not in have])

io.open(os.path.join(D, "data", "poems.json"), "w", encoding="utf-8").write(
    json.dumps(merged, ensure_ascii=False, indent=1))
print("写出 poems.json", os.path.getsize(os.path.join(D, "data", "poems.json")), "字节")
