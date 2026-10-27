# -*- coding: utf-8 -*-
"""把 raw/ 下的原始语料压成 server 可直接加载的紧凑数据文件。

用法：python3 _build_data.py   （原始语料在 raw/，产物写到 data/）
"""
import csv, json, os, re, glob, io

ROOT = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

def dump(name, obj, **kw):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, **kw)
    print("%-26s %8.1f KB" % (name, os.path.getsize(p) / 1024.0))

# ---------- 1. 成语词典（word|pinyin|explanation） ----------
idi = json.load(open(os.path.join(DL, "idiom.json"), encoding="utf-8"))
lines = []
seen = set()
for it in idi:
    w = (it.get("word") or "").strip()
    if len(w) != 4 or not re.fullmatch(r"[\u4e00-\u9fa5]{4}", w):
        continue
    if w in seen:
        continue
    seen.add(w)
    py = (it.get("pinyin") or "").strip()
    ex = (it.get("explanation") or "").strip().replace("\n", " ")
    ex = re.sub(r"\s+", " ", ex)[:120]
    lines.append("%s|%s|%s" % (w, py, ex))
p = os.path.join(OUT, "idioms_dict.txt")
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("%-26s %8.1f KB  条数=%d" % ("idioms_dict.txt", os.path.getsize(p) / 1024.0, len(lines)))

# ---------- 2. 古诗 319 ----------
poems = json.load(open(os.path.join(DL, "poem300.json"), encoding="utf-8"))
out = []
for p0 in poems:
    title = (p0.get("title") or "").strip()
    body = (p0.get("contents") or "").strip()
    if not title or not body:
        continue
    out.append({
        "title": title,
        "author": (p0.get("author") or "").strip(),
        "type": (p0.get("type") or "").strip(),
        "lines": [x.strip() for x in body.split("\n") if x.strip()],
    })
dump("poems.json", out, indent=0)
print("   古诗 %d 首" % len(out))

# ---------- 3. 小学英语 8 册 ----------
grades = ["三上","三下","四上","四下","五上","五下","六上","六下"]
files = ["PEPXiaoXue3_1_T.json","PEPXiaoXue3_2_T.json","PEPXiaoXue4_1_T.json","PEPXiaoXue4_2_T.json",
         "PEPXiaoXue5_1_T.json","PEPXiaoXue5_2_T.json","PEPXiaoXue6_1_T.json","PEPXiaoXue6_2_T.json"]
words, wseen = [], set()
for g, fn in zip(grades, files):
    arr = json.load(open(os.path.join(DL, fn), encoding="utf-8"))
    for it in arr:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in wseen:
            continue
        wseen.add(key)
        trans = it.get("trans") or []
        cn = trans[0] if isinstance(trans, list) and trans else str(trans or "")
        words.append({
            "word": name,
            "cn": str(cn).strip(),
            "us": (it.get("usphone") or "").strip(),
            "uk": (it.get("ukphone") or "").strip(),
            "grade": g,
        })
dump("english_words.json", words, indent=0)
print("   单词 %d 个" % len(words))

# ---------- 4. 组词索引：首字 -> 常用词 ----------
allw = json.load(open(os.path.join(DL, "word.json"), encoding="utf-8"))
idx = {}
for it in allw:
    w = (it.get("word") or "").strip()
    if not (2 <= len(w) <= 4):
        continue
    if not re.fullmatch(r"[\u4e00-\u9fa5]+", w):
        continue
    idx.setdefault(w[0], []).append(w)
# 每字最多留 12 个，短词优先（更可能是常用词）
for ch in idx:
    vs = sorted(set(idx[ch]), key=lambda x: (len(x), x))
    idx[ch] = vs[:12]
dump("words_index.json", idx, separators=(",", ":"))
print("   组词索引 %d 个字" % len(idx))

# ---------- 5. 故事库 ----------
stories = []
with open(os.path.join(DL, "aesop.csv"), encoding="utf-8-sig") as f:
    for row in csv.reader(f):
        if len(row) < 4 or row[0] == "ID":
            continue
        title = (row[1] or "").strip()
        text = (row[2] or "").strip()
        moral = (row[3] or "").strip()
        if not title or len(text) < 40:
            continue
        stories.append({"title": title, "text": text, "moral": moral, "src": "伊索寓言"})

for pth in sorted(glob.glob(os.path.join(DL, "grimms", "*.md"))):
    raw = open(pth, encoding="utf-8").read()
    raw = re.sub(r"^#\s*", "", raw, count=1).strip()
    first_nl = raw.find("\n")
    if first_nl < 0:
        continue
    title = raw[:first_nl].strip()
    body = raw[first_nl:].strip()
    # 去掉重复标题行
    body = re.sub(r"^" + re.escape(title) + r"\s*\n+", "", body).strip()
    body = re.sub(r"\s*\n\s*", "", body)
    if len(title) < 2 or len(body) < 150:
        continue
    stories.append({"title": title, "text": body, "moral": "", "src": "格林童话"})

dump("stories.json", stories, separators=(",", ":"))
print("   故事 %d 篇（伊索 %d + 格林 %d）" % (
    len(stories),
    sum(1 for s in stories if s["src"] == "伊索寓言"),
    sum(1 for s in stories if s["src"] == "格林童话")))
print("\n完成，输出目录：", OUT)
