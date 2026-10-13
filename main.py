import json
import re
from bisect import bisect_left
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel


class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None


class LinksResponse(BaseModel):
    links: List[LawLink]


class TextRequest(BaseModel):
    text: str


_ENDINGS = sorted(
    ["ого", "его", "ому", "ему", "ыми", "ими", "ами", "ями", "ой", "ей", "ий", "ый",
     "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ых", "их", "ым", "им", "ом",
     "ем", "ов", "ев", "ам", "ям", "ах", "ях", "ью", "а", "я", "у", "ю", "ы", "и",
     "е", "о", "ь"],
    key=lambda s: -len(s),
)
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+|\d+|№")
_QUOTES = "«»\"“”„'`"


def stem(word: str) -> str:
    w = word.lower().replace("ё", "е")
    if w == "n":
        return "№"
    if len(w) > 4 and w.isalpha():
        for e in _ENDINGS:
            if w.endswith(e) and len(w) - len(e) >= 3:
                return w[: -len(e)]
    return w


def tokenize(s: str):
    stems, starts, ends, origs = [], [], [], []
    for m in _TOKEN_RE.finditer(s):
        o = m.group()
        stems.append(stem(o))
        starts.append(m.start())
        ends.append(m.end())
        origs.append(o)
    return stems, starts, ends, origs


class LawIndex:
    def __init__(self, aliases: Dict[str, List[str]]):
        self.full: dict = {}
        self.title: dict = {}
        self.num: Dict[tuple, set] = {}
        self.date: Dict[tuple, set] = {}
        num_re = re.compile(r"№\s*(\d+)\s*-\s*([a-zа-я]+)")
        date_re = re.compile(r"от\s+(\d{2})\.(\d{2})\.(\d{4})")
        for k, lst in aliases.items():
            lid = int(k)
            for a in lst:
                st = tokenize(a)[0]
                if not st:
                    continue
                self._insert(self.full, st, lid)

                qs = [i for i, c in enumerate(a) if c in "«»\"“”„"]
                if len(qs) >= 2:
                    t = tokenize(a[qs[0] + 1: qs[-1]])[0]
                    if len(t) >= 3:
                        self._insert(self.title, t, lid)
                al = a.lower()
                m = num_re.search(al)
                if m:
                    self.num.setdefault((m.group(1), stem(m.group(2))), set()).add(lid)
                m = date_re.search(al)
                if m:
                    self.date.setdefault(m.groups(), set()).add(lid)

    @staticmethod
    def _insert(root, toks, lid):
        node = root
        for t in toks:
            node = node.setdefault(t, {})
        node.setdefault(None, set()).add(lid)

    @staticmethod
    def match(root, stems, origs, i):
        if i >= len(stems):
            return None
        node = root.get(stems[i])
        if node is None:
            return None
        if len(stems[i]) <= 3 and stems[i].isalpha() and not origs[i].isupper():
            return None
        best = None
        j = i
        while True:
            if None in node:
                best = (j + 1, node[None])
            j += 1
            if j >= len(stems):
                break
            node = node.get(stems[j])
            if node is None:
                break
        return best


L_SUB, L_PT, L_PART, L_ART, L_ABZ = "sub", "pt", "part", "art", "abz"
RANK = {L_SUB: 0, L_PT: 1, L_PART: 1, L_ART: 2}

_MARK = re.compile(
    r"(?<![а-яa-z0-9])(?:"
    r"(?P<art>статья|статьи|статье|статью|статьей|статей|статьям|статьях|статьями|"
    r"ст\s*\.\s*ст\s*\.|ст\s*\.|ст(?=\s*\d))"
    r"|(?P<abz>абзац\w*|абз\s*\.)"
    r"|(?P<sub>подпункт\w*|подп\s*\.|пп\s*\.|пп(?=\s*[\d\s]))"
    r"|(?P<pt>п\s*\.\s*п\s*\.|пункт\w*|пунт\w*|п\s*\.|п(?=\s+[\dа-я]))"
    r"|(?P<part>част(?:ь|и|ью|ей|ям|ях|ями)(?![а-я])|ч\s*\.|ч(?=\s+\d))"
    r")"
)

_NUM = re.compile(
    r"\d+(?:\.\d+)*(?:\s?[-–—]\s?\d+(?:\.\d+)*)?(?:\^\d+)?(?:[а-я](?![а-я\d]))?"
)
_LET = re.compile(r"[а-яa-z](?![а-яa-z0-9])")
_SEP = re.compile(r"\s*(?:,\s*(?:и\s+|или\s+)?|;\s*|и\s+|или\s+|а\s+также\s+)")
_AMBIG_LET = set("вскоуия")
_LET_OK_NEXT = re.compile(
    r"\s*(?:[,;)]|$|(?:ст|п|пп|ч|подп)(?![а-я])|(?:стать|пункт|подпункт|част|абз))"
)
_NUM_BAD_NEXT = re.compile(
    r"\s*(?:года|год|г\.|г\b|%|руб|тыс|млн|дн|лет|человек|месяц|раз|процент|коп)"
)
_CONNECT = re.compile(r"[\s,;]*(?:(?:и|в|из|к|или|а\s+также)\s+)?")
_GAP_BEFORE = re.compile(r"[\s,(:]*")


def _clean_val(v: str) -> str:
    v = re.sub(r"\s*[–—-]\s*", "-", v.strip())
    return v


def _parse_values(tl: str, t: str, pos: int, allow_letters: bool):
    n = len(tl)
    m = re.compile(r"\s*(?:№\s*)?").match(tl, pos)
    p = m.end()
    vals: List[str] = []
    end = pos
    last_letter = False
    first = True
    while True:
        if not first:
            sm = _SEP.match(tl, p)
            if not sm:
                break
            p2 = sm.end()
        else:
            p2 = p
        vm = _NUM.match(tl, p2)
        if vm:
            if not first and _NUM_BAD_NEXT.match(tl, vm.end()):
                break
            if not first and re.fullmatch(r"(19|20)\d\d", vm.group()):
                break
            if first and _NUM_BAD_NEXT.match(tl, vm.end()) and False:
                break
            vals.append(_clean_val(t[p2:vm.end()]))
            end = vm.end()
            p = end
            last_letter = False
            first = False
            continue
        if allow_letters and (first or last_letter):
            lm = _LET.match(tl, p2)
            if lm:
                ch = lm.group()
                if ch in _AMBIG_LET and not _LET_OK_NEXT.match(tl, lm.end()):
                    break
                vals.append(t[p2:lm.end()])
                end = lm.end()
                p = end
                last_letter = True
                first = False
                continue
        break
    return vals, end


class Group:
    __slots__ = ("level", "vals", "start", "end")

    def __init__(self, level, vals, start, end):
        self.level, self.vals, self.start, self.end = level, vals, start, end


def find_groups(t: str, tl: str) -> List[Group]:
    groups: List[Group] = []
    last_end = 0
    for m in _MARK.finditer(tl):
        if m.start() < last_end:
            continue
        level = m.lastgroup
        vals, end = _parse_values(tl, t, m.end(), allow_letters=(level != L_ART))
        if not vals:
            continue
        groups.append(Group(level, vals, m.start(), end))
        last_end = end
    return groups


class Chain:
    __slots__ = ("groups", "direction", "law", "start", "end")

    def __init__(self, g: Group):
        self.groups = [g]
        self.direction = 0
        self.law = None
        self.start = g.start
        self.end = g.end

    def try_add(self, g: Group, tl: str) -> bool:
        last = self.groups[-1]
        gap = tl[last.end:g.start]
        if len(gap) > 25 or not _CONNECT.fullmatch(gap):
            return False
        if g.level == L_ABZ:
            return False
        if any(x.level == g.level for x in self.groups):
            return False
        r_new, r_last = RANK[g.level], RANK[last.level]
        if r_new == r_last and self.direction == 0:
            pass
        elif self.direction == 0:
            self.direction = 1 if r_new > r_last else -1
        elif self.direction == 1 and r_new < r_last:
            return False
        elif self.direction == -1 and r_new > r_last:
            return False
        self.groups.append(g)
        self.end = g.end
        return True


def build_chains(groups: List[Group], tl: str) -> List[Chain]:
    chains: List[Chain] = []
    cur: Optional[Chain] = None
    pending_abz = False
    for g in groups:
        if g.level == L_ABZ:

            if cur is not None:
                gap = tl[cur.end:g.start]
                if len(gap) <= 25 and _CONNECT.fullmatch(gap):
                    cur.end = g.end
                    cur.groups[-1] = Group(cur.groups[-1].level, cur.groups[-1].vals,
                                           cur.groups[-1].start, g.end)
                    continue
            continue
        if cur is not None and cur.try_add(g, tl):
            continue
        cur = Chain(g)
        chains.append(cur)
    return chains


def expand(chain: Chain):
    art = pt = sub = part = None
    for g in chain.groups:
        if g.level == L_ART:
            art = g.vals
        elif g.level == L_PT:
            pt = g.vals
        elif g.level == L_PART:
            part = g.vals
        elif g.level == L_SUB:
            sub = g.vals
    if part is not None and pt is not None:
        point, subp = part, pt
    elif part is not None:
        point, subp = part, sub
    else:
        point, subp = pt, sub
    return art or [None], point or [None], subp or [None]


_ALLOWED_SKIP = {
    "федеральн", "закон", "фз", "фкз", "российск", "федерац", "рф", "росс", "от",
    "№", "г", "год", "настоящ", "указ", "президент", "распоряжен", "рп", "кодекс",
    "постановлен", "правительств", "приказ", "положен", "стандарт",
}
_ALLOWED_SKIP |= {stem(w) for w in (
    "января февраля марта апреля мая июня июля августа сентября октября ноября декабря "
    "года г").split()}
_BAD_GAP = re.compile(r"[.;:!?]")


class Doc:
    def __init__(self, t: str, idx: LawIndex):
        self.t = t
        self.idx = idx
        self.stems, self.starts, self.ends, self.origs = tokenize(t)
        self._mentions = None

    def _pick(self, ids, i0, i1):
        if len(ids) == 1:
            return next(iter(ids))
        st = self.stems
        for k in range(max(0, i0 - 2), min(len(st) - 2, i1 + 8)):
            if (st[k].isdigit() and st[k + 1].isdigit() and st[k + 2].isdigit()
                    and len(st[k + 2]) == 4 and len(st[k]) <= 2 and len(st[k + 1]) <= 2):
                key = (st[k].zfill(2), st[k + 1].zfill(2), st[k + 2])
                cand = self.idx.date.get(key)
                if cand:
                    inter = ids & cand
                    if inter:
                        return min(inter)
        return min(ids)

    def find_after(self, pos: int) -> Optional[int]:
        st, og = self.stems, self.origs
        n = len(st)
        i = bisect_left(self.starts, pos)
        if i >= n:
            return None
        idx = self.idx
        prev_end = pos
        prev_digit = False
        num_hit = None
        for k in range(i, min(n, i + 14)):
            gap = self.t[prev_end:self.starts[k]]
            if len(gap) > 12:
                break
            if _BAD_GAP.search(gap) and not (
                    (prev_digit and st[k].isdigit() and gap.strip() == ".")
                    or (k > i and st[k - 1] in ("г", "рф") and gap.strip() == ".")):
                break
            r = idx.match(idx.full, st, og, k)
            if r is not None:
                return self._pick(r[1], k, r[0])
            r = idx.match(idx.title, st, og, k)
            if r is not None and k > i - 1 and (k == i or st[k - 1] in _ALLOWED_SKIP or st[k - 1].isdigit()):
                return self._pick(r[1], k, r[0])

            if st[k] == "№" and k + 2 < n and st[k + 1].isdigit():
                cand = idx.num.get((st[k + 1], st[k + 2]))
                if cand:
                    num_hit = (cand, k)
                    ids = cand
                    return self._pick(ids, k, k + 3)
            if not (st[k] in _ALLOWED_SKIP or st[k].isdigit()):
                break
            prev_end = self.ends[k]
            prev_digit = st[k].isdigit()
        return None

    def mentions(self):
        if self._mentions is None:
            res = []
            st, og, idx = self.stems, self.origs, self.idx
            k = 0
            n = len(st)
            root = idx.full
            while k < n:
                if st[k] in root:
                    r = idx.match(root, st, og, k)
                    if r is not None:
                        res.append((self.starts[k], self.ends[r[0] - 1], self._pick(r[1], k, r[0])))
                        k = r[0]
                        continue
                k += 1
            self._mentions = res
        return self._mentions

    def find_before(self, pos: int) -> Optional[int]:
        best = None
        for s, e, lid in self.mentions():
            if e <= pos:
                best = (e, lid)
            else:
                break
        if best and _GAP_BEFORE.fullmatch(self.t[best[0]:pos]):
            return best[1]

        st = self.stems
        j = bisect_left(self.starts, pos) - 1
        if j >= 2 and _GAP_BEFORE.fullmatch(self.t[self.ends[j]:pos]):
            if st[j - 2] == "№" and st[j - 1].isdigit():
                cand = self.idx.num.get((st[j - 1], st[j]))
                if cand:
                    return self._pick(cand, j - 2, j + 1)
        return None


def extract(text: str, idx: LawIndex) -> List[LawLink]:
    t = text.replace("ё", "е").replace("Ё", "Е").replace("\xa0", " ")
    tl = t.lower()
    groups = find_groups(t, tl)
    if not groups:
        return []
    chains = build_chains(groups, tl)
    doc = Doc(t, idx)
    for c in chains:
        c.law = doc.find_after(c.end)

    for i in range(len(chains) - 2, -1, -1):
        c, nx = chains[i], chains[i + 1]
        if c.law is None and nx.law is not None:
            gap = tl[c.end:nx.start]
            if len(gap) <= 6 and re.fullmatch(r"[\s,;]*(?:и\s+|или\s+)?", gap):
                c.law = nx.law
    out: List[LawLink] = []
    seen = set()
    for c in chains:
        has_art = any(g.level == L_ART for g in c.groups)
        law = c.law
        if law is None and has_art:
            law = doc.find_before(c.start)
        if not has_art and law is None:
            continue
        arts, pts, subs = expand(c)
        for a in arts:
            for p in pts:
                for s in subs:
                    key = (law, a, p, s)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(LawLink(law_id=law, article=a, point_article=p, subpoint_article=s))
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("law_aliases.json", "r", encoding="utf-8") as file:
        codex_aliases = json.load(file)
    app.state.codex_aliases = codex_aliases
    app.state.law_index = LawIndex(codex_aliases)
    print("🚀 Сервис запускается...")
    yield
    print("🛑 Сервис завершается...")


def get_codex_aliases(request: Request) -> Dict:
    return request.app.state.codex_aliases


app = FastAPI(
    title="Law Links Service",
    description="Cервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/detect")
async def get_law_links(
    data: TextRequest,
    request: Request,
    codex_aliases: Dict = Depends(get_codex_aliases),
) -> LinksResponse:
    """
    Принимает текст и возвращает список юридических ссылок
    """
    return LinksResponse(links=extract(data.text, request.app.state.law_index))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8978)
