"""Imposters method (Koppel-Winter, PAN) — жанро-контролируемая верификация авторства.

Вопрос: «текст D написан автором A?» при наборе ИМПОСТОРОВ (других авторов) В ТОМ ЖЕ ЖАНРЕ, что
эталон A. Для многих случайных подпространств признаков смотрим: ближе ли D к A, чем к любому
импостору. score = доля подпространств, где D ближайший к A. Высоко → верифицирован как A.

Жанро-контроль: для дневника D сравниваем с ПИСЬМАМИ A против ПИСЕМ импосторов — так регистр у всех
кандидатов одинаков, и сигнал = автор, а не жанр. Это и есть способ обойти кросс-регистровый конфаунд.
"""
from __future__ import annotations
import sys, pathlib, re, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter
from stylo.lang import function_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
RNG = random.Random(7)
FW = sorted(function_words("ru")); FWI = {w: i for i, w in enumerate(FW)}
WORD = re.compile(r"[а-яёА-ЯЁ]+")
K_ITERS = 200; FRAC = 0.5


def prof(text):
    w = WORD.findall(text.lower()); s = re.sub(r"\s+", " ", text.lower())
    fw = np.zeros(len(FW))
    for t in w:
        j = FWI.get(t)
        if j is not None: fw[j] += 1
    fw /= (len(w) + 1e-9)
    c3 = Counter(s[i:i+3] for i in range(len(s)-2))
    return fw, c3


def load(path):
    p = ROOT / path
    files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    return " ".join(f.read_text("utf-8", "ignore") for f in files if f.exists())


def imposters(D_text, A_text, imp_texts):
    """score = доля подпространств, где D ближе к A, чем к любому импостору."""
    cc = Counter()
    for t in [D_text, A_text] + imp_texts:
        s = re.sub(r"\s+", " ", t.lower())[:80000]
        for i in range(len(s) - 2):
            cc[s[i:i+3]] += 1
    top3 = [g for g, _ in cc.most_common(400)]
    ti = {g: i for i, g in enumerate(top3)}
    def vec(text):
        fw, _ = prof(text); s = re.sub(r"\s+", " ", text.lower()); c3 = np.zeros(len(top3))
        for i in range(len(s)-2):
            j = ti.get(s[i:i+3])
            if j is not None: c3[j] += 1
        c3 /= (sum(c3) + 1e-9)
        return np.concatenate([fw / (np.linalg.norm(fw)+1e-9), c3 / (np.linalg.norm(c3)+1e-9)])
    D = vec(D_text); A = vec(A_text); imps = [vec(t) for t in imp_texts]
    n = len(D); wins = 0
    for _ in range(K_ITERS):
        idx = RNG.sample(range(n), int(n * FRAC))
        d, a = D[idx], A[idx]
        ca = float(np.dot(d, a) / (np.linalg.norm(d)*np.linalg.norm(a)+1e-9))
        best_imp = max(float(np.dot(d, im[idx])/(np.linalg.norm(d)*np.linalg.norm(im[idx])+1e-9)) for im in imps)
        wins += ca > best_imp
    return wins / K_ITERS


def main():
    # эталоны-письма (для жанро-контроля дневников)
    L = {"nikolas2":"input_personal/nikolas2_letters","tolstoy":"input_personal/tolstoy_letters",
         "pushkin":"input_personal/pushkin_letters","chehov":"input_personal/chehov_letters",
         "dostoevsky":"input_personal/dostoevsky_letters","bunin":"input_personal/bunin_letters",
         "alexander3":"input_personal/alexander3_letters","pobedonostsev":"input_personal/pobedonostsev_letters"}
    Lt = {k: load(v) for k, v in L.items() if (ROOT/v).exists()}
    D = {"nikolas2":"input_personal/nikolas2_diary","tolstoy":"input_personal/tolstoy_diary",
         "pushkin":"input_personal/pushkin_diary","chehov":"input_personal/chehov_diary",
         "alexander3":"input_personal/alexander3_diary"}
    Dt = {k: load(v) for k, v in D.items() if (ROOT/v).exists()}

    print("=== ЖАНРО-КОНТРОЛЬ: верифицируем ДНЕВНИК автора против его ПИСЕМ (импосторы=письма других) ===")
    print("   (score = P(дневник ближе к СВОИМ письмам, чем к чужим). Высоко=верифицирован как он сам)")
    for a in Dt:
        if a not in Lt: continue
        imps = [Lt[b] for b in Lt if b != a]
        s = imposters(Dt[a], Lt[a], imps)
        tag = "  ← НИКОЛАЙ" if a == "nikolas2" else ""
        print(f"   {a:12} score={s:.3f}{tag}")

    print("\n=== ФАЛЬШИВКА Вырубовой: верифицируем фальшивый дневник против НАСТОЯЩИХ мемуаров Вырубовой ===")
    vy = ROOT/"input_cases/vyrubova"
    if (vy/"vyrubova_diary_fake.txt").exists() and (vy/"vyrubova_memoir_real.txt").exists():
        fake = (vy/"vyrubova_diary_fake.txt").read_text("utf-8","ignore")
        real = (vy/"vyrubova_memoir_real.txt").read_text("utf-8","ignore")
        # импосторы — проза разных авторов (тот же «не-я» фон)
        imps = [load(f"input_clean/{a}") for a in ["chehov","bunin","gorky","dostoevsky","turgenev"]]
        s = imposters(fake, real, imps)
        print(f"   score(фальшивка ближе к настоящей Вырубовой, чем к импосторам) = {s:.3f}")
        print(f"   → {'ВЫСОКО: имитация верифицируется как Вырубова → подделку НЕ поймали (имитация хороша)' if s>0.5 else 'НИЗКО: фальшивка НЕ как Вырубова → поймали'}")


if __name__ == "__main__":
    main()
