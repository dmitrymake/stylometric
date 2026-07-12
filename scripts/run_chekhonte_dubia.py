"""Запустить атрибуционную панель Chekhonte Dubia и записать JSON-отчёт."""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys
import tempfile
from collections import Counter

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
from stylo.lang import function_words  # noqa: E402
from _gate_metrics import work_balanced_centroid  # noqa: E402


CASE = ROOT / "input_cases" / "chekhonte_dubia"
OUT = ROOT / "docs" / "cases" / "chekhonte_dubia.json"

CANDIDATES = {
    "chehov": ROOT / "input_cases" / "chekhonte" / "cand_chehov",
    "lejkin": ROOT / "input_cases" / "chekhonte" / "cand_lejkin",
    "bilibin": ROOT / "input_cases" / "chekhonte" / "cand_bilibin",
    "lazarev_gruzinsky": CASE / "cand_lazarev_gruzinsky",
    "alexander_chekhov": CASE / "cand_alexander_chekhov",
}

PERIOD_CANDIDATES = {
    "chehov_1880_84": ROOT / "input_cases" / "chekhonte" / "cand_chehov",
    "chehov_1884_85": ROOT / "input_cases" / "chekhonte" / "mystery_heldout_chehov_1884_85.txt",
    "lejkin": ROOT / "input_cases" / "chekhonte" / "cand_lejkin",
    "bilibin": ROOT / "input_cases" / "chekhonte" / "cand_bilibin",
    "lazarev_gruzinsky": CASE / "cand_lazarev_gruzinsky",
    "alexander_chekhov": CASE / "cand_alexander_chekhov",
}

BUDILNIK_CANDIDATES = {
    "chehov_budilnik_1884_1885": CASE / "cand_chehov_budilnik_1884_1885",
    "lejkin": ROOT / "input_cases" / "chekhonte" / "cand_lejkin",
    "bilibin": ROOT / "input_cases" / "chekhonte" / "cand_bilibin",
    "lazarev_gruzinsky": CASE / "cand_lazarev_gruzinsky",
    "alexander_chekhov": CASE / "cand_alexander_chekhov",
}

CLEAN_NEIGHBOR_CANDIDATES = {
    "chehov_budilnik_1884_1885": CASE / "cand_chehov_budilnik_1884_1885",
    "lejkin_clean": CASE / "cand_lejkin_clean",
    "bilibin_clean": CASE / "cand_bilibin_clean",
    "lazarev_gruzinsky": CASE / "cand_lazarev_gruzinsky",
    "alexander_chekhov": CASE / "cand_alexander_chekhov",
}

WORD = r"[а-яёА-ЯЁ]+"


SEGMENT_PATTERNS = {
    "sredi_milykh_moskvichey": {
        "file": "15_среди_милых_москвичей.txt",
        "segments": [
            ("1885-05-24", r"(?s)< 24 МАЯ 1885 Г\. >(.*?)(?=\nФОРМУЛЯРНЫЙ СПИСОК)"),
            ("1885-06-14", r"(?s)ФОРМУЛЯРНЫЙ СПИСОК ПЕТЕРБУРГСКИХ ДАМ(.*?)(?=\n< 20 ИЮНЯ)"),
            ("1885-06-20", r"(?s)< 20 ИЮНЯ 1885 г\. >(.*?)(?=\n< 8 АВГУСТА)"),
            ("1885-08-08", r"(?s)< 8 АВГУСТА 1885 г\. >(.*?)(?=\n< 29 АВГУСТА)"),
            ("1885-08-29", r"(?s)< 29 АВГУСТА 1885 г\. >(.*?)(?=\nСноски\n|$)"),
        ],
    },
    "korrespondentsii": {
        "file": "08_корреспонденции.txt",
        "segments": [
            ("glukhov", r"(?s)^Глухов\.(.*?)(?=\nТегеран\.)"),
            ("tehran", r"(?s)Тегеран\.(.*?)(?=\nСызрань\.)"),
            ("syzran", r"(?s)Сызрань\.(.*?)(?=\nПетербург\.)"),
            ("petersburg", r"(?s)Петербург\.(.*?)(?=\nБеседа нашего собственного корреспондента)"),
            ("meshchersky_interview", r"(?s)Беседа нашего собственного корреспондента(.*)$"),
        ],
    },
}


PHILOLOGY_ALIGNMENT = [
    {
        "file": "texts/08_корреспонденции.txt",
        "pss_context": (
            "«Мирской толк»/«Винт», 1883, № 4; подпись Гайка № 0,006. "
            "ПСС трактует подпись как вариант чеховской Гайка № 6 "
            "и указывает на повторяющиеся чеховские мотивы."
        ),
        "model_context": "Достаточно длинный потекстовый диагностический признак указывает на chehov.",
        "priority": "confirm_chekhov",
    },
    {
        "file": "texts/09_ревнивый_муж_и_храбрый_любовник.txt",
        "pss_context": (
            "«Мирской толк»/«Винт», 1883, № 6; подпись Гайка № 101010101. "
            "ПСС опирается на поэтику, жанр и параллели с таганрогским театром."
        ),
        "model_context": "Достаточно длинный потекстовый диагностический признак указывает на chehov.",
        "priority": "confirm_chekhov",
    },
    {
        "file": "texts/10_мачеха.txt",
        "pss_context": (
            "«Мирской толк»/«Винт», 1883, № 7; без подписи. ПСС называет её наиболее "
            "«чеховской» из прозаических текстов, атрибутированных в «Мирском толке»."
        ),
        "model_context": "Достаточно длинный потекстовый диагностический признак указывает на chehov, но с малым отрывом.",
        "priority": "confirm_chekhov",
    },
    {
        "file": "texts/12_моя_семья.txt",
        "pss_context": (
            "«Зритель», 1883, № 19; в оглавлении подпись С. Б. Ч. ПСС читает "
            "её как обращённую форму чеховского криптонима Ч. Б. С. и приводит "
            "формальные параллели."
        ),
        "model_context": "Достаточно длинный потекстовый диагностический признак узко указывает на alexander_chekhov.",
        "priority": "resolve_brother_confound",
    },
    {
        "file": "texts/15_среди_милых_москвичей.txt",
        "pss_context": (
            "«Будильник», 1885, № 20/23/24/34; пять заметок без подписи из "
            "еженедельного обозрения. ПСС обосновывает атрибуцию письмами редактора, "
            "заказывавшего обозрение Чехову, и контекстными параллелями."
        ),
        "model_context": (
            "Объединённый потекстовый признак указывает на bilibin; посегментная "
            "диагностика расходится. Признак, согласованный с «Будильником», разворачивает "
            "объединённый агрегат в сторону Чехова, тогда как 1885-06-14 остаётся за Bilibin, "
            "а 1885-06-20 указывает на чеховские контроли из «Будильника»."
        ),
        "priority": "highest_discrepancy",
    },
]


def read(path: pathlib.Path) -> list[str]:
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    return [file.read_text("utf-8", "ignore") for file in files if file.exists()]


def docs_of(path: pathlib.Path) -> list[str]:
    return [text for _work, text in docs_by_work(path)]


def docs_by_work(path: pathlib.Path):
    out = []
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    for file in files:
        if not file.exists():
            continue
        text = file.read_text("utf-8", "ignore")
        work_id = str(file.resolve())
        words = text.split()
        if len(words) <= 2200:
            out.append((work_id, text))
        else:
            for idx in range(0, len(words), 1500):
                out.append((work_id, " ".join(words[idx : idx + 1500])))
    return out


class WorkVectors(list):
    def __init__(self, vectors, work_ids):
        super().__init__(vectors)
        self.work_ids = tuple(work_ids)


def build_model(mystery_docs: list[str], candidates: dict[str, pathlib.Path] | None = None):
    fw = sorted(function_words("ru"))
    fwi = {word: idx for idx, word in enumerate(fw)}
    corpus = {name: docs_by_work(path) for name, path in (candidates or CANDIDATES).items()}
    everything = [text for docs in corpus.values() for _work, text in docs] + mystery_docs

    grams = Counter()
    for text in everything:
        flat = re.sub(r"\s+", " ", text.lower())
        for idx in range(max(len(flat) - 2, 0)):
            grams[flat[idx : idx + 3]] += 1
    top3 = [gram for gram, _ in grams.most_common(800)]
    t3i = {gram: idx for idx, gram in enumerate(top3)}

    def vec(text: str) -> np.ndarray:
        tokens = re.findall(WORD, text.lower())
        fw_vec = np.zeros(len(fw))
        for token in tokens:
            pos = fwi.get(token)
            if pos is not None:
                fw_vec[pos] += 1
        fw_vec /= len(tokens) or 1

        flat = re.sub(r"\s+", " ", text.lower())
        c3_vec = np.zeros(len(top3))
        for idx in range(max(len(flat) - 2, 0)):
            pos = t3i.get(flat[idx : idx + 3])
            if pos is not None:
                c3_vec[pos] += 1
        c3_vec /= max(len(flat) - 2, 1)
        fw_vec /= np.linalg.norm(fw_vec) + 1e-9
        c3_vec /= np.linalg.norm(c3_vec) + 1e-9
        return np.concatenate([fw_vec, c3_vec])

    docvecs = {
        name: WorkVectors(
            [vec(text) for _work, text in docs],
            [work for work, _text in docs],
        )
        for name, docs in corpus.items()
        if docs
    }
    centroids = {}
    for name, vectors in docvecs.items():
        centroids[name] = work_balanced_centroid(zip(vectors.work_ids, vectors))
    return vec, docvecs, centroids


def similarities(vecs: list[np.ndarray], centroids: dict[str, np.ndarray]) -> list[dict]:
    mystery = np.mean(vecs, axis=0)
    mystery /= np.linalg.norm(mystery) + 1e-9
    rows = [
        {"candidate": name, "cos": round(float(np.dot(mystery, centroid)), 6)}
        for name, centroid in centroids.items()
    ]
    return sorted(rows, key=lambda row: row["cos"], reverse=True)


def loo_accuracy(docvecs: dict[str, list[np.ndarray]]) -> dict:
    names = list(docvecs)
    correct = total = 0
    for name in names:
        works = list(dict.fromkeys(docvecs[name].work_ids))
        if len(works) < 2:
            continue
        for held_work in works:
            test = [
                vector
                for work, vector in zip(docvecs[name].work_ids, docvecs[name])
                if work == held_work
            ]
            centroids = {}
            for other in names:
                train = [
                    (work, vector)
                    for work, vector in zip(docvecs[other].work_ids, docvecs[other])
                    if not (other == name and work == held_work)
                ]
                if not train:
                    continue
                centroids[other] = work_balanced_centroid(train)
            predictions = [
                max(
                    centroids,
                    key=lambda other: float(
                        np.dot(
                            vector / (np.linalg.norm(vector) + 1e-9),
                            centroids[other],
                        )
                    ),
                )
                for vector in test
            ]
            correct += Counter(predictions).most_common(1)[0][0] == name
            total += 1
    return {"correct": correct, "total": total, "accuracy": round(correct / total, 6)}


def bootstrap(vecs: list[np.ndarray], centroids: dict[str, np.ndarray], n_iter: int = 2000) -> dict:
    rng = np.random.default_rng(20260629)
    margins: list[float] = []
    winners: Counter[str] = Counter()
    n = len(vecs)
    arr = np.array(vecs)
    for _ in range(n_iter):
        sample = arr[rng.integers(0, n, size=n)]
        sims = similarities(list(sample), centroids)
        winners[sims[0]["candidate"]] += 1
        margins.append(sims[0]["cos"] - sims[1]["cos"])
    lo, hi = np.quantile(margins, [0.025, 0.975])
    return {
        "n_iter": n_iter,
        "unit": "фрагменты по 1500 слов из агрегированного файла спорного текста",
        "winner_counts": dict(winners.most_common()),
        "top_margin_ci95": [round(float(lo), 6), round(float(hi), 6)],
    }


def aggregate(path: pathlib.Path, candidates: dict[str, pathlib.Path] | None = None) -> dict:
    mystery_docs = docs_of(path)
    vec, docvecs, centroids = build_model(mystery_docs, candidates)
    vecs = [vec(text) for text in mystery_docs]
    sims = similarities(vecs, centroids)
    return {
        "path": str(path.relative_to(ROOT)),
        "candidate_panel": list((candidates or CANDIDATES).keys()),
        "n_chunks": len(mystery_docs),
        "train_centroid_weighting": "equal_work_direction_after_within_work_chunk_mean_l2",
        "similarities": sims,
        "loo": loo_accuracy(docvecs),
        "bootstrap": bootstrap(vecs, centroids),
    }


def per_text(
    candidates: dict[str, pathlib.Path] | None = None,
    min_words: int | None = None,
) -> list[dict]:
    rows = []
    files = sorted((CASE / "texts").glob("*.txt"))
    texts = [file.read_text("utf-8", "ignore") for file in files]
    vec, _, centroids = build_model(texts, candidates)
    for file, text in zip(files, texts):
        words = len(re.findall(r"[А-Яа-яЁёA-Za-z]+", text))
        if min_words is not None and words < min_words:
            continue
        sims = similarities([vec(text)], centroids)
        rows.append(
            {
                "file": str(file.relative_to(CASE)),
                "words": words,
                "top": sims[0]["candidate"],
                "second": sims[1]["candidate"],
                "margin": round(sims[0]["cos"] - sims[1]["cos"], 6),
                "similarities": sims,
                "too_short": words < 300,
            }
        )
    return rows


def clean_segment(text: str) -> str:
    text = re.sub(
        r"«БУДИЛЬНИК».*?(Титульный лист|Последняя страница обложки)",
        " ",
        text,
        flags=re.S,
    )
    return re.sub(r"\s+", " ", text).strip()


def segment_diagnostics(candidates: dict[str, pathlib.Path] | None = None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name, spec in SEGMENT_PATTERNS.items():
        text = (CASE / "texts" / spec["file"]).read_text(encoding="utf-8")
        segments: list[tuple[str, str]] = []
        for label, pattern in spec["segments"]:
            match = re.search(pattern, text)
            if match:
                segments.append((label, clean_segment(match.group(1))))
        if not segments:
            out[name] = []
            continue
        vec, _, centroids = build_model([body for _, body in segments], candidates)
        rows = []
        for label, body in segments:
            sims = similarities([vec(body)], centroids)
            rows.append(
                {
                    "segment": label,
                    "words": len(re.findall(WORD, body)),
                    "top": sims[0]["candidate"],
                    "second": sims[1]["candidate"],
                    "margin": round(sims[0]["cos"] - sims[1]["cos"], 6),
                    "similarities": sims,
                    "too_short": len(re.findall(WORD, body)) < 300,
                }
            )
        out[name] = rows
    return out


def control_self_test(
    target_name: str,
    target_path: pathlib.Path,
    candidates: dict[str, pathlib.Path],
) -> dict:
    files = sorted(target_path.glob("*.txt"))
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"{target_name}_selftest_") as tmp:
        tmp_path = pathlib.Path(tmp)
        for file in files:
            train_path = tmp_path / file.stem
            train_path.mkdir()
            for other in files:
                if other != file:
                    shutil.copy(other, train_path / other.name)
            panel = dict(candidates)
            panel[target_name] = train_path
            text = file.read_text("utf-8", "ignore")
            vec, _, centroids = build_model([text], panel)
            sims = similarities([vec(text)], centroids)
            rows.append(
                {
                    "file": str(file.relative_to(ROOT)),
                    "words": len(re.findall(WORD, text)),
                    "top": sims[0]["candidate"],
                    "second": sims[1]["candidate"],
                    "margin": round(sims[0]["cos"] - sims[1]["cos"], 6),
                    "correct": sims[0]["candidate"] == target_name,
                    "too_short": len(re.findall(WORD, text)) < 300,
                }
            )
    eligible_rows = [row for row in rows if not row["too_short"]]
    correct = sum(row["correct"] for row in rows)
    eligible_correct = sum(row["correct"] for row in eligible_rows)
    return {
        "method": "leave-one-control-out; каждый подписанный чеховский контроль удаляется из чеховского centroid и классифицируется относительно оставшейся панели",
        "target": target_name,
        "summary": {
            "correct": correct,
            "total": len(rows),
            "accuracy": round(correct / len(rows), 6) if rows else None,
            "eligible_threshold_words": 300,
            "eligible_correct": eligible_correct,
            "eligible_total": len(eligible_rows),
            "eligible_accuracy": round(eligible_correct / len(eligible_rows), 6)
            if eligible_rows
            else None,
            "top_counts": dict(Counter(row["top"] for row in rows).most_common()),
            "eligible_top_counts": dict(Counter(row["top"] for row in eligible_rows).most_common()),
        },
        "rows": rows,
    }


def main() -> None:
    manifest = yaml.safe_load((CASE / "manifest.yaml").read_text(encoding="utf-8"))
    prose = aggregate(CASE / "mystery_prose.txt")
    all_items = aggregate(CASE / "mystery_all.txt")
    item_rows = per_text()
    all_counts = Counter(row["top"] for row in item_rows)
    eligible_rows = [row for row in item_rows if not row["too_short"]]
    eligible_counts = Counter(row["top"] for row in eligible_rows)
    report = {
        "case": "chekhonte_dubia",
        "title": "Атрибуционная проба раздела Dubia (ПСС Чехова, т. 18)",
        "data_status": (
            "Загружено из ФЭБ, ПСС т. 18, раздел Dubia: 27 текстов, 25 прозаических; "
            "сырые тексты лежат в игнорируемой папке input_cases/chekhonte_dubia."
        ),
        "candidate_panel": list(CANDIDATES),
        "source_tree": manifest["tree_url"],
        "aggregate_prose": prose,
        "aggregate_all": all_items,
        "per_text_summary": {
            "all_top_counts": dict(all_counts.most_common()),
            "eligible_threshold_words": 300,
            "eligible_n": len(eligible_rows),
            "eligible_top_counts": dict(eligible_counts.most_common()),
        },
        "per_text": item_rows,
        "segment_diagnostics": segment_diagnostics(),
        "period_panel_diagnostic": {
            "status": "diagnostic_only",
            "reason": (
                "Разделение Чехова на 1880-1884 и отложенный 1884-1885 "
                "проверяет смешение периода/регистра, но положительный контроль "
                "проседает, поскольку соседние чеховские панели становится трудно разделить."
            ),
            "aggregate_prose": aggregate(CASE / "mystery_prose.txt", PERIOD_CANDIDATES),
            "segment_diagnostics": segment_diagnostics(PERIOD_CANDIDATES),
        },
        "budilnik_control_diagnostic": {
            "status": "diagnostic_only",
            "control_set": "input_cases/chekhonte_dubia/cand_chehov_budilnik_1884_1885",
            "reason": (
                "Здесь используются двенадцать подписанных или принятых чеховских текстов "
                "из «Будильника» 1884-1885 как контроль того же журнала. Набор содержит "
                "всего около 7 тыс. слов, а самопроверка leave-one-control-out слаба, поэтому "
                "панель полезна лишь как проба на рассогласование регистра."
            ),
            "aggregate_prose": aggregate(CASE / "mystery_prose.txt", BUDILNIK_CANDIDATES),
            "eligible_per_text": per_text(BUDILNIK_CANDIDATES, min_words=300),
            "segment_diagnostics": segment_diagnostics(BUDILNIK_CANDIDATES),
            "control_self_test": control_self_test(
                "chehov_budilnik_1884_1885",
                CASE / "cand_chehov_budilnik_1884_1885",
                {
                    "lejkin": ROOT / "input_cases" / "chekhonte" / "cand_lejkin",
                    "bilibin": ROOT / "input_cases" / "chekhonte" / "cand_bilibin",
                    "lazarev_gruzinsky": CASE / "cand_lazarev_gruzinsky",
                    "alexander_chekhov": CASE / "cand_alexander_chekhov",
                },
            ),
        },
        "clean_neighbor_control_diagnostic": {
            "status": "diagnostic_only",
            "control_sets": {
                "bilibin_clean": "input_cases/chekhonte_dubia/cand_bilibin_clean",
                "lejkin_clean": "input_cases/chekhonte_dubia/cand_lejkin_clean",
            },
            "reason": (
                "Широкие выгрузки кандидатов Билибина/Лейкина с az.lib.ru содержат "
                "оглавления и комментарии. Эта панель использует разделённые тела "
                "произведений для этих соседей, но положительный контроль "
                "слаб, а отрыв агрегата Чехов–Bilibin близок к нулю."
            ),
            "aggregate_prose": aggregate(CASE / "mystery_prose.txt", CLEAN_NEIGHBOR_CANDIDATES),
            "eligible_per_text": per_text(CLEAN_NEIGHBOR_CANDIDATES, min_words=300),
            "segment_diagnostics": segment_diagnostics(CLEAN_NEIGHBOR_CANDIDATES),
        },
        "philology_alignment": PHILOLOGY_ALIGNMENT,
        "verdict": "неоднозначно",
        "confidence": "низкая",
        "key_findings": (
            "На расширенной панели с согласованным регистром объединённая проза "
            "располагается между Александром Чеховым и Bilibin: точечная оценка ставит "
            "Александра Чехова на первое место с отрывом ~0,0002 над Bilibin, тогда как "
            "bootstrap-подсчёты победителей расходятся. Это не является обоснованным "
            "атрибуционным утверждением. Потекстовая диагностика также неоднородна: среди "
            "текстов >=300 слов Чехов лидирует в 3/5, Александр Чехов в 1/5 и Bilibin в 1/5. "
            "Добавление чеховских контролей из того же журнала «Будильник» разворачивает "
            "объединённый агрегат в сторону Чехова, но эти контроли не проходят собственную "
            "самопроверку leave-one-control-out. С очищенными соседями Билибиным и Лейкиным "
            "объединённый агрегат фактически уравнивается между чеховским «Будильником» и "
            "Bilibin. Результат — полезная зацепка для поиска, а не решение вопроса об авторстве."
        ),
        "caveats": [
            "Большинство отдельных текстов Dubia очень коротки; потекстовые победители носят лишь диагностический характер.",
            "Агрегат смешивает театральные заметки, рецензии, сценки и крошечные извещения.",
            "И Bilibin, и Александр Чехов — правдоподобные соседи по регистру; текущая панель не разделяет их с полезным отрывом.",
            "Диагностическая панель с отложенным Чеховым 1884-1885 снижает агрегатный сигнал Bilibin, но имеет слабые положительные контроли, поэтому не может использоваться как вердикт.",
            "Диагностическая панель с двенадцатью чеховскими контролями из «Будильника» 1884-1885 разворачивает объединённый агрегат Dubia в сторону Чехова, но её самопроверка leave-one-control-out распознаёт лишь меньшинство собственных подписанных контролей.",
            "Разделение антологических выгрузок Билибина и Лейкина устраняет очевидное редакторское загрязнение, но не делает панель соседей готовой к атрибуции.",
            "Публикуемое утверждение требует кластеризации текстов по жанру/источнику и большего числа подписанных кандидатов из тех же периодических изданий.",
        ],
        "next_tasks": [
            "Расширить контрольный набор «Будильника» свыше 7 тыс. слов и добавить контроли Билибина/Лейкина той же формы из подписей и рецензионных заметок «Будильника»/«Осколков».",
            "Построить панель для разрешения «братского» смешения по «Зрителю»/«Мирскому толку»: Антон Чехов против Александра Чехова на подписанных текстах 1882-1886.",
            "Отделить «Корреспонденции» от интервью с Мещерским; заметки короче 300 слов использовать только как объединённое серийное свидетельство.",
            "Добавить контроли Билибина и Лейкина той же формы для крошечных заметок, подписей и фельетонов на основе рисунков.",
            "Установить эмпирический порог атрибуции по подписанным контролям объёмом 300-700 слов, прежде чем выдвигать какой-либо потекстовый вердикт.",
        ],
        "analysis_commands": [
            "python scripts/fetch_chekhonte_dubia.py",
            "python scripts/build_chekhonte_budilnik_controls.py",
            "python scripts/build_chekhonte_neighbor_controls.py",
            "PYTHONPATH=src python log/attribute_case.py --lang ru --mystery input_cases/chekhonte_dubia/mystery_prose.txt --cand chehov=input_cases/chekhonte/cand_chehov --cand lejkin=input_cases/chekhonte/cand_lejkin --cand bilibin=input_cases/chekhonte/cand_bilibin --cand lazarev_gruzinsky=input_cases/chekhonte_dubia/cand_lazarev_gruzinsky --cand alexander_chekhov=input_cases/chekhonte_dubia/cand_alexander_chekhov",
            "PYTHONPATH=src python scripts/run_chekhonte_dubia.py",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dumps_strict(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"записан {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
