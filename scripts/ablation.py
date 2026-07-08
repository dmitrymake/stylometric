"""
Ablation Study: Проверка влияния маскировки (Bleaching) на точность.
Использует независимую загрузку и нарезку (Smart Chunking), идентичную продакшену.
"""
import logging
import numpy as np
import pathlib
import os
import argparse
from joblib import Parallel, delayed

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression

from utils import make_sent_chunks
from scripts.split import process_large_text, CombinedDoc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parse_args():
    p = argparse.ArgumentParser(description="Ablation Study (Bleaching Impact)")
    p.add_argument("--lang", default="ru", help="Language code")
    return p.parse_args()

def load_and_chunk_author(author_dir, idx, nlp):
    """
    Независимая загрузка и нарезка для абляции.
    Гарантирует, что данные идентичны тем, что будут в продакшене (Smart Chunking).
    """
    texts, labels, groups = [], [], []
    author_name = author_dir.name
    
    files = sorted(author_dir.rglob("*.txt"))
    # Лимит файлов для скорости абляции
    files = files[:10] 
    
    for fp in files:
        if not fp.is_file(): continue
        try:
            raw = fp.read_text("utf-8").strip()
            if not raw: continue

            all_sentences = process_large_text(raw, nlp)
            if not all_sentences: continue
            
            dummy_doc = CombinedDoc(all_sentences)
            chunks = make_sent_chunks(dummy_doc, size=500, min_size=200, overlap=0.0)
            
            for i, ch in enumerate(chunks):
                texts.append(ch)
                labels.append(idx)
                groups.append(f"{author_name}/{fp.stem}")
                
        except Exception:
            pass
            
    return texts, labels, groups

def train_and_evaluate(texts, y, groups, authors, use_bleach: bool):
    """
    Векторизация -> LOBO -> Accuracy
    """
    from syntax_features import StyloVectorizer
    from lobo_cv import create_book_groups

    mode_name = "WITH_BLEACH" if use_bleach else "NO_BLEACH"
    logging.info(f"--- Running {mode_name} ---")
    
    # Важно: создаем новый векторизатор, чтобы применить (или нет) bleaching
    vec = StyloVectorizer(
        char_ngram_range=(3, 5),
        max_char_features=3000, # Чуть меньше фичей для скорости теста
        char_min_df=2,
        use_char=True, use_func=True, use_mfw=True, use_syntax=True,
        auto_bleach=use_bleach
    )
    
    logging.info("Fitting vectorizer...")
    X = vec.fit_transform(texts)
    
    # LOBO Validation (Simplified)
    BOOK_GROUPS = create_book_groups(y, groups)
    unique_groups = sorted(BOOK_GROUPS.keys())
    
    correct = 0
    total = 0
    
    # Последовательный цикл без Parallel: внутри уже Parallel и для экономии памяти
    for test_group_id in unique_groups:
        test_indices = BOOK_GROUPS[test_group_id]
        mask_test = np.zeros(X.shape[0], dtype=bool)
        mask_test[test_indices] = True
        
        X_train = X[~mask_test]
        y_train = y[~mask_test]
        X_test = X[mask_test]
        
        test_auth_idx = y[test_indices[0]]
        
        # Если автора нет в трейне
        if test_auth_idx not in y_train:
            continue
            
        clf = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=500, class_weight='balanced'))
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        
        # Голосование по большинству (проще, чем proba для быстрого теста)
        from collections import Counter
        vote = Counter(pred).most_common(1)[0][0]
        
        if vote == test_auth_idx:
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0.0
    logging.info(f"{mode_name} Accuracy: {acc:.3%} ({correct}/{total})")
    return acc

def main():
    args = parse_args()

    os.environ["STYLO_LANG"] = args.lang
    from scripts.nlp import get_stylometry_nlp
    
    logging.info(f"Загрузка и нарезка данных для Ablation ({args.lang})...")
    
    clean_root = pathlib.Path("input_clean")
    if not clean_root.exists():
        logging.error("Нет папки input_clean")
        exit(1)
        
    nlp = get_stylometry_nlp()
    with nlp.select_pipes(enable=["sentencizer"]):
        authors_raw = sorted([d.name for d in clean_root.iterdir() if d.is_dir() and d.name != "unknown"])
        
        all_texts = []
        all_y = []
        all_groups = []

        for idx, auth in enumerate(authors_raw):
            t, l, g = load_and_chunk_author(clean_root / auth, idx, nlp)
            all_texts.extend(t)
            all_y.extend(l)
            all_groups.extend(g)

    all_texts = np.array(all_texts, dtype=object)
    all_y = np.array(all_y, dtype=int)
    all_groups = np.array(all_groups)
    
    cnt = np.bincount(all_y, minlength=len(authors_raw))
    keep = [i for i, c in enumerate(cnt) if c >= 3]
    mask = np.isin(all_y, keep)
    
    final_texts = all_texts[mask]
    final_y = all_y[mask]
    final_groups = all_groups[mask]
    
    logging.info(f"Данные готовы: {len(final_texts)} чанков.")

    acc_bleach = train_and_evaluate(final_texts, final_y, final_groups, authors_raw, use_bleach=True)
    acc_no_bleach = train_and_evaluate(final_texts, final_y, final_groups, authors_raw, use_bleach=False)
    
    print("\n=== ABLATION RESULTS ===")
    print(f"Bleaching ON:  {acc_bleach:.3%}")
    print(f"Bleaching OFF: {acc_no_bleach:.3%}")
    diff = acc_bleach - acc_no_bleach
    
    impact_str = "POSITIVE (Masking helps)" if diff > 0 else "NEGATIVE (Masking hurts)"
    print(f"Impact: {diff:+.3%} -> {impact_str}")

    with open("docs/ablation_report.txt", "w") as f:
        f.write(f"Bleaching ON:  {acc_bleach:.3%}\n")
        f.write(f"Bleaching OFF: {acc_no_bleach:.3%}\n")
        f.write(f"Impact: {diff:+.3%} ({impact_str})\n")

if __name__ == "__main__":
    main()
