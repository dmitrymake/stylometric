import joblib
import numpy as np
import pandas as pd
import logging
import sys
from collections import Counter, defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ВАЛИДАЦИЯ ПО ЦЕЛЫМ КНИГАМ (BOOK-LEVEL LOBO)            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        data = joblib.load("data/train_vectors.pkl")
        X = data["X_transformed"]
        y = data["labels"]
        raw_groups = data["groups"] # формат "author/book_00001"
        authors = data["authors"]

        delta_scaler = joblib.load("data/scaler_delta.pkl")
        
    except FileNotFoundError:
        print("❌ Ошибка: Нет файлов данных. Запустите сначала train.py")
        sys.exit(1)

    # Восстанавливаем настоящие имена книг из групп вида "author/book_00005"
    real_books = []
    for g in raw_groups:
        # суффикс _xxxxx добавляет split.py; для коротких имён без него берём как есть
        if "_" in g and g[-5:].isdigit():
            book_name = g.rsplit('_', 1)[0]
        else:
            book_name = g
        real_books.append(book_name)
    
    real_books = np.array(real_books)
    unique_books = sorted(list(set(real_books)))
    
    print(f"Всего чанков (кусочков): {len(y)}")
    print(f"Всего реальных книг:     {len(unique_books)}")
    print("-" * 60)

    # Карта: книга -> список индексов её чанков
    book_to_indices = defaultdict(list)
    for idx, book_name in enumerate(real_books):
        book_to_indices[book_name].append(idx)

    # Цикл LOBO (leave-one-book-out)
    results = []

    total_books = len(unique_books)

    for i, test_book in enumerate(unique_books):
        test_indices = book_to_indices[test_book]

        # если у автора всего одна книга, LOBO невозможен (не на чем учиться)
        test_auth_idx = y[test_indices[0]]

        mask_test = np.zeros(X.shape[0], dtype=bool)
        mask_test[test_indices] = True
        mask_train = ~mask_test
        
        y_train = y[mask_train]

        # автора нет в y_train => у него была всего 1 книга, пропускаем
        if test_auth_idx not in y_train:
            continue

        X_train = X[mask_train]
        X_test = X[mask_test]

        # Logistic Regression, soft voting по чанкам книги
        clf = make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver='lbfgs')
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)

        # усреднение вероятностей по чанкам гасит шум на длинных текстах
        avg_probs = probs.mean(axis=0)
        pred_idx = np.argmax(avg_probs)
        confidence = avg_probs[pred_idx]
        
        is_correct = (pred_idx == test_auth_idx)
        
        results.append({
            "book": test_book,
            "true_author": authors[test_auth_idx],
            "pred_author": authors[pred_idx],
            "correct": is_correct,
            "confidence": confidence,
            "n_chunks": len(test_indices)
        })

        if (i+1) % 5 == 0 or (i+1) == total_books:
            print(f"Проверено {i+1}/{total_books} книг...")

    # Итоговый отчёт
    df = pd.DataFrame(results)
    accuracy = df["correct"].mean()
    
    print("\n" + "="*60)
    print(f"ИТОГОВАЯ ТОЧНОСТЬ (ПО КНИГАМ): {accuracy:.2%}")
    print("="*60)

    errors = df[~df["correct"]]
    if not errors.empty:
        print("\n❌ ОШИБКИ КЛАССИФИКАЦИИ:")
        print(f"{'КНИГА':<40} | {'РЕАЛЬНЫЙ':<15} -> {'ПРЕДСКАЗАННЫЙ':<15} | {'CONF':<6}")
        print("-" * 85)
        for _, row in errors.iterrows():
            book_short = row['book'].split('/')[-1]
            print(f"{book_short:<40} | {row['true_author']:<15} -> {row['pred_author']:<15} | {row['confidence']:.2f}")
    else:
        print("\n🎉 Идеально! Ошибок нет.")

    with open("docs/book_validation_report.txt", "w") as f:
        f.write(f"Book Level Accuracy: {accuracy:.2%}\n\nErrors:\n")
        if not errors.empty:
            for _, row in errors.iterrows():
                 f.write(f"{row['book']} ({row['true_author']}) -> {row['pred_author']}\n")

if __name__ == "__main__":
    main()
