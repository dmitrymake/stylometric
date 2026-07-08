"""stylo — воспроизводимый каркас атрибуции авторства русской прозы.

Единый источник истины параметров — configs/default.yaml (см. stylo.config).
Главные подсистемы:
  - stylo.nlp        : загрузка spaCy + дисковый кеш DocBin (ускоряет LOBO в десятки раз)
  - stylo.features   : каталог фич (FeatureBlock + registry)
  - stylo.vectorizer : сборка вектора из включённых блоков
  - stylo.models     : LR (+калибровка) и НАСТОЯЩАЯ Burrows's Delta
  - stylo.eval       : честный leakage-free LOBO, метрики, CI, значимость, sweep
  - stylo.corpus_tools : валидация и расширение корпуса
"""

__version__ = "0.4.0"
