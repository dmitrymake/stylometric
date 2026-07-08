import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "modernize_orthography",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "modernize_orthography.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_char_map():
    assert mod.modernize("лѣсъ") == "лес"
    assert mod.modernize("міръ") == "мир"
    assert mod.modernize("Ѳедоръ") == "Федор"
    assert mod.modernize("ѵпостась") == "ипостась"


def test_final_hard_sign_only():
    # финальный ъ удаляется, разделительный внутри слова остаётся
    assert mod.modernize("объемъ") == "объем"
    assert mod.modernize("съѣздъ") == "съезд"
    assert mod.modernize("въ лѣсу") == "в лесу"


def test_endings_not_rewritten():
    # морфологические окончания не трогаем: это словоизменение, не орфография
    assert mod.modernize("стараго") == "стараго"
    assert mod.modernize("новыя") == "новыя"


def test_is_oldorfo_detection():
    old = "Въ тѣ дни, когда мнѣ были новы всѣ впечатлѣнья бытія " * 20
    new = "В те дни, когда мне были новы все впечатленья бытия " * 20
    assert mod.is_oldorfo(old)
    assert not mod.is_oldorfo(new)


def test_modernized_text_is_not_oldorfo():
    old = "Помѣщикъ жилъ въ имѣніи со всѣмъ семействомъ своимъ. " * 20
    assert not mod.is_oldorfo(mod.modernize(old))
