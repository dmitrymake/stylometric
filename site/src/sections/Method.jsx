import { Fragment } from "react";
import { Card, Timeline, WhyBlock } from "@dmitrymake/rk-ui";
import { FEATURES, TOMSK, HEADLINE } from "../data.js";
import { CASES, BENCH_EXT } from "../segdata.js";
import { fmtScore, fmtRange, fmtPct } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";
import Sources from "../components/Sources.jsx";

const TA = CASES.tolstoyAn;
// целевая строка панели ищется по флагу hi, а не по позиции в массиве
const TA_ROW = TA.sil.find((r) => r.hi);
// max шкалы берётся из данных, не литералом
const SIL_MAX = Math.max(...TA.sil.map((r) => r.v));
// диапазон сравнения считаем по остальным авторам панели, без самого А. Н. Толстого
const SIL_REF = TA.sil.filter((r) => !r.hi).map((r) => r.v);
const SIL_REF_LOW = Math.min(...SIL_REF);
const SIL_REF_HIGH = Math.max(...SIL_REF);

// склонение «автор» — чтобы число из данных не ломало грамматику
// (только для именительного счётного: «51 автор», «22 автора», «43 автора», «5 авторов»).
const ruAuthors = (n) => {
  const mod100 = Math.abs(n) % 100;
  const mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "авторов";
  if (mod10 === 1) return "автор";
  if (mod10 >= 2 && mod10 <= 4) return "автора";
  return "авторов";
};

// пересчёт по целым книгам на открытых данных группы из ТУСУР: строка для 50 авторов
// и диапазон масштабов (числа для читаемого примечания — из данных, не литералами).
const TOMSK_50 = TOMSK.headToHead.table.find((r) => r.k === 50);
const TOMSK_KMIN = Math.min(...TOMSK.headToHead.table.map((r) => r.k));
const TOMSK_KMAX = Math.max(...TOMSK.headToHead.table.map((r) => r.k));

// Единый стиль заголовка сворачиваемых блоков «детали».
const SUMMARY_STYLE = { cursor: "pointer", fontFamily: "var(--font-text)", fontSize: "var(--fs-caption)", fontWeight: "var(--fw-semibold)", letterSpacing: "var(--tracking-caption)", textTransform: "uppercase", color: "var(--text-muted)" };

// Признаки перечислены без статусных бейджей. Факультативные блоки (kind: «opt») приглушены через opacity.
const KIND_STYLE = { opt: { dim: true } };

// Русские подписи карточек признаков прямо при рендере: данные приходят с англ. слагами,
// data.js не трогаем — переводим на месте, чтобы в карточках не осталось сырых кодов.
const FEAT_NAME_RU = { dependency: "синтаксические связи" };
const FEAT_NOTE_RU = {
  "off — риск утечки темы": "выключено — риск утечки темы",
  "падеж/время/вид (spaCy morph)": "падеж, время, вид (разбор spaCy)",
  "синтаксический скелет, тематически нейтрален":
    "синтаксический скелет; может быть менее чувствителен к теме в этой проверке",
};
const featName = (n) => FEAT_NAME_RU[n] || n;
const featNote = (n) => FEAT_NOTE_RU[n] || n;

const PROTOCOL = [
  { marker: "01", title: "чистим тексты", body: "Убираем библиотечные пометы и точные повторы, приводим дореформенную орфографию к современной, но сохраняем ритм и пунктуацию.", color: "var(--icon-blue)", state: "done" },
  { marker: "02", title: "собираем семьи текстов", body: "Рассказ, отдельное издание и сборник с тем же рассказом считаются связанными и не расходятся по разные стороны проверки.", color: "var(--icon-blue)", state: "done" },
  { marker: "03", title: "прячем книгу целиком", body: "Каждую книгу по очереди убираем вместе со связанными текстами. Профиль автора строится только по тому, что осталось.", color: "var(--gold)", state: "done" },
  { marker: "04", title: "сравниваем ответы", body: "Смотрим не только на долю попаданий, но и на простой «мешок слов»: сложный метод полезен тогда, когда добавляет что-то сверх выбора слов.", color: "var(--success)", state: "done" },
];

export default function Method() {
  return (
    <section className="section" id="method">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Метод</p>
          <h2>Как проверить модель, не давая ей подсказок</h2>
          <p className="prose lead muted">
            Программа читает книги известных авторов, затем получает одну незнакомую и
            должна назвать имя. Проверяемую книгу прячут целиком — вместе со всеми
            сборниками и изданиями, где повторяется то же содержание. Иначе один и тот же
            текст оказался бы и в обучении, и в проверке, и совпадение говорило бы о
            знакомом содержании, а не о стиле.
          </p>
          <p className="prose muted">
            Первый эксперимент прятал книги по одной, но считал отдельный рассказ и сборник
            разными объектами. Теперь граница проходит не по названию файла, а по самому
            содержанию.
          </p>
        </div>

        {/* Строгая проверка без подсказок */}
        <div className="split reveal" style={{ alignItems: "start" }}>
          <div className="prose">
            {/* Заголовок раздела — вручную свёрстанный <h3>, а не <StageHeader>:
                у того title рендерится как <h1> и ломает иерархию заголовков. */}
            <header style={{ marginBottom: 18 }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", marginBottom: "var(--space-2)" }}>
                <span style={{ width: 10, height: 10, background: "var(--gold)", flex: "0 0 auto" }} />
                <span style={{ fontFamily: "var(--font-text)", fontSize: "var(--fs-caption)", fontWeight: "var(--fw-semibold)", letterSpacing: "var(--tracking-caption)", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  Проверка без подсказок
                </span>
              </div>
              <h3 style={{ fontFamily: "var(--font-display)", fontSize: "var(--fs-display-2)", lineHeight: "var(--lh-display-2)", fontWeight: "var(--fw-bold)", letterSpacing: "var(--tracking-tight)", margin: 0, color: "var(--text)" }}>
                Проверяемый текст невидим до самого ответа
              </h3>
              <p style={{ fontFamily: "var(--font-text)", fontSize: "var(--fs-body)", lineHeight: "var(--lh-body)", color: "var(--text-muted)", margin: "var(--space-2) 0 0", maxWidth: "60ch" }}>
                Словарь, частоты и классификатор строятся только по учебным книгам.
                Проверяемая книга и все тексты с тем же содержанием появляются лишь
                после обучения.
              </p>
            </header>
            <details style={{ marginTop: 4 }}>
              <summary style={SUMMARY_STYLE}>Как читать цифры</summary>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 10, maxWidth: "62ch" }}>
                — Accuracy — доля верно названных книг. Macro-F1 усредняет по авторам
                баланс точности и полноты ответов, поэтому каждый автор получает
                одинаковый вес. Числа первого эксперимента собраны в разделе «Первый
                эксперимент».
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Каждую цифру сравниваем с простыми опорными методами: Burrows Delta
                (отклонения по частым словам), мешок слов и точка отсчёта «всегда самый
                частый автор».
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Проверки идут на разных срезах корпуса: открытая выборка классиков, полная
                проверка по книгам и весь корпус. Состав авторов у них разный, поэтому числа
                из разных срезов напрямую не сравниваются.
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — Интервалы и сравнение с простыми методами будут посчитаны заново после
                очистки корпуса. Старый интервал macro-F1 здесь не показывается: он был
                построен для другой статистической постановки.
              </span>
              <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                — ECE показывает расхождение заявленной уверенности с фактической долей
                попаданий. Чем значение ближе к нулю, тем лучше; здесь —{" "}
                {fmtScore(HEADLINE.ece)}. Поэтому вероятности читаются как порядок версий,
                а не как точная уверенность.
              </span>
              {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
                <span style={{ display: "block", fontSize: 12.5, color: "var(--text-muted)", marginTop: 6, maxWidth: "62ch" }}>
                  — В первом эксперименте длинная книга сильнее влияла на профиль автора.
                  В повторном расчёте каждая книга получает один голос.
                </span>
              )}
            </details>
          </div>
          <Card padding={20} parade>
            <Timeline items={PROTOCOL} />
          </Card>
        </div>

        {/* Единица проверки: книга, а не отрывок */}
        <div className="reveal module">
          <WhyBlock title="Почему единица проверки — книга, а не отрывок">
            Отрывки внутри одной книги похожи между собой: общая тема, лексика, герои.
            Если считать их отдельными примерами, погрешность выходит заниженной, а
            точность выглядит надёжнее, чем она есть. Поэтому книга остаётся единицей
            ответа, а все её отрывки движутся вместе.
          </WhyBlock>
        </div>

        {/* Контроль: держится ли один автор сквозь жанры */}
        <div className="reveal module">
          <h3>Держится ли один автор сквозь жанры</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 16 }}>
            Контроль на бесспорном случае. А. Н. Толстой писал в разных жанрах: {TA.genres}.
            Смотрим разброс между книгами одного автора на синтаксических связях (кто с кем
            связан в предложении) — чем ниже, тем ближе книги друг к другу:
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              {TA.sil.slice().sort((a, b) => a.v - b.v).map((r) => (
                <div key={r.a} style={{ display: "grid", gridTemplateColumns: "14ch 1fr 5ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                  <span style={{ fontSize: 12.5, color: r.hi ? "var(--text)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400 }}>{r.a}</span>
                  <MeterBar value={r.v} max={SIL_MAX} accent={r.hi ? "var(--icon-blue)" : "var(--text-muted)"} />
                  <span className="mono" style={{ fontSize: 11, color: r.hi ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gap: 10, alignContent: "start" }}>
              <p className="verdict" style={{ margin: 0 }}>
                А. Н. Толстой ({fmtScore(TA_ROW.v, 3)}) остаётся в том же узком диапазоне,
                что и авторы с бесспорным единственным авторством{" "}
                ({fmtRange(SIL_REF_LOW, SIL_REF_HIGH, (v) => fmtScore(v, 3))}). Смена жанра
                сама по себе не разводит его книги на разные профили.
              </p>
              <p className="note" style={{ margin: 0 }}>
                <strong style={{ color: "var(--text)" }}>{TA.nSelf} из {TA.nBooks}</strong> его
                книг ближе всего к нему самому, и ни одна не относится к его однофамильцу
                Льву Толстому.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Это одно наблюдение на одном признаке. Оно показывает, что жанр не разваливает
                профиль в этой проверке, но не измеряет, насколько вклад темы отделён в
                остальных расчётах.
              </p>
            </div>
          </div>
        </div>

        {/* каталог признаков — справочник */}
        <div className="reveal module">
          <h3>Из чего собран профиль автора</h3>
          <p className="prose muted" style={{ marginBottom: 22, maxWidth: "78ch" }}>
            Часть признаков ближе к поверхности текста: цепочки букв, частые слова, повторяющиеся
            обороты. Другие описывают устройство фразы: служебные слова, синтаксические связи,
            пунктуацию. Каждый блок проверяется отдельно — правдоподобная идея признака не
            считается результатом, пока не показала вклад в общей оценке. Сравнение групп
            признаков между собой — в разделе «Первый эксперимент».
          </p>
          <div className="grid cols-3">
            {FEATURES.map((f) => {
              const k = KIND_STYLE[f.kind] || {};
              return (
                <Card key={f.id} padding={18} style={{ opacity: k.dim ? 0.66 : 1 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ fontFamily: "var(--font-display)", fontSize: "1.05rem", color: "var(--text)" }}>{featName(f.name)}</span>
                  </div>
                  <p className="muted mono" style={{ margin: 0, fontSize: 12.5 }}>{featNote(f.note)}</p>
                </Card>
              );
            })}
          </div>
          <details style={{ marginTop: 14 }}>
            <summary style={SUMMARY_STYLE}>Перевод сокращений</summary>
            <p className="muted" style={{ fontSize: 12.5, marginTop: 10, maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>n-граммы</strong> — цепочки из нескольких
              подряд идущих букв или слов; <strong style={{ color: "var(--text)" }}>MFW-300</strong> — 300 самых частых
              слов; <strong style={{ color: "var(--text)" }}>POS</strong> — часть речи; <strong style={{ color: "var(--text)" }}>TTR</strong> —
              доля неповторяющихся слов; <strong style={{ color: "var(--text)" }}>Hapax</strong> — слова, встреченные ровно
              один раз; <strong style={{ color: "var(--text)" }}>Yule</strong> — мера богатства словаря;{" "}
              <strong style={{ color: "var(--text)" }}>topic-bleaching</strong> — оставляем скелет из частей речи,
              чтобы уменьшить влияние темы; <strong style={{ color: "var(--text)" }}>синтаксические связи</strong> —
              кто с кем в предложении связан и как глубоко ветвится дерево разбора; <strong style={{ color: "var(--text)" }}>морфология</strong> —
              грамматические пометы слов (падеж, время, вид), взятые разбором spaCy (программой грамматического разбора).
            </p>
          </details>
        </div>

        {/* технические сверки — факты и источники, свёрнуты */}
        <div className="reveal module">
          <h3>Технические сверки</h3>
          <p className="prose muted" style={{ fontSize: 13, borderLeft: "2px solid var(--gold)", paddingLeft: 14, maxWidth: "74ch", marginBottom: 14 }}>
            Что можно сверить с внешними данными: два чужих набора текстов, открытый код
            другой группы и команды для повторного прогона.
          </p>

          <details style={{ marginBottom: 10 }}>
            <summary style={SUMMARY_STYLE}>Чужие наборы данных (CCAT50, Proza.ru)</summary>
            <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 8px", maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>CCAT50</strong> — общепринятый англоязычный
              набор (Reuters, 50 авторов). Равновесный ансамбль даёт{" "}
              {fmtScore(BENCH_EXT.ccat50Ensemble, 3)} при одном фиксированном делении данных.
              Опубликованный ориентир на буквенных n-граммах — {fmtScore(BENCH_EXT.ccat50Valla.ngramA, 3)},
              вариант на BERT — {fmtScore(BENCH_EXT.ccat50Valla.bertA, 3)}. Приведённый в
              обзоре результат {fmtScore(BENCH_EXT.ccat50Valla.record, 3)} получен при другом
              способе деления данных и с этим расчётом напрямую не сравнивается.
            </p>
            <p className="muted" style={{ fontSize: 12.5, margin: "0 0 8px", maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>Proza.ru</strong> — внешний русский набор
              (50 авторов), одно деление. Выше всех — один классификатор по цепочкам букв
              ({fmtScore(BENCH_EXT.prozaLeader, 3)}); равновесное усреднение всех групп ниже
              ({fmtScore(BENCH_EXT.prozaEqualEnsemble, 3)}); базовый ruBERT-tiny2 без дообучения —{" "}
              {fmtScore(BENCH_EXT.prozaNeuro, 3)}. Это один готовый вариант нейросетевой модели:
              дообученные и профильные модели для атрибуции авторства здесь не сравнивались,
              и причина низкого числа этим прогоном не установлена.
            </p>
            <p className="muted" style={{ fontSize: 12.5, margin: 0, maxWidth: "80ch" }}>
              Взвешивание по надёжности (веса групп пропорциональны их точности на отложенной
              части обучения) поднимает ансамбль до {fmtScore(BENCH_EXT.prozaEnsemble, 3)}. Его
              настройка выбрана по лучшему результату из небольшого перебора на этом же тесте,
              поэтому перевес +{fmtScore(BENCH_EXT.prozaEnsemble - BENCH_EXT.prozaLeader, 3)} над
              лидером настроен под тест и лежит в пределах шума.
            </p>
          </details>

          <details style={{ marginBottom: 10 }}>
            <summary style={SUMMARY_STYLE}>Сверка протокола с открытым кодом группы из ТУСУР</summary>
            <div className="split" style={{ alignItems: "start", marginTop: 12 }}>
              <div className="note" style={{ fontSize: 13 }}>
                <p style={{ margin: 0 }}>
                  Опубликовано {fmtPct(TOMSK.theirAcc, 1)} на {TOMSK_50.k} {ruAuthors(TOMSK_50.k)}.
                  В их открытом демо-коде отрывки одной книги попадают и в обучение, и в
                  проверку — деления по книге нет. На тех же данных и признаках, но с делением
                  по книгам, точность на {TOMSK_50.k} авторах — около {fmtPct(TOMSK_50.grouped)}{" "}
                  против {fmtPct(TOMSK_50.rand)} без деления. Разрыв того же порядка держится на
                  всех масштабах — от {TOMSK_KMIN} до {TOMSK_KMAX} авторов. Это сверка на открытом
                  демо-коде, а не пересчёт их полного корпуса.
                </p>
                {/* на узком экране таблица прокручивается внутри своей рамки, а не режется */}
                <div style={{ marginTop: 12, overflowX: "auto" }}>
                <div style={{ display: "grid", gridTemplateColumns: "5ch 1fr 1fr 6.5ch", gap: "4px 10px", fontSize: 12, alignItems: "center", minWidth: "36ch" }}>
                  <span className="mono muted">авт.</span>
                  <span className="mono muted">их протокол</span>
                  <span className="mono muted">по книге</span>
                  <span className="mono muted" style={{ textAlign: "right" }}>разрыв</span>
                  {TOMSK.headToHead.table.map((r) => (
                    <Fragment key={r.k}>
                      <span className="mono" style={{ color: "var(--text)" }}>{r.k}</span>
                      <span className="mono muted">{fmtScore(r.rand, 3)}</span>
                      <span className="mono" style={{ color: "var(--text)" }}>{fmtScore(r.grouped, 3)}</span>
                      <span className="mono" style={{ color: "var(--gold)", textAlign: "right" }}>+{r.prem}</span>
                    </Fragment>
                  ))}
                </div>
                </div>
              </div>
              <Sources
                label="Источник"
                items={[
                  { cite: `${TOMSK.ref.cite} · ${TOMSK.ref.group}`, url: TOMSK.ref.url },
                  { cite: TOMSK.ref.baseCite, url: TOMSK.ref.baseUrl },
                  { cite: `Код + демо-корпус — ${TOMSK.data.repo}`, url: TOMSK.data.repoUrl },
                  { cite: TOMSK.headToHead.prCite, url: TOMSK.headToHead.prUrl },
                ]}
                note={TOMSK.data.note}
              />
            </div>
          </details>

          <details>
            <summary style={SUMMARY_STYLE}>Команды и сохранённые расчёты</summary>
            <p className="muted" style={{ fontSize: 12.5, margin: "10px 0 8px", maxWidth: "72ch" }}>
              Первые две строки — команды открытых прогонов: каждая запускается целиком и
              пишет результат в отдельный файл. Третья строка — исходные модули первого
              эксперимента и его сохранённый результат, а не готовый к запуску путь расчёта.
              Полный путь от корпуса до вердикта — в разделе «Можно повторить у себя».
            </p>
            <div style={{ display: "grid", gap: 8, maxWidth: "72ch" }}>
              {[
                { what: "Открытая выборка классиков", cmd: "python scripts/run_benchmark.py --pd-only", out: "docs/validation_pd.json" },
                { what: "Русский набор Proza.ru", cmd: "python scripts/run_proza_ru.py", out: null },
                { what: "Первый эксперимент по книгам", cmd: "src/stylo/eval/final.py + src/stylo/eval/lobo.py", out: "docs/final_comparison.csv" },
              ].map((r) => (
                <div key={r.cmd} style={{ display: "grid", gridTemplateColumns: "minmax(0, 20ch) minmax(0, 1fr)", gap: 10, alignItems: "baseline", borderBottom: "1px solid color-mix(in srgb, var(--line) 40%, transparent)", paddingBottom: 7 }}>
                  <span style={{ fontSize: 12.5, color: "var(--text)" }}>{r.what}</span>
                  <span className="mono muted" style={{ fontSize: 11, overflowWrap: "anywhere" }}>
                    {r.cmd}{r.out ? <> → {r.out}</> : null}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </div>
      </div>
    </section>
  );
}
