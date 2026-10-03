import { Fragment } from "react";
import { EditorialPanel, EditorialFigure, EditorialNote } from "@dmitrymake/rk-ui";
import { FEATURES, TOMSK, MEASUREMENT } from "../data.js";
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
const SUMMARY_STYLE = { cursor: "pointer", fontFamily: "var(--font-text)", fontSize: 16, fontWeight: "var(--fw-semibold)", letterSpacing: "var(--tracking-caption)", textTransform: "uppercase", color: "var(--text-muted)" };

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
  { marker: "02", title: "проверяем пересечения", body: "При подготовке корпуса проверяем совпадения содержания. Библиографические связи между изданиями и состав сборников требуют отдельной сверки.", color: "var(--icon-blue)", state: "done" },
  { marker: "03", title: "прячем книгу целиком", body: "Каждое произведение по очереди исключаем из обучения. Профиль автора строится по другим произведениям.", color: "var(--gold-ink)", state: "done" },
  { marker: "04", title: "сравниваем варианты", body: "На одинаковых отложенных произведениях сравниваем пакеты обучения и наборы признаков. Считаем попадания и изменения отдельных ответов.", color: "var(--editorial-positive)", state: "done" },
];

export default function Method() {
  return (
    <section className="section" id="method">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">02 / Метод</p>
          <h2>Проверка на независимых произведениях</h2>
          <p className="prose lead muted">
            Одну книгу целиком откладывают, а профиль её автора строят по другим
            произведениям. Пересечения содержания проверяют при подготовке корпуса;
            библиографические связи изданий требуют отдельной сверки. Затем проверяют, какой профиль
            окажется ближе к отложенной книге.
          </p>
          <p className="prose muted">
            Важно, чтобы такой ответ выдерживал смену книги. Множество окон одного
            романа связано общими героями, сюжетом и словарём. Окна помогают исследовать
            изменения внутри текста, а надёжность сравнения оценивается по произведениям.
          </p>
        </div>

        <div className="prose">
          <p>Словарь, частоты и классификатор строятся только по учебным книгам. Проверяемое произведение появляется лишь после обучения. Для каждой следующей книги модель строится заново.</p>
          <p>Каждое отложенное произведение образует отдельную проверку — fold. Всего в каждом варианте {MEASUREMENT.works} таких проверок.</p>
        </div>
        <EditorialFigure label="Схема 1" caption="Порядок проверки. Отложенная книга не участвует ни в выборе словаря, ни в оценке частот, ни в обучении классификатора.">
          <ol className="protocol-list">{PROTOCOL.map((step) => <li key={step.marker}><strong>{step.title}</strong><p>{step.body}</p></li>)}</ol>
        </EditorialFigure>
        <div className="prose">
          <h3>Два пакета обучения и два набора признаков</h3>
          <p>При работе с фрагментами авторские классы уже сбалансированы, но внутри автора длинные книги влияют сильнее. Пакет балансировки выравнивает суммарный вес авторов и вес книг внутри каждого автора; словарь и частотные веса IDF строятся по произведениям, а частоты делятся на полную длину текста.</p>
          <p>IDF учитывает, насколько редко признак встречается в учебных текстах. Поэтому разницу между пакетами нельзя свести к одному весовому коэффициенту.</p>
          <p>Сокращённый набор использует фиксированный список служебных слов вместо наиболее частых слов; из синтаксического блока исключены доли частей речи и словарное богатство. Это уменьшает некоторые пути влияния темы и жанра, но не гарантирует их полного устранения.</p>
        </div>
        <EditorialNote title="Почему единица проверки — произведение">
          <p>Окна внутри одной книги связаны общей темой, лексикой и героями. Поэтому число окон не равно числу независимых наблюдений. Здесь каждое произведение даёт один ответ: правильно или ошибочно назван автор. Авторы с большим числом книг сильнее влияют на общую точность.</p>
        </EditorialNote>

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
                <div key={r.a} className="data-row" style={{ display: "grid", gridTemplateColumns: "14ch 1fr 5ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                  <span style={{ fontSize: 16, color: r.hi ? "var(--text)" : "var(--text-muted)", fontWeight: r.hi ? 700 : 400 }}>{r.a}</span>
                  <MeterBar value={r.v} max={SIL_MAX} accent={r.hi ? "var(--icon-blue)" : "var(--text-muted)"} />
                  <span className="mono" style={{ fontSize: 16, color: r.hi ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v, 3)}</span>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gap: 10, alignContent: "start" }}>
              <p className="verdict" style={{ margin: 0 }}>
                А. Н. Толстой ({fmtScore(TA_ROW.v, 3)}) остаётся в том же узком диапазоне,
                что и авторы с бесспорным единственным авторством{" "}
                ({fmtRange(SIL_REF_LOW, SIL_REF_HIGH, (v) => fmtScore(v, 3))}). В этой подборке жанровое разнообразие не сопровождается более чётким делением
                фрагментов на группы.
              </p>
              <p className="note" style={{ margin: 0 }}>
                <strong style={{ color: "var(--text)" }}>{TA.nSelf} из {TA.nBooks}</strong> его
                книг ближе всего к нему самому, и ни одна не относится к его однофамильцу
                Льву Толстому.
              </p>
              <p className="note" style={{ margin: 0 }}>
                Это наблюдение на одной группе признаков и одной подборке книг.
                Его нельзя переносить на чувствительность остальных сравнений к теме и жанру.
              </p>
            </div>
          </div>
        </div>

        {/* каталог признаков — справочник */}
        <div className="reveal module">
          <h3>Какие признаки сравниваются</h3>
          <p className="prose muted" style={{ marginBottom: 22, maxWidth: "78ch" }}>
            Часть признаков ближе к поверхности текста: цепочки букв, частые слова, повторяющиеся
            обороты. Другие описывают устройство фразы: служебные слова, синтаксические связи,
            пунктуацию. Каждый блок проверяется отдельно — правдоподобная идея признака не
            считается результатом, пока не показала вклад в общей оценке. Две конфигурации
            признаков сопоставляются в <a href="#framework/results">результатах проверки на целых произведениях</a>.
          </p>
          <div className="grid cols-3">
            {FEATURES.map((f) => {
              const k = KIND_STYLE[f.kind] || {};
              return (
                <EditorialPanel key={f.id} style={{ opacity: k.dim ? 0.66 : 1 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ fontFamily: "var(--font-display)", fontSize: "1.05rem", color: "var(--text)" }}>{featName(f.name)}</span>
                  </div>
                  <p className="muted mono" style={{ margin: 0, fontSize: 16 }}>{featNote(f.note)}</p>
                </EditorialPanel>
              );
            })}
          </div>
          <details style={{ marginTop: 14 }}>
            <summary style={SUMMARY_STYLE}>Перевод сокращений</summary>
            <p className="muted" style={{ fontSize: 16, marginTop: 10, maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>n-граммы</strong> — цепочки из нескольких
              подряд идущих букв или слов; <strong style={{ color: "var(--text)" }}>MFW-300</strong> — 300 самых частых
              слов; <strong style={{ color: "var(--text)" }}>POS</strong> — часть речи; <strong style={{ color: "var(--text)" }}>TTR</strong> —
              доля неповторяющихся слов; <strong style={{ color: "var(--text)" }}>Hapax</strong> — слова, встреченные ровно
              один раз; <strong style={{ color: "var(--text)" }}>Yule</strong> — мера богатства словаря;{" "}
              <strong style={{ color: "var(--text)" }}>Замена слов структурными признаками</strong> — оставляем скелет из частей речи,
              вместо самих слов. Тематическую устойчивость такого представления проверяют отдельно;
              <strong style={{ color: "var(--text)" }}>синтаксические связи</strong> —
              кто с кем в предложении связан и как глубоко ветвится дерево разбора; <strong style={{ color: "var(--text)" }}>морфология</strong> —
              грамматические пометы слов (падеж, время, вид), взятые разбором spaCy (программой грамматического разбора).
            </p>
          </details>
        </div>

        {/* технические сверки — факты и источники, свёрнуты */}
        <details className="reveal module">
          <summary style={SUMMARY_STYLE}>Сверки на внешних данных и команды</summary>
          <p className="prose muted" style={{ fontSize: 16, borderLeft: "2px solid var(--gold)", paddingLeft: 14, maxWidth: "74ch", marginBottom: 14 }}>
            Что можно сверить с внешними данными: два чужих набора текстов, открытый код
            другой группы и команды для повторного прогона.
          </p>

          <details style={{ marginBottom: 10 }}>
            <summary style={SUMMARY_STYLE}>Чужие наборы данных (CCAT50, Proza.ru)</summary>
            <p className="muted" style={{ fontSize: 16, margin: "10px 0 8px", maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>CCAT50</strong> — общепринятый англоязычный
              набор (Reuters, 50 авторов). Равновесный ансамбль даёт{" "}
              {fmtScore(BENCH_EXT.ccat50Ensemble, 3)} при одном фиксированном делении данных.
              Опубликованный ориентир на буквенных n-граммах — {fmtScore(BENCH_EXT.ccat50Valla.ngramA, 3)},
              вариант на BERT — {fmtScore(BENCH_EXT.ccat50Valla.bertA, 3)}. Приведённый в
              обзоре результат {fmtScore(BENCH_EXT.ccat50Valla.record, 3)} получен при другом
              способе деления данных и с этим расчётом напрямую не сравнивается.
            </p>
            <p className="muted" style={{ fontSize: 16, margin: "0 0 8px", maxWidth: "80ch" }}>
              <strong style={{ color: "var(--text)" }}>Proza.ru</strong> — внешний русский набор
              (50 авторов), одно деление. Выше всех — один классификатор по цепочкам букв
              ({fmtScore(BENCH_EXT.prozaLeader, 3)}); равновесное усреднение всех групп ниже
              ({fmtScore(BENCH_EXT.prozaEqualEnsemble, 3)}); базовый ruBERT-tiny2 без дообучения —{" "}
              {fmtScore(BENCH_EXT.prozaNeuro, 3)}. Это один готовый вариант нейросетевой модели:
              дообученные и профильные модели для атрибуции авторства здесь не сравнивались,
              и причина низкого числа этим прогоном не установлена.
            </p>
            <p className="muted" style={{ fontSize: 16, margin: 0, maxWidth: "80ch" }}>
              Взвешивание по надёжности (веса групп пропорциональны их точности на отложенной
              части обучения) поднимает ансамбль до {fmtScore(BENCH_EXT.prozaEnsemble, 3)}. Его
              настройка выбрана по лучшему результату из небольшого перебора на этом же тесте,
              поэтому перевес +{fmtScore(BENCH_EXT.prozaEnsemble - BENCH_EXT.prozaLeader, 3)} над
              лидером настроен под тест; независимая проверка этого перевеса отсутствует.
            </p>
          </details>

          <details style={{ marginBottom: 10 }}>
            <summary style={SUMMARY_STYLE}>Сверка протокола с открытым кодом группы из ТУСУР</summary>
            <div className="split" style={{ alignItems: "start", marginTop: 12 }}>
              <div className="note" style={{ fontSize: 16 }}>
                <p style={{ margin: 0 }}>
                  Опубликовано {fmtPct(TOMSK.theirAcc, 1)} на {TOMSK_50.k} {ruAuthors(TOMSK_50.k)}.
                  В их открытом демо-коде отрывки одной книги попадают и в обучение, и в
                  проверку — деления по книге нет. На тех же данных и признаках, но с делением
                  по книгам, точность на {TOMSK_50.k} авторах — около {fmtPct(TOMSK_50.grouped)}{" "}
                  против {fmtPct(TOMSK_50.rand)} без деления. Разрыв того же порядка держится на
                  всех масштабах — от {TOMSK_KMIN} до {TOMSK_KMAX} авторов. Это сверка на открытом
                  демо-коде, и относится только к доступной демо-подборке.
                </p>
                {/* на узком экране таблица прокручивается внутри своей рамки, а не режется */}
                <p className="table-scroll-hint">Таблица прокручивается по горизонтали →</p>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Сравнение протоколов — таблицу можно прокрутить по горизонтали" style={{ marginTop: 12 }}>
                <div style={{ display: "grid", gridTemplateColumns: "5ch 1fr 1fr 6.5ch", gap: "4px 10px", fontSize: 16, alignItems: "center", minWidth: "36ch" }}>
                  <span className="mono muted">авт.</span>
                  <span className="mono muted">их протокол</span>
                  <span className="mono muted">по книге</span>
                  <span className="mono muted" style={{ textAlign: "right" }}>разрыв</span>
                  {TOMSK.headToHead.table.map((r) => (
                    <Fragment key={r.k}>
                      <span className="mono" style={{ color: "var(--text)" }}>{r.k}</span>
                      <span className="mono muted">{fmtScore(r.rand, 3)}</span>
                      <span className="mono" style={{ color: "var(--text)" }}>{fmtScore(r.grouped, 3)}</span>
                      <span className="mono" style={{ color: "var(--gold-ink)", textAlign: "right" }}>+{r.prem}</span>
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
            <summary style={SUMMARY_STYLE}>Команды и артефакты</summary>
            <p className="muted" style={{ fontSize: 16, margin: "10px 0 8px", maxWidth: "72ch" }}>
              Первые две строки — команды открытых прогонов: каждая запускается целиком и
              пишет результат в отдельный файл. Третья строка указывает модули покнижной диагностики и файл её результата.
              Для собственного сравнения предназначен раздел «Как запустить своё сравнение».
            </p>
            <div style={{ display: "grid", gap: 8, maxWidth: "72ch" }}>
              {[
                { what: "Открытая выборка классиков", cmd: "python scripts/run_benchmark.py --pd-only", out: "docs/validation_pd.json" },
                { what: "Русский набор Proza.ru", cmd: "python scripts/run_proza_ru.py", out: null },
                { what: "Корпус с пересечениями", cmd: "src/stylo/eval/final.py + src/stylo/eval/lobo.py", out: "docs/final_comparison.csv" },
              ].map((r) => (
                <div key={r.cmd} className="data-row" style={{ display: "grid", gridTemplateColumns: "minmax(0, 20ch) minmax(0, 1fr)", gap: 10, alignItems: "baseline", borderBottom: "1px solid color-mix(in srgb, var(--line) 40%, transparent)", paddingBottom: 7 }}>
                  <span style={{ fontSize: 16, color: "var(--text)" }}>{r.what}</span>
                  <span className="mono muted" style={{ fontSize: 16, overflowWrap: "anywhere" }}>
                    {r.cmd}{r.out ? <> → {r.out}</> : null}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </details>
      </div>
    </section>
  );
}
