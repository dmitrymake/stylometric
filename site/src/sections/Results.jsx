import { Card, Stat, ConfidenceBar } from "@dmitrymake/rk-ui";
import { MODELS, CHANNELS, HEADLINE, WORST_CLASSIC_P } from "../data.js";
import { CORPUS } from "../corpus.js";
import { fmtScore, fmtPct, fmtP } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";

const ACCENT = {
  ours: "var(--gold)",
  baseline: "var(--icon-blue)",
  classic: "var(--text-muted)",
  floor: "var(--danger)",
};

const CH_ACCENT = { base: "var(--icon-blue)", bow: "var(--text-muted)", weak: "var(--danger)" };
const bow = MODELS.find((m) => m.id === "bow_lr");
const CH_MAX = Math.max(CHANNELS.ensembleTop1, CHANNELS.rows[0].top1);

// Единые русские подписи лидерборда (данные в data.js только читаем, не меняем):
// у обоих вариантов Дельты — «частых слов» вместо MFW; сырой char-3gram сопровождаем
// словами «цепочки букв».
const MODEL_LABEL = { "char-3gram косинус": "косинус по цепочкам букв (char-3gram)" };
const modelLabel = (name) => MODEL_LABEL[name] || name.replace(" MFW", " частых слов");

// Подписи каналов: технические токены (char, POS, идиолект) — с короткой русской глоссой,
// как «цепочки букв» в тексте секции.
const CHANNEL_LABEL = {
  "char-n-граммы 2–5": "цепочки букв (char 2–5)",
  "синтаксис (связи + POS + метрики)": "синтаксис (связи, части речи, метрики)",
  "синтакс. связи (чистый идиолект)": "синтакс. связи (личный почерк)",
};
const channelLabel = (name) => CHANNEL_LABEL[name] || name;

// Склонение «книга» в предложном падеже: «на 251 книге», но «на 250 книгах».
// Единственное число — только когда число оканчивается на 1 (кроме 11).
const ruBooksPrep = (n) => {
  const mod100 = Math.abs(n) % 100;
  return mod100 % 10 === 1 && mod100 !== 11 ? "книге" : "книгах";
};

export default function Results() {
  return (
    <section className="section" id="results">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Результаты · книгу прячут целиком</p>
          <h2>Какие приметы стиля проходят проверку</h2>
          <p className="prose lead muted">
            Спрятать от программы целую книгу — все её отрывки до одного — и только потом
            спросить: чья? При обучении этой книги программа не может подсмотреть
            ответ у себя. Так устроена проверка на {CORPUS.lobo.books}{" "}
            {ruBooksPrep(CORPUS.lobo.books)}. Ответ ищут среди {CORPUS.lobo.tested_authors}{" "}
            авторов. У каждого есть и другие книги — по ним профиль и виден. У кого книга
            одна, тот без неё профиль теряет: в зачёт не идёт, только помогает учиться.
            Полоски ниже — доля верно угаданных книг у каждого метода.
            Границы (95%) считаются повторными пересчётами на случайных наборах книг, а не
            отрывков: иначе похожие куски одного романа нарисовали бы обманчиво гладкую
            картину.
          </p>
        </div>

        {/* лидерборд моделей с CI — кастомные бары, яркая контрастная скоба интервала */}
        <Card padding={24} className="reveal">
          <div style={{ display: "grid", gap: 15 }}>
            {MODELS.map((m) => {
              const SC = 0.95;
              const w = (v) => `${Math.max(0, Math.min(100, (v / SC) * 100))}%`;
              return (
                <div key={m.id} style={{ display: "grid", gridTemplateColumns: "minmax(0,16ch) 1fr 5ch", alignItems: "center", gap: 12 }}>
                  <span style={{ fontSize: 13, color: m.kind === "ours" ? "var(--text)" : "var(--text-muted)", fontWeight: m.kind === "ours" ? 700 : 400 }}>{modelLabel(m.name)}</span>
                  <span style={{ position: "relative", height: 16, borderRadius: 5, background: "var(--surface-sunken)" }}>
                    <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: w(m.acc), background: ACCENT[m.kind], borderRadius: 5, opacity: 0.92 }} />
                    {m.ci && (
                      <>
                        <span style={{ position: "absolute", left: w(m.ci[0]), width: `calc(${w(m.ci[1])} - ${w(m.ci[0])})`, top: "50%", height: 2, transform: "translateY(-50%)", background: "var(--text)", opacity: 0.9 }} />
                        {[m.ci[0], m.ci[1]].map((c, i) => (
                          <span key={i} style={{ position: "absolute", left: w(c), top: 1, bottom: 1, width: 2, marginLeft: -1, background: "var(--text)", borderRadius: 1 }} />
                        ))}
                      </>
                    )}
                  </span>
                  <span className="mono" style={{ fontSize: 12, color: m.kind === "ours" ? "var(--gold)" : "var(--text-muted)", textAlign: "right" }}>{fmtScore(m.acc, 3)}</span>
                </div>
              );
            })}
          </div>
          <p className="mono muted" style={{ fontSize: 11, marginTop: 16 }}>
            заливка — доля верных ответов (по целым книгам) · <span style={{ color: "var(--text)" }}>┠─┨</span> границы 95% (повторные пересчёты по книгам)
          </p>
        </Card>

        {/* вывод: признаки окупаются */}
        <div className="split reveal module">
          <div className="prose">
            <p className="verdict">
              Полный набор примет (<strong style={{ color: "var(--text)" }}>{fmtPct(HEADLINE.accuracy, 1)}</strong>) уверенно
              обгоняет и классику, и простой{" "}
              <strong style={{ color: "var(--icon-blue)" }}>мешок слов</strong> ({fmtPct(bow.acc, 1)}).
            </p>
            {HEADLINE.trainingWeighting === "chunk_weighted_training_legacy" && (
              <p className="mono muted" style={{ fontSize: 12 }}>
                Оговорка: при обучении длинная книга сейчас весит больше короткой. Пересчёт «одна книга —
                один голос» ещё впереди — число может немного сдвинуться.
              </p>
            )}
            <p>
              Классика — это метод Бэрроуза (Burrows Delta) и его косинусный вариант: оба сравнивают тексты
              по самым частым словам. Мешок слов проще — он смотрит лишь на то, какие слова и как часто
              встречаются, без их порядка.
            </p>
            <p>
              Перевес не случаен. Парный тест (McNemar) отвергает удачу — и против
              мешка слов (p&nbsp;{fmtP(bow.p)}), и против сильнейшего из классических методов
              (p&nbsp;{fmtP(WORST_CLASSIC_P)}).
            </p>
            <p>
              Когда рядом много близких авторов — донская школа, одесситы, деревенщики, —
              одних слов уже мало: общая тема даёт общую лексику. Тогда вступают приметы
              формы: построение фразы, служебные слова, пунктуация. Тема бывает общей,
              а способ собрать фразу остаётся личным.
            </p>
          </div>
          <div className="grid cols-2" style={{ alignContent: "start" }}>
            <Stat label="stylo · доля угаданных книг" value={fmtScore(HEADLINE.accuracy, 3)} accent="var(--gold)" parade />
            <Stat label="мешок слов" value={fmtScore(bow.acc, 3)} accent="var(--icon-blue)" />
            <Stat label="точность по авторам (macro-F1)" value={fmtScore(HEADLINE.styloMacroF1, 3)} accent="var(--icon-blue)" hint="единая оценка на всех авторах сразу; author-clustered интервал отозван — пересборка по авторам меняет набор классов и недействительна как разброс macro-F1" />
            <Stat label="p · перевес над мешком слов" value={fmtP(bow.p)} accent="var(--gold)" hint="тот же парный тест: перевес не случаен" />
          </div>
        </div>

        {/* что несёт сигнал — вклад каждого канала */}
        <div className="reveal module">
          <h3>Из чего складывается почерк</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 18 }}>
            Каждый набор примет проверили отдельно, одной и той же моделью; баллы — у полосок справа.
            Сильнее всех поодиночке — цепочки букв (символьные n-граммы, {fmtScore(CHANNELS.byId("char (2-5)").top1, 3)}).
            Чуть позади идут построение фразы, не зависящее от темы, и служебные слова. По отдельности приметы
            формы слабее. Зато <strong style={{ color: "var(--text)" }}>все вместе (ансамбль, {fmtScore(CHANNELS.ensembleTop1, 3)})
            обходят лучший одиночный набор на +{fmtScore(CHANNELS.ensembleTop1 - CHANNELS.rows[0].top1, 3)}</strong>{" "}
            (значимость именно этого зазора отдельно не проверялась): разные приметы описывают разные грани почерка.
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div>
              {CHANNELS.rows.map((r) => (
                <div key={r.name} style={{ display: "grid", gridTemplateColumns: "minmax(0,22ch) 1fr 5ch", alignItems: "center", gap: 8, padding: "3px 0" }}>
                  <span style={{ fontSize: 12, color: r.kind === "bow" ? "var(--icon-blue)" : r.kind === "weak" ? "var(--danger)" : "var(--text-muted)" }}>{channelLabel(r.name)}</span>
                  <MeterBar value={r.top1} max={CH_MAX} accent={CH_ACCENT[r.kind]} />
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{fmtScore(r.top1, 3)}</span>
                </div>
              ))}
              <div style={{ display: "grid", gridTemplateColumns: "minmax(0,22ch) 1fr 5ch", alignItems: "center", gap: 8, padding: "6px 0 0", borderTop: "1px solid color-mix(in srgb, var(--line) 50%, transparent)", marginTop: 6 }}>
                <span style={{ fontSize: 12, color: "var(--text)", fontWeight: 700 }}>АНСАМБЛЬ (равновесный)</span>
                <MeterBar value={CHANNELS.ensembleTop1} max={CH_MAX} accent="var(--success)" />
                <span className="mono" style={{ fontSize: 10.5, color: "var(--success)" }}>{fmtScore(CHANNELS.ensembleTop1, 3)}</span>
              </div>
            </div>
            <p className="note" style={{ margin: 0 }}>
              <strong style={{ color: "var(--danger)" }}>Ложный след — словообразование по суффиксам (DSP), {fmtScore(CHANNELS.byId("DSP (suffixes)").top1, 3)}</strong>:
              то, как автор лепит слова из приставок и суффиксов, на вид многообещающе, но на {CORPUS.benchmark.authors} авторах
              почти не различает писателей. Красивая примета подтверждается числом,
              а не интуицией.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
