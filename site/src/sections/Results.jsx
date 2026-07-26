import { Card, Stat } from "@dmitrymake/rk-ui";
import { MODELS, CHANNELS, HEADLINE } from "../data.js";
import { CORPUS } from "../corpus.js";
import { fmtScore, fmtPct } from "../format.js";
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

export default function Results() {
  return (
    <section className="section" id="results">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Первый эксперимент</p>
          <h2>Какие приметы стиля оказались полезны</h2>
          <p className="prose lead muted">
            Программа по очереди прятала каждую из {CORPUS.lobo.books} книг и искала её
            автора среди {CORPUS.lobo.tested_authors} кандидатов. Ни один отрывок
            проверяемой книги не участвовал в обучении. Позже выяснилось, что этого
            недостаточно: тот же рассказ иногда входил ещё и в сборник. Поэтому полоски
            ниже — результаты первого опыта и карта интересных сигналов, а не финальный
            рейтинг точности.
          </p>
        </div>

        {/* Наблюдаемые доли первого эксперимента. Интервалы не показываем:
            после обнаруженного пересечения они не описывают итоговую точность. */}
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
                  </span>
                  <span className="mono" style={{ fontSize: 12, color: m.kind === "ours" ? "var(--gold)" : "var(--text-muted)", textAlign: "right" }}>{fmtScore(m.acc, 3)}</span>
                </div>
              );
            })}
          </div>
          <p className="mono muted" style={{ fontSize: 11, marginTop: 16 }}>
            доля книг, верно распознанных в первом эксперименте · все значения будут
            пересчитаны на очищенном корпусе
          </p>
        </Card>

        {/* вывод: признаки окупаются */}
        <div className="split reveal module">
          <div className="prose">
            <p className="verdict">
              В первом эксперименте полный набор примет дал{" "}
              <strong style={{ color: "var(--text)" }}>{fmtPct(HEADLINE.accuracy, 1)}</strong>,
              а мешок слов — <strong style={{ color: "var(--icon-blue)" }}>{fmtPct(bow.acc, 1)}</strong>.
              Разница выглядит многообещающе: форма текста может добавлять информацию
              поверх выбора слов. Но подтвердить размер этого преимущества должен новый
              прогон без пересекающихся произведений.
            </p>
            <p>
              Классика — это метод Бэрроуза (Burrows Delta) и его косинусный вариант: оба сравнивают тексты
              по самым частым словам. Мешок слов проще — он смотрит лишь на то, какие слова и как часто
              встречаются, без их порядка.
            </p>
            <p>
              Когда рядом много близких авторов — донская школа, одесситы, деревенщики, —
              одних слов уже мало: общая тема даёт общую лексику. Тогда вступают приметы
              формы: построение фразы, служебные слова, пунктуация. Тема бывает общей,
              а способ собрать фразу остаётся личным.
            </p>
          </div>
          <div className="grid cols-2" style={{ alignContent: "start" }}>
            <Stat label="полный профиль · первый опыт" value={fmtScore(HEADLINE.accuracy, 3)} accent="var(--gold)" parade />
            <Stat label="мешок слов · первый опыт" value={fmtScore(bow.acc, 3)} accent="var(--icon-blue)" />
            <Stat label="macro-F1 · первый опыт" value={fmtScore(HEADLINE.styloMacroF1, 3)} accent="var(--icon-blue)" hint="Каждый автор получает равный вес. Это описательная точка первого эксперимента." />
            <Stat label="книг в проверке" value={CORPUS.lobo.books} accent="var(--text)" />
          </div>
        </div>

        {/* что несёт сигнал — вклад каждого канала */}
        <div className="reveal module">
          <h3>Из чего складывается почерк</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 18 }}>
            Каждый набор примет проверили отдельно одной и той же моделью. Баллы справа
            показывают, какие направления выглядели сильнее в первом опыте.
            Сильнее всех поодиночке — цепочки букв (символьные n-граммы, {fmtScore(CHANNELS.byId("char (2-5)").top1, 3)}).
            Чуть позади идут построение фразы, не зависящее от темы, и служебные слова. По отдельности приметы
            формы слабее. Зато <strong style={{ color: "var(--text)" }}>все вместе (ансамбль, {fmtScore(CHANNELS.ensembleTop1, 3)})
            дали на +{fmtScore(CHANNELS.ensembleTop1 - CHANNELS.rows[0].top1, 3)} больше лучшего одиночного набора</strong>{" "}
            (сам этот зазор отдельно не проверялся). Это хорошая гипотеза для повторного
            эксперимента: разные приметы описывают разные стороны почерка.
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
