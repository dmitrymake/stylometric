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
          <h2>Результаты по группам признаков</h2>
          <p className="prose lead muted">
            В первом эксперименте каждую книгу по очереди использовали для проверки: одну из{" "}
            {CORPUS.lobo.books} книг, автора которой искали среди {CORPUS.lobo.tested_authors}{" "}
            кандидатов. Отрывки проверяемой книги в обучение не попадали, но тот же рассказ
            иногда входил ещё и в сборник. Поэтому полоски ниже — наблюдения первого опыта,
            а не итоговая оценка точности.
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
            доля верно распознанных книг в первом эксперименте · будет пересчитано на
            очищенном корпусе
          </p>
        </Card>

        {/* вывод: признаки окупаются */}
        <div className="split reveal module">
          <div className="prose">
            <p className="verdict">
              В первом эксперименте полный набор признаков дал{" "}
              <strong style={{ color: "var(--text)" }}>{fmtPct(HEADLINE.accuracy, 1)}</strong>,
              а мешок слов — <strong style={{ color: "var(--icon-blue)" }}>{fmtPct(bow.acc, 1)}</strong>.
              Размер этой разницы должен подтвердить новый прогон без пересекающихся
              произведений.
            </p>
            <p>
              Классика — метод Бэрроуза (Burrows Delta) и его косинусный вариант: оба сравнивают
              тексты по самым частым словам. Мешок слов смотрит только на то, какие слова и как
              часто встречаются, без их порядка.
            </p>
          </div>
          <div className="grid cols-2" style={{ alignContent: "start" }}>
            <Stat label="полный профиль · первый опыт" value={fmtScore(HEADLINE.accuracy, 3)} accent="var(--gold)" parade />
            <Stat label="мешок слов · первый опыт" value={fmtScore(bow.acc, 3)} accent="var(--icon-blue)" />
            <Stat label="macro-F1 · первый опыт" value={fmtScore(HEADLINE.styloMacroF1, 3)} accent="var(--icon-blue)" hint="Каждый автор получает равный вес. Наблюдение первого эксперимента." />
            <Stat label="книг в проверке" value={CORPUS.lobo.books} accent="var(--text)" />
          </div>
        </div>

        {/* что несёт сигнал — вклад каждого канала */}
        <div className="reveal module">
          <h3>Из чего складывается почерк</h3>
          <p className="prose muted" style={{ maxWidth: "74ch", marginBottom: 18 }}>
            Каждый набор признаков проверили одной и той же моделью. Выше всех поодиночке —
            цепочки букв (символьные n-граммы, {fmtScore(CHANNELS.byId("char (2-5)").top1, 3)}),
            за ними построение фразы и служебные слова.{" "}
            <strong style={{ color: "var(--text)" }}>Все вместе (ансамбль, {fmtScore(CHANNELS.ensembleTop1, 3)})
            дали на +{fmtScore(CHANNELS.ensembleTop1 - CHANNELS.rows[0].top1, 3)} больше лучшего одиночного набора</strong>{" "}
            — сам этот зазор отдельно не проверялся.
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
              <strong style={{ color: "var(--danger)" }}>Словообразование по суффиксам (DSP) — {fmtScore(CHANNELS.byId("DSP (suffixes)").top1, 3)}</strong>:
              это самый слабый из показанных наборов признаков. В этой проверке он заметно
              уступает остальным группам.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
