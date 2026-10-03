import ArticleContents from "../components/ArticleContents.jsx";
import { EditorialPanel, EditorialMetric, EditorialFigure, EditorialBar, AnomalyGlyph } from "@dmitrymake/rk-ui";
import AuthorshipTimeline from "../components/AuthorshipTimeline.jsx";
import Sources from "../components/Sources.jsx";
import { fmtPct, fmtScore, fmtZ, fmtInt } from "../format.js";
import { ILF_PETROV, CASES, RIGOR } from "../segdata.js";

const D = ILF_PETROV.dvenadtsat;
const GOLD = ILF_PETROV.gold;
const H = ILF_PETROV.heterogeneity;
const SOLO = ILF_PETROV.solo;
const BG = CASES.bulgakov;

// Вопрос 1 (оба романа вне обучения): главный вопрос — не Булгаков ли.

// Палитра timeline. ВАЖНО: «чужие» куски схлопнуты в ОДИН цвет — это не «разные руки»,
// а отнесение неоднозначных отрывков к ближайшему из ВНЕШНИХ авторов корпуса (тип текста/шум).
const IP_NAME = "Илья Ильф и Евгений Петров";
const OTHER = "другие профили корпуса";
const COLOR_MAP = {
  [IP_NAME]: "var(--gold)",
  "Михаил Булгаков": "var(--cinnabar)",
  [OTHER]: "var(--text-muted)",
};
const TL_COLLAPSED = D.timeline.map(([a, c]) => [a === IP_NAME || a === "Михаил Булгаков" ? a : OTHER, c]);
const GOLD_COLLAPSED = GOLD.timeline.map(([a, c]) => [a === IP_NAME || a === "Михаил Булгаков" ? a : OTHER, c]);

// Разведение долей «12 стульев», чтобы подпись совпадала с цветами карты и сумма давала ровно 100%.
// Серое = чужие окна МИНУС булгаковские: на карте булгаковские окна окрашены отдельным (киноварным) цветом.
const D_GRAY = D.foreign - D.bulgakovShare;              // доля серых окон (другие профили корпуса)
const D_GRAY_N = Math.round(D_GRAY * D.nChunks);          // столько же серых окон
const D_BG_N = Math.round(D.bulgakovShare * D.nChunks);   // булгаковских окон

// Вопрос 2: силуэты — цель против одноавторских контролей.
const CONTROLS = Object.entries(H.controls).sort((a, b) => b[1] - a[1]);
const SIL_MAX = Math.max(H.targetSil, ...CONTROLS.map(([, v]) => v));

export default function IlfPetrov() {
  return (
    <section className="section" id="ilfpetrov">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Совместная проза</p>
          <h1>«Двенадцать стульев»: сравнение с Булгаковым</h1>
          <p className="prose lead muted">
            Версия об участии Булгакова требует сравнить роман с его собственной
            прозой и совместными произведениями Ильфа и Петрова. Другой вопрос —
            можно ли различить вклад самих соавторов. Эти задачи используют разные
            эталоны и разные способы проверки.
          </p>
        </div>
        <ArticleContents chapter="ilfpetrov" items={[
            ["ilfpetrov-section-1", "К какому профилю ближе «12 стульев»?"],
            ["ilfpetrov-section-2", "Та же проверка на обеих книгах"],
            ["ilfpetrov-section-3", "Карта ближайших профилей"],
            ["ilfpetrov-section-4", "Где Ильф, а где Петров?"]
          ]} />


        {/* ─────────────────────── ВОПРОС 1 ─────────────────────── */}
        <div className="module reveal">
          <p className="eyebrow" style={{ color: "var(--icon-blue)" }}>Вопрос 1 · сходство с Булгаковым</p>
          <h3 id="ilfpetrov-section-1" style={{ marginTop: 0 }}>К какому профилю ближе «12 стульев»?</h3>
          <p className="prose muted">
            В оконном сравнении с корпусом русской прозы роман отложен целиком. Все {fmtInt(D.nChunks)} его отрывков
            проходят проверку окно за окном по{" "}
            <strong style={{ color: "var(--text)" }}>всем авторам корпуса</strong> — не по одной
            заранее выбранной паре «дуэт против Булгакова». Карта показывает ближайший
            профиль каждого окна. Для поиска участия другого автора нужно отдельно
            проверить устойчивость соседних участков и чувствительность такого поиска.
          </p>
        </div>

        <div className="reveal">
          {/* слева — карта авторства */}
          <EditorialPanel>
            <AuthorshipTimeline
              timeline={TL_COLLAPSED}
              host={IP_NAME}
              colorMap={COLOR_MAP}
              height={84}
              caption={`«12 стульев», ${fmtInt(D.nChunks)} отрывков. Ближе к профилю дуэта — ${fmtPct(D.ipShare, 0)}; к внешним профилям — ${fmtPct(D.foreign, 0)}. Серое: ${fmtPct(D_GRAY, 1)}, ${fmtInt(D_GRAY_N)} окна; к Булгакову — ${fmtInt(D_BG_N)} окна (${fmtPct(D.bulgakovShare, 1)}), показанные отдельным цветом.`}
            />
          </EditorialPanel>


        </div>

        {/* методическая оговорка по вопросу 1 */}
        <div className="split reveal">
          <div className="prose">
            <p className="verdict">
              По Булгакову: <strong style={{ color: "var(--text)" }}>{fmtPct(D.bulgakovShare, 1)}</strong> отрывков
              — доля окон, для которых его профиль оказался ближайшим в этой панели. Сузим круг до
              дуэта, Булгакова, Катаева и Олеши — чаще всего ближайшим остаётся дуэт:
              дуэт забирает <strong style={{ color: "var(--text)" }}>{fmtPct(D.closed.ipShare, 0)}</strong> отрывков,
              Катаеву достаётся {fmtPct(D.closed.kataev, 0)}, а Булгакову лишь {fmtPct(D.closed.bulgakov, 0)}.
              Серые окна ближе к одному из внешних профилей. Причину такого сходства
              сама карта не устанавливает. Вклад Ильфа и Петрова внутри общего текста
              этим способом не разделяется.
            </p>
            <p className="note">
              Для отдельной сегментной проверки булгаковского участия первичный агрегат
              отсутствует в артефакте. Поэтому число связных «булгаковских» участков здесь не показано.
              Доступная карта отражает назначения отдельных окон.
            </p>
          </div>
          <div style={{ display: "grid", gap: 8, alignContent: "start" }}>
            <div className="mono muted" style={{ fontSize: 16 }}>расширенный список · доля окон к профилям авторов корпуса</div>
            <div className="grid cols-2">
              <EditorialMetric label="→ Ильф-Петров (напрямую)" value={fmtPct(D.ipShare, 0)} accent="var(--gold)" hint="доля отрывков" />
              <EditorialMetric label="→ Булгаков" value={fmtPct(D.bulgakovShare, 1)} accent="var(--icon-blue)" hint="доля отрывков; версия не поддержана" />
              <EditorialMetric label="→ Катаев" value={fmtPct(D.topForeign[0][1], 0)} accent="var(--cosmos)" hint="доля отрывков; внешний автор, близкий по стилю, не доказательство соавторства" />
            </div>
          </div>
        </div>

        {/* перепроверка на ЧИСТОМ признаке dependency (после кейса Шолохова) */}
        <div className="module reveal">
          <h3 id="ilfpetrov-section-2">Та же проверка на обеих книгах</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Чтобы жанр меньше мешал, тот же вопрос проверен на{" "}
            <strong style={{ color: "var(--text)" }}>синтаксических связях</strong> —
            группе признаков с меньшим влиянием тематической лексики в этой проверке. Различение авторов —
            {fmtScore(RIGOR.fa2[0].author)} (1.0 — идеально, 0.5 — наугад), различение
            жанра — {fmtScore(RIGOR.fa2[0].genreXA)}. Эталоны —
            совместная проза Ильфа и Петрова вне проверяемой дилогии и проза Булгакова. Через модель проходят обе книги
            дилогии: «12 стульев» и «Золотой телёнок».
          </p>
          <div className="split" style={{ alignItems: "start" }}>
            <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
              <div className="mono muted" style={{ fontSize: 16 }}>Оценка сходства с профилем дуэта в этом сравнении — по синтаксису, частям речи и связям слов:</div>
              {[
                { a: "эталон: проза Ильфа-Петрова", v: BG.ens.ipRef, kind: "ip" },
                { a: "«12 стульев» (спорная)", v: BG.ens.b12, kind: "td" },
                { a: "«Золотой телёнок» (спорная)", v: BG.ens.gold, kind: "td" },
                { a: "эталон: проза Булгакова", v: BG.ens.buRef, kind: "bu" },
              ].map((r) => (
                <div key={r.a} className="data-row" style={{ display: "grid", gridTemplateColumns: "1fr 4ch", alignItems: "center", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 16, color: r.kind === "td" ? "var(--text)" : "var(--text-muted)", fontWeight: r.kind === "td" ? 700 : 400, marginBottom: 3 }}>{r.a}</div>
                    <span style={{ display: "block", height: 8, borderRadius: 4, background: "var(--surface-sunken)", position: "relative", overflow: "hidden" }}>
                      <span style={{ position: "absolute", left: `${BG.ens.mid * 100}%`, top: 0, bottom: 0, width: 1, background: "var(--cinnabar)" }} title="граница между дуэтом и Булгаковым" />
                      <span style={{ display: "block", height: "100%", width: `${r.v * 100}%`, background: r.kind === "td" ? "var(--icon-blue)" : r.kind === "bu" ? "var(--cinnabar)" : "var(--gold)" }} />
                    </span>
                  </div>
                  <span className="mono" style={{ fontSize: 16, color: r.kind === "td" ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v)}</span>
                </div>
              ))}
              <div className="mono muted" style={{ fontSize: 16, marginTop: 2 }}>
                Красная черта — граница между дуэтом и Булгаковым: правее — ближе к дуэту, левее — к Булгакову.
              </div>
            </div>
            <p className="verdict" style={{ margin: 0 }}>
              Обе книги стоят на стороне Ильфа-Петрова: <strong style={{ color: "var(--text)" }}>{fmtScore(BG.ens.b12)}</strong>{" "}
              для «12 стульев» и <strong style={{ color: "var(--text)" }}>{fmtScore(BG.ens.gold)}</strong> для
              «Золотого телёнка» при эталоне дуэта {fmtScore(BG.ens.ipRef)} и Булгакове {fmtScore(BG.ens.buRef)}.
              На синтаксических связях вся дилогия тоже уходит к дуэту ({fmtScore(BG.dep.dilogy)}).
              «Золотой телёнок» — второй проверяемый роман той же дилогии. Он даёт
              то же направление при общей методике и эталонах.
            </p>
          </div>
        </div>

        {/* ── Кейс-близнец: «Золотой телёнок» — собственная карта авторства (пик истории) ── */}
        <div className="module reveal flow">
          <p className="eyebrow" style={{ color: "var(--icon-blue)" }}>Вторая книга · «Золотой телёнок»</p>
          <h3 id="ilfpetrov-section-3" style={{ marginTop: 0 }}>Карта ближайших профилей</h3>
          <p className="prose muted">
            «Золотой телёнок» проходит ту же проверку отрывок за отрывком, что и «12 стульев»:
            роман целиком убран из обучения, {fmtInt(GOLD.nChunks)} его отрывков отнесены к авторам
            всего корпуса. Карта второй книги почти сплошь золотая.
          </p>

          <div className="module" style={{ marginTop: "var(--beat-group)" }}>
            <EditorialPanel>
              <AuthorshipTimeline
                timeline={GOLD_COLLAPSED} host={IP_NAME} colorMap={COLOR_MAP} height={84}
                caption={`«Золотой телёнок», ${fmtInt(GOLD.nChunks)} отрывков, роман не участвовал в обучении. К дуэту — ${fmtPct(GOLD.ipShare, 1)} отрывков; серое — ${fmtPct(GOLD.foreign, 1)} (${fmtInt(Math.round(GOLD.foreign * GOLD.nChunks))} из ${fmtInt(GOLD.nChunks)}) к ближайшим внешним авторам; к Булгакову — ни одного.`}
              />
            </EditorialPanel>
          </div>

          <EditorialFigure label="Таблица 1" caption="Назначения окон двух романов в одной процедуре. Число отдельных участков отличается от числа окон: соседние окна могут образовывать один участок.">
            <table><thead><tr><th scope="col">Показатель</th><th scope="col">«12 стульев»</th><th scope="col">«Золотой телёнок»</th></tr></thead><tbody>
              <tr><th scope="row">Окон к профилю дуэта</th><td>{fmtPct(D.ipShare, 0)}</td><td>{fmtPct(GOLD.ipShare, 1)}</td></tr>
              <tr><th scope="row">Участков к внешним профилям</th><td>{fmtInt(D.nForeign)}</td><td>{fmtInt(GOLD.nForeign)}</td></tr>
            </tbody></table>
          </EditorialFigure>
          <p className="note">У «12 стульев» к внешним профилям, кроме Булгакова, ближе {fmtPct(D_GRAY, 1)} окон ({fmtInt(D_GRAY_N)} из {fmtInt(D.nChunks)}); к Булгакову — {fmtInt(D_BG_N)} окна. У «Телёнка» к внешним профилям ближе {fmtPct(GOLD.foreign, 1)} окон ({fmtInt(Math.round(GOLD.foreign * GOLD.nChunks))} из {fmtInt(GOLD.nChunks)}); назначений к Булгакову нет.</p>
          <p className="prose">В сравнении на отложенных текстах сводная оценка сходства «Телёнка» с дуэтом по синтаксису, частям речи и связям слов — {fmtScore(BG.ens.gold)}; на синтаксических связях — {fmtScore(BG.dep.gold)}.</p>
        </div>

        <hr className="rule reveal" />

        {/* ─────────────────────── ВОПРОС 2 ─────────────────────── */}
        <div className="module reveal">
          <p className="eyebrow" style={{ color: "var(--icon-blue)" }}>Вопрос 2 · различимость соавторов</p>
          <h3 id="ilfpetrov-section-4" style={{ marginTop: 0 }}>Где Ильф, а где Петров?</h3>
          <p className="prose muted">
            Для различения соавторов нужны сопоставимые сольные произведения каждого.
            В использованной подборке Ильф представлен записными книжками, Петров —
            военной публицистикой и мемуаром. Их различие смешивает автора и жанр.
          </p>
          <p className="prose muted">
            Остаётся <strong style={{ color: "var(--text)" }}>ход без обучающих примеров</strong>.
            Проверяем, делится ли текст на две группы сильнее, чем произведения
            одного автора. Используем служебные слова, ритм и синтаксис, чтобы
            уменьшить влияние тематической лексики. Это разделение сравниваем
            с текстами одного автора. Мера — насколько чётко текст распадается на две группы (силуэт).
          </p>
        </div>

        <div className="split reveal">
          {/* слева — силуэты: цель против контролей */}
          <EditorialPanel>
            <div style={{ display: "grid", gap: 16 }}>
              <EditorialBar
                value={H.targetSil / SIL_MAX}
                valueText={fmtScore(H.targetSil, 3)}
                label={<span style={{ color: "var(--gold-ink)" }}>Ильф-Петров · цель</span>}
                accent="var(--gold)"
              />
              {CONTROLS.map(([name, sil]) => (
                <EditorialBar
                  key={name}
                  value={sil / SIL_MAX}
                  valueText={fmtScore(sil, 3)}
                  label={<span style={{ color: "var(--text-muted)" }}>{name}</span>}
                  accent="var(--text-muted)"
                />
              ))}
            </div>
            <p className="muted mono" style={{ fontSize: 16, marginTop: 18 }}>
              Силуэт показывает, насколько отчётливо фрагменты разделяются на две группы.
              Разделение может отражать автора, тему или композицию; число групп не определяет
              число авторов. Контроли взяты из подборок с одной авторской меткой, включая Шолохова.
            </p>
          </EditorialPanel>

          {/* справа — z-оценка и глиф */}
          <div style={{ display: "grid", placeItems: "center", gap: 22 }}>
            <div style={{ textAlign: "center" }}>
              <div className="bignum ring-num" style={{ color: "var(--text)" }}>
                z = {fmtZ(H.z)}
              </div>
              <div className="mono muted" style={{ fontSize: 16, marginTop: 6 }}>
                цель {fmtScore(H.targetSil, 3)} · контроли в среднем {fmtScore(H.controlMean, 3)}
              </div>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <AnomalyGlyph kind="relation_mismatch" size={40} />
              <span className="muted" style={{ fontSize: 16, maxWidth: "24ch" }}>
                разделение слабее контрольного — z ниже нуля
              </span>
            </div>
            <p className="muted" style={{ fontSize: 16, textAlign: "center", maxWidth: "30ch" }}>
              Силуэт дуэта <strong style={{ color: "var(--text)" }}>ниже</strong> среднего
              по выбранным контрольным текстам. z выражает отклонение от их среднего
              в единицах контрольного разброса. В этой проверке разделение дилогии на две
              группы слабее, чем у контролей.
            </p>
          </div>
        </div>

        {/* вердикт по вопросу 2 */}
        <p className="verdict reveal">
          Низкий силуэт не даёт основания считать две группы фрагментов отчётливо разделимыми.
          По этой проверке нельзя распределить страницы между соавторами или установить число авторов.
        </p>

        {/* мини-кейс: сольные тексты Ильфа и Петрова — разделимы ли руки соавторов */}
        <div className="module reveal">
          <h4 style={{ marginBottom: 6 }}>А если добавить их сольные тексты?</h4>
          <p className="prose muted" style={{ maxWidth: "76ch", marginBottom: 16 }}>
            Для каждого соавтора есть отдельная подборка. В корпус входят
            общедоступные одиночные тексты — <strong style={{ color: "var(--text)" }}>«Записные книжки» Ильфа</strong>{" "}
            ({fmtInt(SOLO.ilfWords)} слов) и <strong style={{ color: "var(--text)" }}>военная
            публицистика и мемуар Петрова</strong> ({fmtInt(SOLO.petrovWords)} слов); на них обучена
            модель «Ильф против Петрова», через неё прошли романы.
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div className="grid cols-2">
              <EditorialMetric label="различимость сольных подборок" value={fmtScore(SOLO.soloAuc)} accent="var(--gold)" hint="автор и жанр различаются одновременно" />
              <EditorialMetric label="только служебные слова" value={fmtScore(SOLO.fwAuc)} accent="var(--icon-blue)" hint="различает эти подборки; влияние жанра остаётся" />
              <EditorialMetric label="«12 стульев» → Петров" value={fmtPct(SOLO.projP12, 0)} accent="var(--text-muted)" />
              <EditorialMetric label="«Зол. телёнок» → Петров" value={fmtPct(SOLO.projPgt, 0)} accent="var(--text-muted)" />
            </div>
            <p className="verdict" style={{ margin: 0 }}>
              Различимость сольных подборок — {fmtScore(SOLO.soloAuc)} из 1.0; при этом{" "}
              <strong style={{ color: "var(--editorial-accent)" }}>автор и жанр здесь связаны</strong>: сольный Ильф в этой подборке —{" "}
              <em>афоризмы из записных книжек</em>, сольный Петров — <em>военные очерки</em>. Разные
              жанры различить легко. Романы (третий жанр) сбиваются в узкую полосу
              ({fmtPct(SOLO.projP12, 0)}–{fmtPct(SOLO.projPgt, 0)} «петровских» отрывков) и одинаково
              далеки от обоих сольных жанров. Такое сравнение не позволяет раздать
              страницы между соавторами. Следующий шаг — подобрать самостоятельную
              прозу сопоставимого периода и регистра и проверить её пересечения с дилогией.
            </p>
          </div>
        </div>

        <p className="verdict reveal">
          В этих сравнениях обе книги ближе к профилю дуэта, чем к профилю Булгакова.
          Проверка неоднородности не позволяет разделить страницы дилогии между Ильфом и Петровым;
          сама по себе она не устанавливает число соавторов.
        </p>

        <Sources
          label="Внешние исследования и контекст"
          items={[
            { cite: "B. Ryabko, N. Savina (2021). Using Data Compression to Build a Method for Statistically Verified Attribution of Literary Texts — компрессионный подход и сравнение дилогии с прозой Булгакова", url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8534409/" },
            { cite: "Е. С. Тарасова (2020). «Илья Ильф и Евгений Петров как соавторы романов Ильфа и Петрова» — связи сольных рассказов и фельетонов с совместными романами", url: "https://www.philology.nsc.ru/journals/sis/article.php?id=264" },
          ]}
          note="Это внешние работы о постановке вопроса и способах сравнения. Назначения блоков в компрессионном исследовании не являются вероятностями авторства и не входят в показатели графиков Stylo."
        />
        <Sources
          label="Источники графиков Stylo"
          items={[
            { cite: "Карты окон и сравнение кандидатов — docs/ilfpetrov_timeline.json, docs/disputed_ilfpetrov.json" },
            { cite: "Сольные подборки и неоднородность — docs/ilf_vs_petrov.json, docs/ilfpetrov_heterogeneity.json" },
            { cite: "Сравнение профилей и групп признаков — docs/cases_attribution.json, docs/feature_audit2.json; полная привязка полей — site/src/generated/manifest.json" },
          ]}
          note="Показатели главы относятся к оконным, синтаксическим и сольным сравнениям перечисленных корпусов; это отдельные протоколы от проверки на 248 произведениях."
        />
      </div>
    </section>
  );
}
