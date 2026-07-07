import { Card, Stat, ConfidenceBar, AnomalyGlyph } from "@dmitrymake/rk-ui";
import AuthorshipTimeline from "../components/AuthorshipTimeline.jsx";
import RingStat from "../components/RingStat.jsx";
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
const OTHER = "соседи по типу текста и шум";
const COLOR_MAP = {
  [IP_NAME]: "var(--gold)",
  "Михаил Булгаков": "var(--cinnabar)",
  [OTHER]: "var(--text-muted)",
};
const TL_COLLAPSED = D.timeline.map(([a, c]) => [a === IP_NAME || a === "Михаил Булгаков" ? a : OTHER, c]);
const GOLD_COLLAPSED = GOLD.timeline.map(([a, c]) => [a === IP_NAME || a === "Михаил Булгаков" ? a : OTHER, c]);

// Разведение долей «12 стульев», чтобы подпись совпадала с цветами карты и сумма давала ровно 100%.
// Серое = чужие окна МИНУС булгаковские: на карте булгаковские окна окрашены отдельным (киноварным) цветом.
const D_GRAY = D.foreign - D.bulgakovShare;              // доля серых окон (соседи по типу текста и шум)
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
          <p className="eyebrow">Кейс · Ильф и Петров</p>
          <h2>Ильф и Петров: писал ли дилогию Булгаков?</h2>
          <p className="prose lead muted">
            Ходит версия: будто «12 стульев» написал Булгаков, а
            Ильф и Петров лишь дали роману свои имена. Проверить её можно текстом.
            Но здесь спрятаны два разных вопроса, и путать их нельзя. Первый:
            виден ли в дилогии почерк Булгакова. Второй: можно ли внутри
            совместной прозы отделить руку Ильфа от руки Петрова. На первый вопрос
            текст отвечает ясно. Со вторым мешает то, что у соавторов нет ни
            одного романа, написанного порознь.
          </p>
        </div>

        {/* ─────────────────────── ВОПРОС 1 ─────────────────────── */}
        <div className="module reveal">
          <p className="eyebrow" style={{ color: "var(--icon-blue)" }}>Вопрос 1 · авторство</p>
          <h3 style={{ marginTop: 0 }}>Писал ли «12 стульев» Булгаков?</h3>
          <p className="prose muted">
            Версию проверяем строго: роман сравниваем сразу со всеми авторами, а
            сам он в обучении не участвует. Все {fmtInt(D.nChunks)} его отрывков
            проходят проверку окно за окном по{" "}
            <strong style={{ color: "var(--text)" }}>всем авторам корпуса</strong> — не по одной
            заранее выбранной паре «дуэт против Булгакова». Если версия верна, Булгаков
            проявится не в одиночных окнах, а в заметной доле текста или в связном куске.
          </p>
        </div>

        <div className="split reveal">
          {/* слева — карта авторства */}
          <Card padding={24}>
            <AuthorshipTimeline
              timeline={TL_COLLAPSED}
              host={IP_NAME}
              colorMap={COLOR_MAP}
              height={84}
              caption={`«12 стульев», ${fmtInt(D.nChunks)} отрывков, роман не участвовал в обучении. К дуэту — ${fmtPct(D.ipShare, 0)} отрывков. Остальные ${fmtPct(D.foreign, 0)} почти сплошь — соседи по типу текста и шум (серое: ${fmtPct(D_GRAY, 1)}, ${fmtInt(D_GRAY_N)} окна, не вторая рука внутри дуэта); к Булгакову — лишь ${fmtInt(D_BG_N)} окна (${fmtPct(D.bulgakovShare, 1)}) отдельного цвета, ни одного связного куска.`}
            />
          </Card>

          {/* справа — кольцо «не Булгаков» */}
          <div style={{ display: "grid", placeItems: "center", gap: 22 }}>
            <RingStat frac={D.bulgakovShare} big={fmtPct(D.bulgakovShare, 1)} caption="отрывков к Булгакову" accent="var(--gold)" />
            <p className="muted" style={{ fontSize: 13.5, textAlign: "center", maxWidth: "30ch" }}>
              К Булгакову относится только <strong style={{ color: "var(--text)" }}>{fmtPct(D.bulgakovShare, 1)}</strong> отрывков.
              Напрямую к дуэту — {fmtPct(D.ipShare, 0)}. Остальное уходит к внешним авторам,
              похожим по манере, а не к Булгакову.
            </p>
          </div>
        </div>

        {/* методическая оговорка по вопросу 1 */}
        <div className="split reveal">
          <div className="prose">
            <p className="verdict">
              По Булгакову: <strong style={{ color: "var(--text)" }}>{fmtPct(D.bulgakovShare, 1)}</strong> отрывков
              и ни одного связного куска — это доля среди всех внешних авторов корпуса. Сузим круг до
              четырёх реальных подозреваемых (дуэт, Булгаков, Катаев, Олеша) — победитель тот же:
              дуэт забирает <strong style={{ color: "var(--text)" }}>{fmtPct(D.closed.ipShare, 0)}</strong> отрывков,
              Катаеву достаётся {fmtPct(D.closed.kataev, 0)}, а Булгакову лишь {fmtPct(D.closed.bulgakov, 0)}.
              Серые участки на карте — не «вторая рука» внутри дуэта, а спорные окна, где ближе
              всего оказался внешний автор из-за темы или манеры. Вклад Ильфа и Петрова внутри общего
              текста этим способом не разделяется.
            </p>
            <p className="note">
              Второй, независимой проверки для этого корпуса нет. Вывод
              опирается на один метод — он взвешивает множество мелких признаков речи.
            </p>
          </div>
          <div style={{ display: "grid", gap: 8, alignContent: "start" }}>
            <div className="mono muted" style={{ fontSize: 11 }}>открытый круг · доля среди всех авторов корпуса</div>
            <div className="grid cols-2">
              <Stat label="→ Ильф-Петров (напрямую)" value={fmtPct(D.ipShare, 0)} accent="var(--gold)" parade hint="доля отрывков" />
              <Stat label="→ Булгаков" value={fmtPct(D.bulgakovShare, 1)} accent="var(--icon-blue)" hint="доля отрывков; версия не поддержана" />
              <Stat label="→ Катаев" value={fmtPct(D.topForeign[0][1], 0)} accent="var(--cosmos)" hint="доля отрывков; внешний автор, близкий по стилю, не доказательство соавторства" />
              <Stat label="булгаковских сегментов" value={fmtInt(D.nSegments)} accent="var(--text-muted)" hint={`${fmtInt(D_BG_N)} разрозненных окна — ни одного связного куска`} />
            </div>
          </div>
        </div>

        {/* перепроверка на ЧИСТОМ признаке dependency (после кейса Шолохова) */}
        <div className="module reveal">
          <h3>Та же проверка на обеих книгах</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Чтобы жанр меньше мешал, тот же вопрос проверен на{" "}
            <strong style={{ color: "var(--text)" }}>синтаксических связях</strong> —
            признаке, который почти не зависит от темы. Авторов он различает уверенно
            ({fmtScore(RIGOR.fa2[0].author)}: 1.0 — идеально, 0.5 — наугад) и при этом
            почти не цепляется за жанр ({fmtScore(RIGOR.fa2[0].genreXA)}). Эталоны —
            своя проза Ильфа-Петрова и проза Булгакова. Через модель проходят обе книги
            дилогии: «12 стульев» и «Золотой телёнок».
          </p>
          <div className="split" style={{ alignItems: "start" }}>
            <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
              <div className="mono muted" style={{ fontSize: 11 }}>Насколько текст похож на дуэт (0 — совсем нет, 1 — точно дуэт), на отложенных текстах, по синтаксису, частям речи и связям слов:</div>
              {[
                { a: "эталон: проза Ильфа-Петрова", v: BG.ens.ipRef, kind: "ip" },
                { a: "«12 стульев» (спорная)", v: BG.ens.b12, kind: "td" },
                { a: "«Золотой телёнок» (спорная)", v: BG.ens.gold, kind: "td" },
                { a: "эталон: проза Булгакова", v: BG.ens.buRef, kind: "bu" },
              ].map((r) => (
                <div key={r.a} style={{ display: "grid", gridTemplateColumns: "1fr 4ch", alignItems: "center", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 12, color: r.kind === "td" ? "var(--text)" : "var(--text-muted)", fontWeight: r.kind === "td" ? 700 : 400, marginBottom: 3 }}>{r.a}</div>
                    <span style={{ display: "block", height: 8, borderRadius: 4, background: "var(--surface-sunken)", position: "relative", overflow: "hidden" }}>
                      <span style={{ position: "absolute", left: `${BG.ens.mid * 100}%`, top: 0, bottom: 0, width: 1, background: "var(--cinnabar)" }} title="граница между дуэтом и Булгаковым" />
                      <span style={{ display: "block", height: "100%", width: `${r.v * 100}%`, background: r.kind === "td" ? "var(--icon-blue)" : r.kind === "bu" ? "var(--cinnabar)" : "var(--gold)" }} />
                    </span>
                  </div>
                  <span className="mono" style={{ fontSize: 11, color: r.kind === "td" ? "var(--icon-blue)" : "var(--text-muted)" }}>{fmtScore(r.v)}</span>
                </div>
              ))}
              <div className="mono muted" style={{ fontSize: 11, marginTop: 2 }}>
                Красная черта — граница между дуэтом и Булгаковым: правее — ближе к дуэту, левее — к Булгакову.
              </div>
            </div>
            <p className="verdict" style={{ margin: 0 }}>
              Обе книги стоят на стороне Ильфа-Петрова: <strong style={{ color: "var(--text)" }}>{fmtScore(BG.ens.b12)}</strong>{" "}
              для «12 стульев» и <strong style={{ color: "var(--text)" }}>{fmtScore(BG.ens.gold)}</strong> для
              «Золотого телёнка» при эталоне дуэта {fmtScore(BG.ens.ipRef)} и Булгакове {fmtScore(BG.ens.buRef)}.
              На синтаксических связях вся дилогия тоже уходит к дуэту ({fmtScore(BG.dep.dilogy)}).
              «Золотой телёнок» здесь не новая «булгаковская» версия, а вторая,
              независимая проверка: она даёт то же направление.
            </p>
          </div>
        </div>

        {/* ── Кейс-близнец: «Золотой телёнок» — собственная карта авторства (пик истории) ── */}
        <div className="module reveal flow">
          <p className="eyebrow" style={{ color: "var(--icon-blue)" }}>Кейс-близнец · Золотой телёнок</p>
          <h3 style={{ marginTop: 0 }}>Своя карта авторства</h3>
          <p className="prose muted">
            «Золотой телёнок» проходит ту же проверку отрывок за отрывком, что и «12 стульев»:
            роман целиком убран из обучения, {fmtInt(GOLD.nChunks)} его отрывков отнесены к авторам
            всего корпуса. Карта второй книги почти сплошь золотая.
          </p>

          <div className="split module" style={{ marginTop: "var(--beat-group)" }}>
            <Card padding={24}>
              <AuthorshipTimeline
                timeline={GOLD_COLLAPSED} host={IP_NAME} colorMap={COLOR_MAP} height={84}
                caption={`«Золотой телёнок», ${fmtInt(GOLD.nChunks)} отрывков, роман не участвовал в обучении. К дуэту — ${fmtPct(GOLD.ipShare, 1)} отрывков; серое — ${fmtPct(GOLD.foreign, 1)} (${fmtInt(Math.round(GOLD.foreign * GOLD.nChunks))} из ${fmtInt(GOLD.nChunks)}) к ближайшим внешним авторам; к Булгакову — ни одного.`}
              />
            </Card>
            <RingStat frac={GOLD.ipShare} big={fmtPct(GOLD.ipShare, 1)} caption="отрывков к дуэту" accent="var(--gold)" />
          </div>

          {/* прямой контраст двух книг — визуальная пауза, не курсив */}
          <div className="grid cols-2 module" style={{ marginTop: "var(--beat-group)" }}>
            <Stat label="«12 стульев» → дуэт" value={fmtPct(D.ipShare, 0)} accent="var(--text-muted)" parade hint="доля отрывков" />
            <Stat label="«Золотой телёнок» → дуэт" value={fmtPct(GOLD.ipShare, 1)} accent="var(--gold)" parade hint="доля отрывков" />
            <Stat label="«12 стульев»: чужих связных участков" value={fmtInt(D.nForeign)} accent="var(--text-muted)" hint={`соседи по стилю — ${fmtPct(D_GRAY, 1)} отрывков (${fmtInt(D_GRAY_N)} из ${fmtInt(D.nChunks)}); к Булгакову — ${fmtInt(D_BG_N)} окна, ни одного связного`} />
            <Stat label="«Телёнок»: чужих связных участков" value={fmtInt(GOLD.nForeign)} accent="var(--gold)" hint={`${fmtPct(GOLD.foreign, 1)} отрывков (${fmtInt(Math.round(GOLD.foreign * GOLD.nChunks))} из ${fmtInt(GOLD.nChunks)}); к Булгакову — ни одного`} />
            <Stat label="похоже на дуэт (сводно)" value={fmtScore(BG.ens.gold)} accent="var(--gold)" hint="на отложенных текстах: по синтаксису, частям речи и связям слов" />
            <Stat label="на синтаксических связях" value={fmtScore(BG.dep.gold)} accent="var(--icon-blue)" hint="признак, меньше всего зависящий от темы" />
          </div>
        </div>

        <hr className="rule reveal" />

        {/* ─────────────────────── ВОПРОС 2 ─────────────────────── */}
        <div className="module reveal">
          <p className="eyebrow" style={{ color: "var(--icon-blue)" }}>Вопрос 2 · две руки внутри дуэта</p>
          <h3 style={{ marginTop: 0 }}>Где Ильф, а где Петров?</h3>
          <p className="prose muted">
            Здесь обычная атрибуция бессильна. У Ильфа и Петрова нет сольной прозы,
            на которой модель научилась бы различать «это Ильф, а это Петров»:
            оба всегда писали вместе. Обучать не на чем.
          </p>
          <p className="prose muted">
            Остаётся <strong style={{ color: "var(--text)" }}>ход без обучающих примеров</strong>.
            Если внутри дуэта две разные руки, текст должен делиться надвое
            сильнее, чем у одного автора. Отрывки делим на две группы по признакам, не
            зависящим от темы: служебные слова, ритм, синтаксис. Эту «двугорбость» сравниваем
            с текстами одного автора. Мера — насколько чётко текст распадается на две группы (силуэт).
          </p>
        </div>

        <div className="split reveal">
          {/* слева — силуэты: цель против контролей */}
          <Card padding={24}>
            <div style={{ display: "grid", gap: 16 }}>
              <ConfidenceBar
                value={H.targetSil / SIL_MAX}
                valueText={fmtScore(H.targetSil, 3)}
                label={<span style={{ color: "var(--gold)" }}>Ильф-Петров · цель</span>}
                accent="var(--gold)"
              />
              {CONTROLS.map(([name, sil]) => (
                <ConfidenceBar
                  key={name}
                  value={sil / SIL_MAX}
                  valueText={fmtScore(sil, 3)}
                  label={<span style={{ color: "var(--text-muted)" }}>{name}</span>}
                  accent="var(--text-muted)"
                />
              ))}
            </div>
            <p className="muted mono" style={{ fontSize: 12.5, marginTop: 18 }}>
              насколько чётко текст делится на две группы (силуэт). Выше = заметнее «две руки».
              Для сравнения взяты тексты одного автора. (Шолохов здесь взят как практический
              одиночка, чтобы задать обычный разброс этой меры — к спору об авторстве «Тихого
              Дона» это отношения не имеет.)
            </p>
          </Card>

          {/* справа — z-оценка и глиф */}
          <div style={{ display: "grid", placeItems: "center", gap: 22 }}>
            <div style={{ textAlign: "center" }}>
              <div className="bignum ring-num" style={{ color: "var(--text)" }}>
                z = {fmtZ(H.z)}
              </div>
              <div className="mono muted" style={{ fontSize: 12.5, marginTop: 6 }}>
                цель {fmtScore(H.targetSil, 3)} · контроли в среднем {fmtScore(H.controlMean, 3)}
              </div>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <AnomalyGlyph kind="relation_mismatch" size={40} />
              <span className="muted" style={{ fontSize: 13.5, maxWidth: "24ch" }}>
                двугорбость ниже одноавторского фона — z ниже нуля
              </span>
            </div>
            <p className="muted" style={{ fontSize: 13.5, textAlign: "center", maxWidth: "30ch" }}>
              Силуэт дуэта <strong style={{ color: "var(--text)" }}>ниже</strong> среднего
              по текстам одного автора. z показывает, насколько это отклонение велико
              на фоне обычного разброса: здесь z ниже нуля — дуэт делится надвое
              даже слабее одного автора. Никакого «второго горба» нет.
            </p>
          </div>
        </div>

        {/* вердикт по вопросу 2 */}
        <p className="verdict reveal">
          Проза Ильфа и Петрова делится на две группы слабее, чем тексты одного автора.
          В доступных данных их совместная работа выглядит как один устойчивый почерк —
          разделить вклад каждого внутри романов нечем.
        </p>

        {/* мини-кейс: сольные тексты Ильфа и Петрова — разделимы ли руки соавторов */}
        <div className="module reveal">
          <h4 style={{ marginBottom: 6 }}>А если добавить их сольные тексты?</h4>
          <p className="prose muted" style={{ maxWidth: "76ch", marginBottom: 16 }}>
            Возражение по делу: у каждого есть написанное в одиночку. В корпусе есть
            общедоступные одиночные тексты — <strong style={{ color: "var(--text)" }}>«Записные книжки» Ильфа</strong>{" "}
            ({fmtInt(SOLO.ilfWords)} слов) и <strong style={{ color: "var(--text)" }}>военная
            публицистика и мемуар Петрова</strong> ({fmtInt(SOLO.petrovWords)} слов); на них обучена
            модель «Ильф против Петрова», через неё прошли романы.
          </p>
          <div className="split" style={{ alignItems: "center" }}>
            <div className="grid cols-2">
              <Stat label="сольные различимы · уверенность различения" value={fmtScore(SOLO.soloAuc)} accent="var(--gold)" parade hint="но это разные ЖАНРЫ" />
              <Stat label="только служебные слова" value={fmtScore(SOLO.fwAuc)} accent="var(--icon-blue)" hint="меньше зависит от жанра — но всё равно высоко" />
              <Stat label="«12 стульев» → Петров" value={fmtPct(SOLO.projP12, 0)} accent="var(--text-muted)" />
              <Stat label="«Зол. телёнок» → Петров" value={fmtPct(SOLO.projPgt, 0)} accent="var(--text-muted)" />
            </div>
            <p className="verdict" style={{ margin: 0 }}>
              Сольные тексты различаются «уверенно» (различает на {fmtScore(SOLO.soloAuc)} из 1.0) — но это{" "}
              <strong style={{ color: "var(--cinnabar)" }}>жанр, не рука</strong>: единственный сольный Ильф —{" "}
              <em>афоризмы из записных книжек</em>, единственный сольный Петров — <em>военные очерки</em>. Разные
              жанры различить легко. Романы (третий жанр) сбиваются в узкую полосу
              ({fmtPct(SOLO.projP12, 0)}–{fmtPct(SOLO.projPgt, 0)} «петровских» отрывков) и одинаково
              далеки от обоих сольных жанров — это не разделение рук, а расстояние до чужого жанра. Сольных образцов{" "}
              <strong style={{ color: "var(--text)" }}>в романном жанре</strong> у них нет и быть не может: они
              писали каждую строку вдвоём. Тот же предел <strong style={{ color: "var(--text)" }}>«жанр совпадает с автором»</strong>,
              что и у «Тихого Дона».
            </p>
          </div>
        </div>

        <p className="verdict reveal">
          Версию о Булгакове данные не поддерживают, а совместная проза
          Ильфа и Петрова не делится на два устойчивых почерка. Дилогия совместима
          с авторством дуэта, но разделить страницы между соавторами не даёт.
        </p>
      </div>
    </section>
  );
}
