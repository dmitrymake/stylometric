import ArticleContents from "../components/ArticleContents.jsx";
import { EditorialPanel, EditorialMetric, Badge } from "@dmitrymake/rk-ui";
import { TARAS } from "../segdata.js";
import { fmtScore, fmtP, fmtPct, fmtInt } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";
import Sources from "../components/Sources.jsx";

// в данных авторы и статусы лежат сырыми ярлыками папок — показываем по-русски.
const NAME = {
  gogol: "Гоголь",
  gogol_early: "ранний Гоголь",
  gogol_late: "поздний Гоголь",
  annenkov_1840s: "Анненков (1840-е)",
  somov: "Сомов",
  grebenka: "Гребёнка",
  narezhny: "Нарежный",
  sollogub: "Соллогуб",
  pushkin: "Пушкин",
  turgenev: "Тургенев",
  dostoevsky: "Достоевский",
  saltykov: "Салтыков-Щедрин",
  tolstoy: "Толстой",
  leskov: "Лесков",
};
const nm = (s) => NAME[s] || s;

// дательная форма имени для оборотов «ближе к …» (после «к» нужен дательный падеж).
const DATIVE = {
  gogol: "Гоголю",
  gogol_early: "раннему Гоголю",
  gogol_late: "позднему Гоголю",
  annenkov_1840s: "Анненкову (1840-е)",
  somov: "Сомову",
  grebenka: "Гребёнке",
  narezhny: "Нарежному",
  sollogub: "Соллогубу",
  pushkin: "Пушкину",
  turgenev: "Тургеневу",
  dostoevsky: "Достоевскому",
  saltykov: "Салтыкову-Щедрину",
  tolstoy: "Толстому",
  leskov: "Лескову",
};
const nmDat = (s) => DATIVE[s] || nm(s);

// Русское склонение слова «слово» после числа (373 → «слова», 36 577 → «слов»).
const ruWords = (n) => {
  const mod100 = Math.abs(n) % 100, mod10 = mod100 % 10;
  if (mod100 >= 11 && mod100 <= 14) return "слов";
  if (mod10 === 1) return "слово";
  if (mod10 >= 2 && mod10 <= 4) return "слова";
  return "слов";
};

// доля, которая округлилась бы в ноль при заданной точности, показывается порогом снизу
// («< 0.01%»), чтобы «почти нет» не выглядело как точный ноль.
const fmtPctFloor = (frac, digits) => {
  const floor = Math.pow(10, -digits);
  return frac != null && frac * 100 > 0 && frac * 100 < floor
    ? `< ${floor.toFixed(digits)}%`
    : fmtPct(frac, digits);
};

// статусы протокола на человеческий язык.
const STATUS_LABEL = {
  strong: "сильный перевес",
  moderate: "перевес",
  fail: "проверка не пройдена",
  inconclusive: "нет ответа",
};
const STATUS_TONE = { strong: "success", moderate: "warning", fail: "warning", inconclusive: "warning" };
const statusText = (s) => STATUS_LABEL[s] || s;

// заголовки карточек-контролей по стабильному id (в данных описания на английском).
const CONTROL_TITLE = {
  taras_control_annenkov_holdout_v2_fw_2000: "Путевые записки Анненкова (спрятаны)",
  taras_control_shinel_holdout_v2_fw_2000: "«Шинель» Гоголя (спрятана)",
  taras_control_gogol1835_base_v2_fw_2000: "«Тарас Бульба», редакция 1835 (база)",
  taras_control_turgenev_holdout_v2_fw_2000: "«Отцы и дети» Тургенева (спрятаны)",
};

const sha = (s) => `${s.slice(0, 10)}…${s.slice(-6)}`;

function WinnerLine({ row }) {
  if (!row.gatePass) {
    return (
      <p className="note" style={{ margin: 0 }}>
        Порог надёжности не пройден — кому принадлежит текст, на этой панели не определяем.
      </p>
    );
  }
  // один кусок не даёт «доли» — показываем только направление, без 100%-шкалы,
  // чтобы единственный кусок не читался как сильное свидетельство.
  if (row.targetChunks === 1) {
    return (
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
        <span style={{ fontSize: 16, color: "var(--text-muted)" }}>единственный кусок — только направление</span>
        <span className="mono" style={{ fontSize: 16, color: "var(--text)" }}>ближе к {nmDat(row.top)}</span>
      </div>
    );
  }
  const share = row.winnerShare[row.top] ?? 0;
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
        <span style={{ fontSize: 16, color: "var(--text-muted)" }}>доля отрывков к ближайшему профилю</span>
        <span className="mono" style={{ fontSize: 16, color: "var(--text)" }}>{nm(row.top)} · {fmtPct(share, 1)}</span>
      </div>
      <MeterBar value={share} max={1} accent="var(--icon-blue)" />
    </div>
  );
}

function ResultCard({ row, title, accent = "var(--icon-blue)" }) {

  return (
    <EditorialPanel>
      <div style={{ display: "grid", gap: 13 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h4 style={{ margin: 0, color: "var(--text)" }}>{title}</h4>
          <Badge tone={STATUS_TONE[row.status] || "warning"}>{statusText(row.status)}</Badge>
        </div>
        <div className="grid cols-3" style={{ gap: 10 }}>
          <EditorialMetric label="надёжность панели" value={fmtScore(row.gate, 4)} accent={row.gatePass ? "var(--success)" : "var(--cinnabar)"} hint="узнаёт ли панель известных авторов; нужно ≥ 0.80" />
          <EditorialMetric label="случайность" value={fmtP(row.p)} accent="var(--gold)" hint="вероятность такого совпадения при случайной перетасовке" />
          <EditorialMetric label="оценка доводов" value={fmtScore(row.score, 1)} accent={accent} hint="сводный балл по правилам этой проверки; это не вероятность авторства" />
        </div>
        <WinnerLine row={row} />
        <div className="mono" style={{ display: "grid", gap: 5, fontSize: 16, color: "var(--text-muted)" }}>
          <span>кусков текста: {row.targetChunks}</span>
          <span>разница оценок ближайших профилей: {fmtScore(row.margin, 4)}</span>
          <span>по кускам: {Object.entries(row.perChunk).map(([k, v]) => `${nm(k)} ${v}/${row.targetChunks}`).join(", ") || "—"}</span>
        </div>
      </div>
    </EditorialPanel>
  );
}

function ControlCard({ row }) {
  const share = row.winnerShare[row.top] ?? 0;
  return (
    <EditorialPanel>
      <div style={{ display: "grid", gap: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <span style={{ color: "var(--text)", fontSize: 17 }}>{CONTROL_TITLE[row.id] || row.target}</span>
          <Badge tone={STATUS_TONE[row.status] || "warning"}>{statusText(row.status)}</Badge>
        </div>
        <span className="mono" style={{ fontSize: 16, color: "var(--text-muted)" }}>
          → {nm(row.top)} · {fmtPct(share, 1)} · запас {fmtScore(row.margin, 4)}
        </span>
      </div>
    </EditorialPanel>
  );
}

export default function Taras() {
  const [strict, loose] = TARAS.headline;
  const [sameStrict, sameLoose] = TARAS.samePeriod;
  const [extStrict, extLoose] = TARAS.extended;
  const R = TARAS.replication;
  const E = TARAS.extraction;
  const AB = TARAS.annenkovBinary;
  const annChunks = AB.perChunk[AB.top] ?? Object.values(AB.perChunk)[0];
  const PA = TARAS.postAudit;
  if (PA) {
    const annShare = PA.annenkovBinary.winner_share.gogol ?? 0;
    const somovShare = PA.somovBinary.winner_share.somov ?? 0;
    return (
      <section className="section" id="hohol">
        <div className="wrap flow">
          <div className="section-head reveal">
            <p className="eyebrow">Две редакции повести</p>
            <h1>«Тарас Бульба»: сравнение добавлений</h1>
            <p className="prose lead muted">
              Сравниваем добавления к редакции 1842 года с прозой Гоголя и его
              современников. Вопрос об участии Анненкова и Прокоповича требует
              отделить сходство авторской манеры от общей литературной нормы эпохи.
            </p>
            <div className="grid cols-4 reveal" style={{ maxWidth: 900 }}>
              <EditorialMetric label="слов в строгом наборе" value={fmtInt(TARAS.manifest.strictWords)} accent="var(--icon-blue)" />
              <EditorialMetric label="слов в широком наборе" value={fmtInt(TARAS.manifest.looseWords)} accent="var(--cosmos)" />
              <EditorialMetric label="узнаваемость группы подозреваемых" value={fmtScore(PA.suspectsStrict.work_macro_recall, 4)} accent="var(--cinnabar)" hint="Ниже принятого порога 0.80: группа недостаточно надёжно узнаёт известные тексты." />
              <EditorialMetric label="узнаваемость авторов эпохи" value={fmtScore(PA.samePeriodStrict.work_macro_recall, 4)} accent="var(--cinnabar)" hint="Ниже принятого порога 0.80." />
            </div>
          </div>
        <ArticleContents chapter="hohol" items={[
            ["hohol-section-1", "Что именно сравниваем"],
            ["hohol-section-2", "Проверка профилей с равным весом произведений"],
            ["hohol-section-3", "Парные сравнения: Гоголь, Анненков и Сомов"],
            ["hohol-section-4", "Состав кандидатов и набор признаков"]
          ]} />


          <div className="module reveal">
            <h3 id="hohol-section-1">Что именно сравниваем</h3>
            <p className="prose muted" style={{ maxWidth: "76ch" }}>
              Обе редакции взяты из академического издания. Алгоритм выделяет
              предложения, появившиеся в 1842 году и почти отсутствующие в версии
              1835-го. Строгий и широкий наборы задают два способа выделения добавлений.
              Служебные слова — союзы, частицы и предлоги — уменьшают прямое влияние
              тематической лексики, хотя также зависят от жанра. «Тарас Бульба»
              исключён из эталонной прозы Гоголя.
            </p>
          </div>

          <div className="module reveal">
            <h3 id="hohol-section-2">Проверка профилей с равным весом произведений</h3>
            <p className="prose muted" style={{ maxWidth: "76ch" }}>
              Для каждого произведения строится отдельный профиль; профиль автора
              усредняет произведения с равным весом. При сравнении ближайших профилей
              группа кандидатов узнаёт контрольные тексты с
              результатом {fmtScore(PA.suspectsStrict.work_macro_recall, 4)}, а группа
              авторов той же эпохи — {fmtScore(PA.samePeriodStrict.work_macro_recall, 4)}.
              Это средние по авторам доли верно распознанных произведений. Обе ниже
              рабочего порога 0.80, поэтому назначения добавлений на этих панелях
              остаются диагностикой.
            </p>
          </div>

          <div className="module reveal">
            <h3 id="hohol-section-3">Парные сравнения: Гоголь, Анненков и Сомов</h3>
            <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
              <EditorialPanel>
                <h4 style={{ color: "var(--text)" }}>Гоголь против Анненкова</h4>
                <p className="prose muted">
                  Контрольные тексты различаются надёжно ({fmtScore(PA.annenkovBinary.work_macro_recall, 3)}).
                  Добавления ближе к {nm(PA.annenkovBinary.top)}: туда уходят{" "}
                  {fmtPct(annShare, 1)} отрывков.
                </p>
              </EditorialPanel>
              <EditorialPanel>
                <h4 style={{ color: "var(--text)" }}>Гоголь против Сомова</h4>
                <p className="prose muted">
                  Эта пара тоже хорошо различается ({fmtScore(PA.somovBinary.work_macro_recall, 3)}),
                  но добавления ближе к {nm(PA.somovBinary.top)}:{" "}
                  {fmtPct(somovShare, 1)} отрывков.
                </p>
              </EditorialPanel>
            </div>
            <p className="callout">
              Версия «все большие вставки написал Анненков» не подтверждается. Но
              победитель меняется вместе с кругом сравнения: одна пара указывает на
              Гоголя, другая — на Сомова. Значит, единственного автора эти данные не называют.
            </p>
          </div>

          <div className="module reveal">
            <h3 id="hohol-section-4">Состав кандидатов и набор признаков</h3>
            <p className="verdict">
              Данные возражают против простой версии «всё написал Анненков», но не
              дают устойчивого выбора между Гоголем, Сомовым и другими авторами эпохи.
              В методе Delta фиксированный список служебных слов и список частых слов,
              выбранных по обучающим текстам, дают в паре Гоголь–Сомов разных лидеров: {nm(PA.delta.somovBinaryFw.targets.strict_additions.top)} и{" "}
              {nm(PA.delta.somovBinaryMfw.targets.strict_additions.top)}. Прозы
              Прокоповича почти не сохранилось, поэтому его участие напрямую
              проверить нельзя. Этот набор сравнений не даёт устойчивого выбора автора.
            </p>
          </div>

          <Sources
            items={[
              { cite: "Сравнение с равным весом книг: 16 проверок по 2000 перестановок", url: "https://github.com/dmitrymake/stylometric/blob/main/docs/cases/work_balanced_audit/README.md" },
              { cite: "Проверка методом Delta с равным весом книг", url: "https://github.com/dmitrymake/stylometric/blob/main/docs/cases/work_balanced_audit/custom/taras_delta_full_refit_work_balanced.json" },
              { cite: "«Тарас Бульба», редакция 1835 года — ФЭБ", url: "https://feb-web.ru/feb/gogol/texts/gtb/gtb-097-.htm" },
              { cite: "«Тарас Бульба», редакция 1842 года — ФЭБ", url: "https://feb-web.ru/feb/gogol/texts/gtb/gtb-005-.htm" },
            ]}
            note="Каждая книга получает одинаковый вес. Эти панели используют собственный корпус прозы XIX века и не входят в проверку на 248 произведениях."
          />
        </div>
      </section>
    );
  }
  return (
    <section className="section" id="hohol">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Разбор · вторая редакция «Тараса Бульбы»</p>
          <h1>«Тарас Бульба»: сравнение добавлений</h1>
          <p className="prose lead muted">
            Между первой редакцией повести (1835) и школьной (1842) — большая разница. Текст вырос
            почти вдвое, изменился по объёму и устройству повествования, в нём появилась речь о товариществе. Гоголь тогда
            жил за границей. Книгу в Петербурге готовил его друг Николай Прокопович. А в Риме под
            гоголевскую диктовку набело писал Павел Анненков — его рукой выведена беловая рукопись
            «Мёртвых душ». Отсюда стойкая версия: большие патриотические вставки в «Тараса Бульбу»
            написал тот же человек — их сочинил <em>не Гоголь</em>. Названы конкретные
            люди — версию можно проверить напрямую: сравнить добавления с тем, что эти люди
            писали сами.
          </p>
          <div className="grid cols-4 reveal" style={{ maxWidth: 860 }}>
            <EditorialMetric label="строгий набор, слов" value={fmtInt(TARAS.manifest.strictWords)} accent="var(--icon-blue)" hint="слова, вошедшие в узкое выделение добавлений" />
            <EditorialMetric label="широкий набор, слов" value={fmtInt(TARAS.manifest.looseWords)} accent="var(--cosmos)" hint="слова более широкого выделения добавлений" />
            <EditorialMetric label="надёжность панели" value={fmtScore(strict.gate, 4)} accent="var(--success)" hint="панель узнаёт известных авторов выше порога 0.80" />
            <EditorialMetric label="случайность" value={fmtP(strict.p)} accent="var(--gold)" hint="вероятность такого результата при случайной перетасовке" />
          </div>
        </div>
        <ArticleContents chapter="hohol" items={[
            ["hohol-section-5", "Что именно дописали"],
            ["hohol-section-6", "Проверка №1 · Сравнение с Анненковым"],
            ["hohol-section-7", "Проверка №2 · Ответы на контрольных произведениях"],
            ["hohol-section-8", "Проверка №3 · Так писала вся эпоха?"],
            ["hohol-section-9", "Почему всплыл Сомов"],
            ["hohol-section-10", "Прокопович: ограничение эталонных текстов"],
            ["hohol-section-11", "Речь о товариществе"],
            ["hohol-section-12", "Панель поздних классиков"],
            ["hohol-section-13", "Итог"]
          ]} />


          {/* 1. Сам текст добавлений */}
        <div className="module reveal">
          <h3 id="hohol-section-5">Что именно дописали</h3>
          <p className="prose muted" style={{ maxWidth: "72ch" }}>
            Обе редакции взяты целиком из академического издания (ФЭБ, изд. АН СССР):
            редакция 1842 года — {fmtInt(E.edition1842Words)} {ruWords(E.edition1842Words)}, редакция 1835-го —{" "}
            {fmtInt(E.edition1835Words)}. Добавлениями считаем предложения 1842 года, которых нет
            в 1835-м. Выделение проверила машина: все выделенные добавления есть в тексте 1842 года
            и почти отсутствуют в тексте 1835-го —{" "}
            <strong style={{ color: "var(--text)" }}>{fmtPctFloor(E.strictIn1835, 2)}</strong>.
            Получилось два набора добавлений: строгий ({fmtInt(TARAS.manifest.strictWords)} {ruWords(TARAS.manifest.strictWords)})
            и широкий ({fmtInt(TARAS.manifest.looseWords)}). Оба идут через один протокол. Сравниваем
            по служебным словам — союзам, частицам, предлогам. Это уменьшает прямое влияние
            тематической лексики, но не исключает жанровых различий. Почерк Гоголя для сравнения собираем{" "}
            <strong style={{ color: "var(--text)" }}>без</strong> самого «Тараса Бульбы».
          </p>
        </div>

        {/* 2. Главный подозреваемый */}
        <div className="module reveal">
          <h3 id="hohol-section-6">Проверка №1 · Сравнение с Анненковым</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Анненков переписывал гоголевские рукописи римского периода. Подозрение простое: в добавления
            «Тараса Бульбы» он вписал уже своё. Его
            собственной прозы 1840-х сохранилось много: «Письма из-за границы» выходили в те же
            годы, что и добавления. Сравниваем добавления с Гоголем и Анненковым — сначала только
            эти двое, потом добавляем для контроля Тургенева и Достоевского. Гоголя и Анненкова
            метод различает безошибочно в этой контрольной подборке ({fmtScore(AB.gate, 2)} из {fmtScore(1, 2)}).
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={strict} title="Строгий набор · панель подозреваемых" />
            <ResultCard row={loose} title="Широкий набор · панель подозреваемых" accent="var(--cosmos)" />
          </div>
          <p className="callout">
            В парном сравнении с Анненковым ближе профиль Гоголя ({annChunks} из {AB.targetChunks}).
            Многоавторная панель ближайших профилей с равным весом произведений не проходит
            контроль качества; парное сравнение с Сомовым даёт другое направление. Ответ зависит от состава сравнения.
          </p>
        </div>

        {/* 3. Контроль честности */}
        <div className="module reveal">
          <h3 id="hohol-section-7">Проверка №2 · Ответы на контрольных произведениях</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Честное возражение: инструмент выбирает Гоголя для любого текста?
            Проверяем на четырёх задачах с заранее известными ответами: прячем от эталона по одной
            работе и смотрим, вернётся ли она к своему автору.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            {TARAS.controls.map((row) => <ControlCard key={row.id} row={row} />)}
          </div>
          <p className="callout">
            Все четыре контрольных произведения отнесены к своим авторам.
            Этот результат описывает выбранные контроли и не заменяет проверку всей панели.
          </p>
        </div>

        {/* 4. Панель эпохи */}
        <div className="module reveal">
          <h3 id="hohol-section-8">Проверка №3 · Так писала вся эпоха?</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Расширяем круг: Пушкин, Соллогуб, Анненков, Гребёнка — проза тех же 1830–40-х. Если
            добавления написаны «просто языком эпохи», на широкой панели они расползутся между
            авторами.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={sameStrict} title="Строгий набор · панель эпохи" />
            <ResultCard row={sameLoose} title="Широкий набор · панель эпохи" accent="var(--cosmos)" />
          </div>
          <p className="callout">
            При равном весе произведений средняя полнота этой панели по авторам равна 0.7876,
            ниже порога 0.80. Назначения добавлений на ней остаются диагностикой.
          </p>
        </div>

        {/* 5. Панель, которая показала не туда */}
        <div className="module reveal">
          <h3 id="hohol-section-9">Почему всплыл Сомов</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            На панели из казачьей и украинской прозы
            (Орест Сомов, Нарежный, Гребёнка) добавления потянулись к Сомову — с очень маленьким
            запасом. Разбираемся. Первое: бесспорный текст 1835 года на <em>той же</em> панели идёт
            к Гоголю. Значит, панель умеет узнавать Гоголя в казачьем материале. Просто добавления звучат
            «сказовее» базового текста — переработка 1839–1842 годов сделала повесть
            эпичнее. Метод Delta с нормировкой по выбранным словам даёт иной результат.
            Его ответ также зависит от выбора слов и состава панели.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={TARAS.topic.additions} title="Строгий набор · казачья панель" accent="var(--cinnabar)" />
            <ResultCard row={TARAS.topic.base1835} title="«Тарас Бульба» 1835 · та же панель" />
          </div>
          <p className="verdict">
            В Delta словарь и нормировка строятся заново при каждой перестановке меток.
            На панели кандидатов этот протокол проходит контроль и выбирает Гоголя.
            В паре Гоголь–Сомов фиксированный список служебных слов выбирает Гоголя,
            а частые слова из обучающего корпуса — Сомова. Наборы признаков не дают устойчивого выбора.
          </p>
        </div>

        {/* 6. Второй подозреваемый */}
        <div className="module reveal">
          <h3 id="hohol-section-10">Прокопович: ограничение эталонных текстов</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Редактора издания 1842 года напрямую проверить не выйдет: своей прозы Прокопович почти
            не оставил (стихи и пара писем). Это честное ограничение. Его письма 1843 года сравниваются с Гоголем и Анненковым. Фрагмент один и короткий — поэтому
            только направление, без вывода.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={TARAS.prokopovich} title="Письма Прокоповича, 1843" accent="var(--gold)" />
            <div className="note" style={{ margin: 0 }}>
              Есть и прямое свидетельство из переписки. Получив издание 1842 года, Гоголь жаловался Прокоповичу
              на ошибки набора — то есть <em>внимательно вычитывал</em> итоговый текст. (Опечатки набора
              — это не то же самое, что редактура содержания.) Сравнение профилей не
              устанавливает единственного автора больших вставок. Участие Прокоповича
              нельзя оценить по сопоставимой подборке собственной прозы.
            </div>
          </div>
        </div>

        {/* 7. Речь о товариществе */}
        <div className="module reveal">
          <h3 id="hohol-section-11">Речь о товариществе</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Самый цитируемый фрагмент добавлений — речь Тараса о товариществе — выносим отдельной
            целью: именно его чаще всего называют «идеологической вставкой». Но фрагмент короткий
            ({fmtInt(TARAS.manifest.speechWords)} {ruWords(TARAS.manifest.speechWords)} — всего один кусок для анализа), поэтому по
            правилам протокола он получает только диагностический статус, не сильный вердикт.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={TARAS.speech} title="Речь о товариществе" accent="var(--gold)" />
            <div className="note" style={{ margin: 0 }}>
              Речь представлена одним фрагментом. Многоавторная панель ближайших профилей
              с равным весом произведений не проходит контроль качества, поэтому
              её назначение не служит основанием для атрибуции.
            </div>
          </div>
        </div>

        {/* 8. Панель-неудача */}
        <div className="module reveal">
          <h3 id="hohol-section-12">Панель поздних классиков</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Стресс-панель с Толстым, Лесковым и Салтыковым-Щедриным (авторы более поздней эпохи) не
            берёт порог надёжности, поэтому её результаты не читаем — показываем как есть. Панель,
            которая не прошла проверку, не превращается в вывод.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={extStrict} title="Строгий набор · поздние классики" accent="var(--cinnabar)" />
            <ResultCard row={extLoose} title="Широкий набор · поздние классики" accent="var(--cinnabar)" />
          </div>
        </div>

        {/* 9. Итог */}
        <div className="module reveal">
          <h3 id="hohol-section-13">Итог</h3>
          <p className="verdict">
            {TARAS.claim} Речь о <strong style={{ color: "var(--text)" }}>больших
            добавленных пассажах</strong> — точечную редакторскую правку, замену отдельных слов и
            орфографию этот протокол не проверяет. Прокопович проверен лишь косвенно: его прозы не
            сохранилось. Статусы карточек описывают диагностическое сравнение фрагментов внутри выбранной панели.
          </p>
        </div>

        <Sources
          items={[
            { cite: "«Тарас Бульба», редакция 1835 года — ФЭБ, издание АН СССР", url: "https://feb-web.ru/feb/gogol/texts/gtb/gtb-097-.htm" },
            { cite: "«Тарас Бульба», редакция 1842 года — ФЭБ, издание АН СССР", url: "https://feb-web.ru/feb/gogol/texts/gtb/gtb-005-.htm" },
            { cite: "Проза Анненкова — az.lib.ru", url: "http://az.lib.ru/a/annenkow_p_w/" },
            { cite: "Проза Сомова — az.lib.ru", url: "http://az.lib.ru/s/somow_o_m/" },
            { cite: "Проза Нарежного — az.lib.ru", url: "http://az.lib.ru/n/narezhnyj_w/" },
            { cite: "Проза Гребёнки — az.lib.ru", url: "http://az.lib.ru/g/grebenka_e_p/" },
            { cite: `Контрольные суммы наборов: строгий ${sha(TARAS.manifest.strictSha)}, широкий ${sha(TARAS.manifest.looseSha)}` },
            { cite: "Описание кейса, протокол и машинные артефакты", url: "https://github.com/dmitrymake/stylometric/blob/main/docs/cases/taras_hardened/README.md" },
          ]}
          note="Сами тексты в репозиторий не входят — публикуются контрольные суммы и результаты проверок."
        />
      </div>
    </section>
  );
}
