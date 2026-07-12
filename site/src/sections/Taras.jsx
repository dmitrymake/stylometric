import { Card, Stat, Badge } from "@dmitrymake/rk-ui";
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
  strong: "уверенно",
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
        <span style={{ fontSize: 12.5, color: "var(--text-muted)" }}>единственный кусок — только направление</span>
        <span className="mono" style={{ fontSize: 13, color: "var(--text)" }}>ближе к {nmDat(row.top)}</span>
      </div>
    );
  }
  const share = row.winnerShare[row.top] ?? 0;
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
        <span style={{ fontSize: 12.5, color: "var(--text-muted)" }}>доля кусков за победителя</span>
        <span className="mono" style={{ fontSize: 13, color: "var(--text)" }}>{nm(row.top)} · {fmtPct(share, 1)}</span>
      </div>
      <MeterBar value={share} max={1} accent="var(--icon-blue)" />
    </div>
  );
}

function ResultCard({ row, title, accent = "var(--icon-blue)" }) {
  const ci = row.ci.length ? `[${fmtScore(row.ci[0], 4)}, ${fmtScore(row.ci[1], 4)}]` : "—";
  return (
    <Card padding={22}>
      <div style={{ display: "grid", gap: 13 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h4 style={{ margin: 0, color: "var(--text)" }}>{title}</h4>
          <Badge tone={STATUS_TONE[row.status] || "warning"}>{statusText(row.status)}</Badge>
        </div>
        <div className="grid cols-3" style={{ gap: 10 }}>
          <Stat label="надёжность панели" value={fmtScore(row.gate, 4)} accent={row.gatePass ? "var(--success)" : "var(--cinnabar)"} hint="узнаёт ли панель известных авторов; нужно ≥ 0.80" />
          <Stat label="случайность" value={fmtP(row.p)} accent="var(--gold)" hint="вероятность такого совпадения при случайной перетасовке" />
          <Stat label="оценка доводов" value={fmtScore(row.score, 1)} accent={accent} hint="сводная сила доводов по шкале 0–100. Высокая оценка не равна статусу «уверенно»: статус смотрит ещё на запас за победителя и на устойчивость" />
        </div>
        <WinnerLine row={row} />
        <div className="mono" style={{ display: "grid", gap: 5, fontSize: 12.5, color: "var(--text-muted)" }}>
          <span>кусков текста: {row.targetChunks}</span>
          <span>запас за победителя: {fmtScore(row.margin, 4)} · интервал {ci}</span>
          <span>по кускам: {Object.entries(row.perChunk).map(([k, v]) => `${nm(k)} ${v}/${row.targetChunks}`).join(", ") || "—"}</span>
        </div>
      </div>
    </Card>
  );
}

function ControlCard({ row }) {
  const share = row.winnerShare[row.top] ?? 0;
  return (
    <Card padding={16}>
      <div style={{ display: "grid", gap: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
          <span style={{ color: "var(--text)", fontSize: 14 }}>{CONTROL_TITLE[row.id] || row.target}</span>
          <Badge tone={STATUS_TONE[row.status] || "warning"}>{statusText(row.status)}</Badge>
        </div>
        <span className="mono" style={{ fontSize: 12.5, color: "var(--text-muted)" }}>
          → {nm(row.top)} · {fmtPct(share, 1)} · запас {fmtScore(row.margin, 4)}
        </span>
      </div>
    </Card>
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
            <p className="eyebrow">Adversarial audit · вторая редакция «Тараса Бульбы»</p>
            <h2>Кто дописал «Тараса Бульбу»?</h2>
            <p className="verdict">
              Текущая батарея <strong style={{ color: "var(--text)" }}>не устанавливает единственную руку</strong>.
              Старый гоголевский headline снят после обнаружения train-side pseudoreplication: длинные произведения
              раньше сильнее тянули авторский центроид числом своих кусков. Теперь каждая работа имеет один равный вес.
            </p>
            <div className="grid cols-4 reveal" style={{ maxWidth: 900 }}>
              <Stat label="панель подозреваемых" value={fmtScore(PA.suspectsStrict.work_macro_recall, 4)} accent="var(--cinnabar)" hint="ниже обязательного порога 0.80" />
              <Stat label="панель эпохи" value={fmtScore(PA.samePeriodStrict.work_macro_recall, 4)} accent="var(--cinnabar)" hint="ниже обязательного порога 0.80" />
              <Stat label="Гоголь–Анненков" value={fmtScore(PA.annenkovBinary.work_macro_recall, 3)} accent="var(--success)" hint="бинарная панель проходит gate" />
              <Stat label="Гоголь–Сомов" value={fmtScore(PA.somovBinary.work_macro_recall, 3)} accent="var(--success)" hint="бинарная панель проходит gate" />
            </div>
          </div>

          <div className="module reveal">
            <h3>Что изменил аудит</h3>
            <p className="prose muted" style={{ maxWidth: "76ch" }}>
              При неизменных текстах, окнах, признаках и seed многоавторная панель подозреваемых падает до {" "}
              <strong style={{ color: "var(--text)" }}>{fmtScore(PA.suspectsStrict.work_macro_recall, 4)}</strong>,
              а панель авторов той же эпохи — до {" "}
              <strong style={{ color: "var(--text)" }}>{fmtScore(PA.samePeriodStrict.work_macro_recall, 4)}</strong>.
              Малые permutation p не спасают результат: prereg требует одновременно p ≤ 0.05 и recall ≥ 0.80.
              Поэтому сохранённые top=Gogol на этих панелях не являются атрибуцией.
            </p>
          </div>

          <div className="module reveal">
            <h3>Почему бинарные проверки не дают имени автора</h3>
            <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
              <Card padding={22}>
                <h4 style={{ color: "var(--text)" }}>Гоголь против Анненкова</h4>
                <p className="prose muted">Gate {fmtScore(PA.annenkovBinary.work_macro_recall, 3)}; цель → {nm(PA.annenkovBinary.top)}, {fmtPct(annShare, 1)} кусков.</p>
              </Card>
              <Card padding={22}>
                <h4 style={{ color: "var(--text)" }}>Гоголь против Сомова</h4>
                <p className="prose muted">Gate {fmtScore(PA.somovBinary.work_macro_recall, 3)}; цель → {nm(PA.somovBinary.top)}, {fmtPct(somovShare, 1)} кусков.</p>
              </Card>
            </div>
            <p className="callout">
              Прямая версия «всё написал Анненков» не поддерживается, но выбор победителя меняется вместе с
              составом альтернатив. Это evidence против уникальной закрытой атрибуции, а не доказательство Гоголя или Сомова.
            </p>
          </div>

          <div className="module reveal">
            <h3>Научный статус</h3>
            <p className="verdict">
              {PA.conclusion} Пересчёт exploratory: он исправляет обнаруженную после результата ошибку estimand,
              но не превращается задним числом в предрегистрацию. Исправленный Delta с полным model-refit null
              ведёт к Гоголю на suspects, однако в прошедшей binary Гоголь–Сомов меняет ответ по признаку:
              fixed FW → {nm(PA.delta.somovBinaryFw.targets.strict_additions.top)}, MFW → {nm(PA.delta.somovBinaryMfw.targets.strict_additions.top)}.
              Поэтому cross-feature устойчивости нет; исторические паспорта и старый Delta-отчёт сохранены как legacy.
            </p>
          </div>

          <Sources
            items={[
              { cite: "Парный work-balanced аудит: 16 кейсов, по 2000 перестановок", url: "https://github.com/dmitrymake/stylometric/blob/main/docs/cases/work_balanced_audit/README.md" },
              { cite: "Исправленный Delta: equal-work centroids и full-refit permutation null", url: "https://github.com/dmitrymake/stylometric/blob/main/docs/cases/work_balanced_audit/custom/taras_delta_full_refit_work_balanced.json" },
              { cite: "«Тарас Бульба», редакция 1835 года — ФЭБ", url: "https://feb-web.ru/feb/gogol/texts/gtb/gtb-097-.htm" },
              { cite: "«Тарас Бульба», редакция 1842 года — ФЭБ", url: "https://feb-web.ru/feb/gogol/texts/gtb/gtb-005-.htm" },
            ]}
            note={`Формула центроида: ${PA.centroidWeighting}. Дата аудита: ${PA.date}.`}
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
          <h2>Кто дописал «Тараса Бульбу»?</h2>
          <p className="prose lead muted">
            Между первой редакцией повести (1835) и школьной (1842) — большая разница. Текст вырос
            почти вдвое, стал эпичнее и «русее», в нём появилась речь о товариществе. Гоголь тогда
            жил за границей. Книгу в Петербурге готовил его друг Николай Прокопович. А в Риме под
            гоголевскую диктовку набело писал Павел Анненков — его рукой выведена беловая рукопись
            «Мёртвых душ». Отсюда стойкая версия: большие патриотические вставки в «Тараса Бульбу»
            написал тот же человек — их сочинил <em>не Гоголь</em>. Названы конкретные
            люди — версию можно проверить напрямую: сравнить добавления с тем, что эти люди
            писали сами.
          </p>
          <div className="grid cols-4 reveal" style={{ maxWidth: 860 }}>
            <Stat label="строгий набор, слов" value={fmtInt(TARAS.manifest.strictWords)} accent="var(--icon-blue)" hint="слова, вошедшие в узкое выделение добавлений" />
            <Stat label="широкий набор, слов" value={fmtInt(TARAS.manifest.looseWords)} accent="var(--cosmos)" hint="слова более широкого выделения добавлений" />
            <Stat label="надёжность панели" value={fmtScore(strict.gate, 4)} accent="var(--success)" parade hint="панель узнаёт известных авторов выше порога 0.80" />
            <Stat label="случайность" value={fmtP(strict.p)} accent="var(--gold)" hint="вероятность такого результата при случайной перетасовке" />
          </div>
        </div>

        {/* 1. Сам текст добавлений */}
        <div className="module reveal">
          <h3>Что именно дописали</h3>
          <p className="prose muted" style={{ maxWidth: "72ch" }}>
            Обе редакции взяты целиком из академического издания (ФЭБ, изд. АН СССР):
            редакция 1842 года — {fmtInt(E.edition1842Words)} {ruWords(E.edition1842Words)}, редакция 1835-го —{" "}
            {fmtInt(E.edition1835Words)}. Добавлениями считаем предложения 1842 года, которых нет
            в 1835-м. Выделение проверила машина: все выделенные добавления есть в тексте 1842 года
            и почти отсутствуют в тексте 1835-го —{" "}
            <strong style={{ color: "var(--text)" }}>{fmtPctFloor(E.strictIn1835, 2)}</strong>.
            Получилось два набора добавлений: строгий ({fmtInt(TARAS.manifest.strictWords)} {ruWords(TARAS.manifest.strictWords)})
            и широкий ({fmtInt(TARAS.manifest.looseWords)}). Оба идут через один протокол. Сравниваем
            только по служебным словам — союзам, частицам, предлогам (они не зависят от того,{" "}
            <em>о чём</em> текст). Почерк Гоголя для сравнения собираем{" "}
            <strong style={{ color: "var(--text)" }}>без</strong> самого «Тараса Бульбы».
          </p>
        </div>

        {/* 2. Главный подозреваемый */}
        <div className="module reveal">
          <h3>Проверка №1 · Главный подозреваемый Анненков</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Анненков переписывал гоголевские рукописи римского периода. Подозрение простое: в добавления
            «Тараса Бульбы» он вписал уже своё. Его
            собственной прозы 1840-х сохранилось много: «Письма из-за границы» выходили в те же
            годы, что и добавления. Сравниваем добавления с Гоголем и Анненковым — сначала только
            эти двое, потом добавляем для контроля Тургенева и Достоевского. Гоголя и Анненкова
            метод различает безошибочно (надёжность {fmtScore(AB.gate, 2)} из {fmtScore(1, 2)}),
            так что сравнение честное.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={strict} title="Строгий набор · панель подозреваемых" />
            <ResultCard row={loose} title="Широкий набор · панель подозреваемых" accent="var(--cosmos)" />
          </div>
          <p className="callout">
            Этот блок сохранён только как legacy-иллюстрация: бинарная пара с Анненковым ведёт к
            Гоголю ({annChunks} из {AB.targetChunks}), но исправленная полная панель не проходит gate,
            а другая валидная binary ведёт к Сомову. Уникального вывода нет.
          </p>
        </div>

        {/* 3. Контроль честности */}
        <div className="module reveal">
          <h3>Проверка №2 · Не «прилипает» ли метод к Гоголю?</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Честное возражение: инструмент просто отдаёт всё спорное самому известному автору?
            Проверяем на четырёх задачах с заранее известными ответами: прячем от эталона по одной
            работе и смотрим, вернётся ли она к своему автору.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            {TARAS.controls.map((row) => <ControlCard key={row.id} row={row} />)}
          </div>
          <p className="callout">
            Четыре из четырёх вернулись к своим. Спрятанная проза Анненкова уверенно приходит обратно
            к Анненкову — метод умеет говорить «не Гоголь», когда это правда.
          </p>
        </div>

        {/* 4. Панель эпохи */}
        <div className="module reveal">
          <h3>Проверка №3 · Так писала вся эпоха?</h3>
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
            Legacy-направление было к Гоголю, но work-balanced gate этой панели равен 0.7876.
            Поэтому target после аудита не интерпретируется.
          </p>
        </div>

        {/* 5. Панель, которая показала не туда */}
        <div className="module reveal">
          <h3>Почему всплыл Сомов</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Одна проверка сначала показала не на Гоголя. На панели из казачьей и украинской прозы
            (Орест Сомов, Нарежный, Гребёнка) добавления потянулись к Сомову — с очень маленьким
            запасом. Разбираемся. Первое: бесспорный текст 1835 года на <em>той же</em> панели идёт
            к Гоголю. Значит, панель умеет узнавать Гоголя в казачьем материале. Просто добавления звучат
            «сказовее» базового текста — переработка 1839–1842 годов сделала повесть
            эпичнее. Второе: считаем то же самое другим методом (Burrows Delta, другая
            математика) — и сомовский результат исчезает.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={TARAS.topic.additions} title="Строгий набор · казачья панель" accent="var(--cinnabar)" />
            <ResultCard row={TARAS.topic.base1835} title="«Тарас Бульба» 1835 · та же панель" />
          </div>
          <p className="verdict">
            Исторический Delta-отчёт был невалиден как permutation null. Исправленный full-refit
            rerun проходит на suspects и ведёт к Гоголю, но в binary Гоголь–Сомов меняет top:
            fixed FW → Гоголь, learned MFW → Сомов. Cross-feature устойчивости нет.
          </p>
        </div>

        {/* 6. Второй подозреваемый */}
        <div className="module reveal">
          <h3>Второй подозреваемый · Прокопович</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Редактора издания 1842 года напрямую проверить не выйдет: своей прозы Прокопович почти
            не оставил (стихи и пара писем). Это честное ограничение. Что можно — сделали: сравнили его
            письма 1843 года с Гоголем и Анненковым. Фрагмент один и короткий — поэтому
            только направление, без вывода.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={TARAS.prokopovich} title="Письма Прокоповича, 1843" accent="var(--gold)" />
            <div className="note" style={{ margin: 0 }}>
              Есть и прямое свидетельство из переписки. Получив издание 1842 года, Гоголь жаловался Прокоповичу
              на ошибки набора — то есть <em>внимательно вычитывал</em> итоговый текст. (Опечатки набора
              — это не то же самое, что редактура содержания.) Стилометрия после аудита не
              устанавливает единственную руку больших вставок; Прокопович остаётся
              documented-but-unmodelled кандидатом.
            </div>
          </div>
        </div>

        {/* 7. Речь о товариществе */}
        <div className="module reveal">
          <h3>Речь о товариществе</h3>
          <p className="prose muted" style={{ maxWidth: "72ch", marginBottom: 16 }}>
            Самый цитируемый фрагмент добавлений — речь Тараса о товариществе — выносим отдельной
            целью: именно его чаще всего называют «идеологической вставкой». Но фрагмент короткий
            ({fmtInt(TARAS.manifest.speechWords)} {ruWords(TARAS.manifest.speechWords)} — всего один кусок для анализа), поэтому по
            правилам протокола он получает только диагностический статус, не сильный вердикт.
          </p>
          <div className="grid cols-2" style={{ marginTop: "var(--beat-group)" }}>
            <ResultCard row={TARAS.speech} title="Речь о товариществе" accent="var(--gold)" />
            <div className="note" style={{ margin: 0 }}>
              Legacy-направление речи было к Гоголю, но это один кусок, а corrected
              multi-candidate gate не пройден. Атрибуционного вывода из него нет.
            </div>
          </div>
        </div>

        {/* 8. Панель-неудача */}
        <div className="module reveal">
          <h3>Что не сработало · Панель поздних классиков</h3>
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
          <h3>Итог</h3>
          <p className="verdict">
            {TARAS.claim} Речь о <strong style={{ color: "var(--text)" }}>больших
            добавленных пассажах</strong> — точечную редакторскую правку, замену отдельных слов и
            орфографию этот протокол не проверяет. Прокопович проверен лишь косвенно: его прозы не
            сохранилось. Статус «перевес» означает уверенное большинство кусков при положительном
            интервале запаса, но доля кусков за победителя не дотягивает до порога статуса «уверенно».
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
            { cite: "Полный отчёт по кейсу (спецификации, паспорта, аудит выделения)", url: "https://github.com/dmitrymake/stylometric/blob/main/docs/cases/taras_hardened/reports/dossier.md" },
          ]}
          note="Сами тексты в репозиторий не входят — публикуются контрольные суммы и результаты проверок."
        />
      </div>
    </section>
  );
}
