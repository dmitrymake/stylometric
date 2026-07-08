import { Stat, AnomalyGlyph } from "@dmitrymake/rk-ui";
import { SEGMENT } from "../data.js";
import { fmtScore } from "../format.js";
import MeterBar from "../components/MeterBar.jsx";

const S = SEGMENT;

// Отображаемые имена авторов — никаких сырых слагов базы в читаемом тексте.
const NAME = { krukov: "Крюков", serafimovich: "Серафимович", bunin: "Бунин", kuprin: "Куприн", chehov: "Чехов" };
const nm = (slug) => NAME[slug] || slug;

export default function ForeignHands() {
  // всё из данных: потолок ложных срабатываний по грубому правилу «три делить на число книг».
  const falseUpperPct = Math.ceil((3 / S.fpr.totalBooks) * 100);
  // единственная чисто донская пара среди похожих склеек — вычисляем из данных, не зашиваем в текст.
  const donPair = S.splices.similar.find((p) => p.host === "krukov" || p.intruder === "serafimovich");
  const donPairLabel = donPair ? `${nm(donPair.host)} и ${nm(donPair.intruder)}` : null;

  return (
    <section className="section" id="foreign">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">Проверка · чужая рука</p>
          <h2>Можно ли поймать чужую руку</h2>
          <p className="prose lead muted">
            Прежде чем искать чужую руку в спорной книге, детектор проходит две проверки. Книгу режут
            на отрывки по порядку. Каждому отрывку подбирают ближайшего по манере автора и смотрят,
            где почерк сменяется на чужой. Первая: подложить настоящую вставку — заметит ли шов?
            Вторая: дать цельную книгу одного автора — не выдумает ли соавтора, которого нет?
          </p>
        </div>

        <div className="split reveal" style={{ alignItems: "start" }}>
          {/* Позитив-контроль: склейки + кривая подмеса */}
          <div>
            <h3 style={{ marginTop: 0 }}>Заведомая склейка</h3>
            <p className="prose muted" style={{ marginBottom: 16 }}>
              Сшиваем встык две книги <strong style={{ color: "var(--text)" }}>разных</strong> авторов — детектор
              обязан нащупать шов. Повторяем на близких по манере парах, где стык почти незаметен.
            </p>
            <div className="grid cols-2" style={{ marginBottom: 22 }}>
              <Stat label="склейки разных авторов" value={`${S.recallDissimilar.detected}/${S.recallDissimilar.total}`} accent="var(--success)" parade hint="все склейки найдены" />
              <Stat label="склейки похожих авторов" value={`${S.recallSimilar.detected}/${S.recallSimilar.total}`} accent="var(--gold)" hint="шов найден во всех парах" />
            </div>
            <div className="mono muted" style={{ fontSize: 11, marginBottom: 8 }}>
              насколько малую вставку видно: в книгу одного автора подмешиваем всё больше кусков другого — и смотрим, сколько отрывков детектор назовёт «чужими»:
            </div>
            {S.admixture.map((a) => (
              <div key={a.pct} style={{ display: "grid", gridTemplateColumns: "5ch 1fr 6ch", alignItems: "center", gap: 8, padding: "2.5px 0" }}>
                <span className="mono" style={{ fontSize: 11, color: "var(--text-muted)" }}>{a.pct}%</span>
                <MeterBar value={a.foreign} accent={a.detected ? "var(--icon-blue)" : "var(--border-strong)"} />
                <span className="mono" style={{ fontSize: 10.5, color: a.detected ? "var(--icon-blue)" : "var(--text-muted)" }}>
                  {fmtScore(a.foreign)}{a.detected ? " — найден" : ""}
                </span>
              </div>
            ))}
            <p className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
              Для <em>разных</em> авторов хватает уже <strong style={{ color: "var(--text)" }}>{S.minDetectedAdmixPct}%</strong> подмеса.
              Чем больше вставка, тем больше отрывков детектор помечает — почти один к одному.
            </p>
          </div>

          {/* Негатив-контроль + потолок */}
          <div style={{ display: "grid", gap: 18, alignContent: "start" }}>
            <h3 style={{ marginTop: 0 }}>Цельная книга</h3>
            <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
              <AnomalyGlyph kind="relation_mismatch" size={40} />
              <div>
                <div className="bignum ring-num" style={{ color: "var(--success)" }}>
                  {S.fpr.falseBooks}<span style={{ fontSize: ".4em", color: "var(--text-muted)" }}>/{S.fpr.totalBooks}</span>
                </div>
                <div className="mono muted" style={{ fontSize: 12 }}>одноавторских книг, где нашёлся ложный «чужой» участок</div>
              </div>
            </div>
            <p className="verdict" style={{ margin: 0 }}>
              На <strong style={{ color: "var(--text)" }}>{S.fpr.totalBooks} заведомо одноавторских</strong> книгах —
              <strong style={{ color: "var(--success)" }}> ноль</strong> ложных участков (0 из {S.fpr.totalBooks}).
              Строгим нулём это не назовёшь: на такой выборке честный потолок ошибки — около&nbsp;{falseUpperPct}%
              (грубое правило: три делить на число книг). Соавторов, которых нет, детектор на этой выборке не выдумывает.
            </p>
            <p className="note" style={{ margin: 0 }}>
              <strong style={{ color: "var(--cinnabar)" }}>Честная граница.</strong> Заметную и непохожую чужую
              руку метод ловит уверенно. С похожими по манере — сложнее. Мы сшили встык{" "}
              <strong style={{ color: "var(--text)" }}>{S.recallSimilar.total}</strong> пары близких авторов —
              шов нашёлся в каждой. Среди них единственная чисто донская пара
              {donPairLabel && <> — <strong style={{ color: "var(--text)" }}>{donPairLabel}</strong></>}. Но чужой
              в этих парах была немалая часть книги. Самая малая вставка, что мы поймали у похожих авторов, —
              около половины{" "}
              (<strong style={{ color: "var(--text)" }}>{S.similarDetectionFloorPct}%</strong>). Доли меньше мы
              на них не проверяли: это не доказанный порог провала, а лишь самая малая из пойманных долей.
              Поэтому маленький, размазанный по книге вклад <em>похожего</em> донского соавтора в «Тихий Дон»
              метод подтвердить не берётся. Это граница его мощности — и именно она решает исход того кейса.
            </p>
          </div>
        </div>

        <p className="muted reveal" style={{ fontSize: 13.5, maxWidth: "64ch" }}>
          На этом же детекторе держатся вердикты по «12 стульям» (булгаковского участка нет) и
          «Тихому Дону» (чужой руки в одном месте нет).
        </p>
      </div>
    </section>
  );
}
