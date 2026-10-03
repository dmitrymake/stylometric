import { EditorialFigure, EditorialNote } from "@dmitrymake/rk-ui";
import { MEASUREMENT, PUBLICATION } from "../data.js";
import { fmtInt, fmtPct } from "../format.js";

const NAMES = {
  stylo_A0_current: "Stylo · обучение на фрагментах",
  stylo_A4_current: "Stylo · пакет с балансировкой произведений",
  stylo_A4_current_mass_control: "Stylo · тот же пакет, контроль регуляризации",
  stylo_A0_topic_strict: "Stylo · сокращённые признаки",
  stylo_A4_topic_strict: "Stylo · балансировка и сокращённые признаки",
  delta_300: "Delta · 300 частых слов",
  cosine_delta_300: "Cosine Delta · 300 частых слов",
  char_tfidf_lr: "Символьные последовательности и классификатор",
};
const pct = (value) => fmtPct(value, 2).replace(".", ",");

export default function Results() {
  const p = PUBLICATION;
  const dev = p.results.development_lobo.metrics;
  const test = p.results.locked_test.metrics;
  const base = dev.stylo_A0_current;
  const shortened = p.sensitivity.results.metrics;
  return <section className="section" id="results"><div className="wrap flow">
    <div className="section-head">
      <p className="eyebrow">03 / Проверка метода</p>
      <h2>Сравнение начинается с известных произведений</h2>
      <p className="prose lead">В панели {fmtInt(p.panel.works)} произведения Андреева, Чехова, Горького и Куприна.
        Цифровые копии связаны с источниками и заявленными библиотекой изданиями.
        {" "}{base.total} работы образуют подборку для разработки; ещё {test.stylo_A0_current.total} отложены для проверки.</p>
    </div>
    <p className="prose">На подборке для разработки каждое произведение по очереди исключается из обучения.
      Затем методы обучаются на всей этой подборке и один раз распознают отложенные работы.
      Их тексты участвуют только в получении ответа: словари, нормировка и классификатор строятся по обучающим данным.
      Настройки методов фиксированы до расчёта; всего выполнено {fmtInt(p.fits)} обучений.</p>
    <EditorialFigure label="Таблица 1" caption={<>Верно распознанные произведения и средняя полнота по четырём авторам.
      Последняя даёт каждому автору одинаковый вес. Отложенная подборка проверяет новые произведения в пределах этой панели.</>}
      source={<a href={`${import.meta.env.BASE_URL}${p.publicArtifact}`} download>Результаты и настройки · JSON</a>}>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Сравнение методов: таблицу можно прокручивать"><table>
        <thead><tr><th>Метод</th><th>Покнижная проверка</th><th>Средняя полнота</th><th>Отложенные работы</th></tr></thead>
        <tbody>{p.arms.map((arm) => <tr key={arm}>
          <td>{NAMES[arm]}</td><td className="mono">{dev[arm].correct}/{dev[arm].total}</td>
          <td className="mono">{pct(dev[arm].macro_author_recall)}</td>
          <td className="mono">{test[arm].correct}/{test[arm].total}</td>
        </tr>)}</tbody>
      </table></div>
    </EditorialFigure>
    <p className="table-scroll-hint">Таблицу можно прокрутить по горизонтали.</p>
    <h3>Более сложный пакет не даёт преимущества на этой панели</h3>
    <p className="prose">Пять вариантов Stylo правильно распознают {base.correct} из {base.total} работ в покнижной проверке.
      Балансировка, сокращение признаков и отдельный контроль регуляризации не меняют число верных ответов.
      Cosine Delta распознаёт {dev.cosine_delta_300.correct} из {dev.cosine_delta_300.total}.
      Различия сосредоточены у Андреева: Stylo распознаёт {base.per_author.andreev.correct} из {base.per_author.andreev.total}
      {" "}его работ, Cosine Delta — {dev.cosine_delta_300.per_author.andreev.correct}.</p>
    <p className="prose">Контроль регуляризации нужен потому, что A4 меняет не только относительный вклад книг,
      но и общую массу весов. При одинаковом параметре C это меняет силу штрафа.
      В контрольном варианте масштаб штрафа выровнен; остальные части A4 сохранены.
      Отсутствие изменения числа верных ответов здесь не устанавливает безразличия к этому фактору на других данных.</p>
    <EditorialNote title="Все ответы верны — в пределах небольшой подборки">
      <p>Семь методов распознают {test.stylo_A0_current.correct} из {test.stylo_A0_current.total} отложенных работ.
        Эта подборка не различает их качество. Символьная модель распознаёт {test.char_tfidf_lr.correct}; оба её промаха относятся к Андрееву.
        Ни результат без ошибок, ни равенство итоговых чисел не устанавливают универсальную точность или эквивалентность методов.</p>
    </EditorialNote>
    <h3>Начало произведения даёт другой ответ</h3>
    <p className="prose">В дополнительном опыте от каждой работы для разработки оставлены два начальных фрагмента —
      от {p.sensitivity.prefixTokens.min} до {p.sensitivity.prefixTokens.max} токенов.
      Stylo распознаёт {shortened.stylo_A0_current.correct} из {shortened.stylo_A0_current.total} вместо {base.correct};
      Cosine Delta — {shortened.cosine_delta_300.correct} вместо {dev.cosine_delta_300.correct}.
      Для этого опыта отложенные тексты не читаются и не оцениваются.</p>
    <p className="prose">Такой опыт показывает чувствительность к ограниченному начальному материалу.
      Одновременно меняются объём обучения, положение отрывка и представление текста;
      отдельный причинный эффект длины он не измеряет. Это дополнительный анализ после основного расчёта,
      без подбора настроек по отложенным ответам.</p>
    <p><a href={`${import.meta.env.BASE_URL}${p.sensitivity.publicArtifact}`} download>Опыт с начальными фрагментами · JSON</a></p>
    <EditorialNote title="Граница результата">
      <p>Панель содержит четырёх современников и повествовательную прозу 1890–1904 годов.
        Издания связаны с авторами; короткие и длинные работы распределены неравномерно.
        Проверка не отделяет авторскую манеру от всех тематических, жанровых и редакторских влияний.</p>
      <p>Она не проверяет автора вне списка, поиск соавтора или вставки. Литературные случаи используют свои
        эталоны и исторические протоколы; эти числа не устанавливают авторство обсуждаемых романов.</p>
    </EditorialNote>
    <details><summary>Корпус и воспроизводимые данные</summary>
      <p>Каталог фиксирует произведения, цифровые источники, заявленные издания, роли в проверке и контрольные суммы.
        Выполненная повторная загрузка воспроизвела обработанные тексты; служебная HTML-разметка библиотеки может меняться.
        Исходные литературные тексты вместе с кодом не распространяются.</p>
      <p><a href={`${import.meta.env.BASE_URL}${p.catalogArtifact}`} download>Каталог источников и условия отбора · JSON</a></p>
    </details>
    <details><summary>Диагностический расчёт на исторической подборке</summary>
      <p>Архивный расчёт содержит {MEASUREMENT.works} отложенных входных файлов с авторскими метками.
        Их библиографическое происхождение не удостоверено. Один файл под названием «Война и мир» содержит
        аномально повторяющийся текст; он не представляет полный роман.
        Архивные числа описывают классификацию тех файлов, но не качество на аттестованном литературном корпусе.</p>
      <p><a href={`${import.meta.env.BASE_URL}${MEASUREMENT.publicArtifact}`} download>Архивный агрегат · JSON</a></p>
    </details>
  </div></section>;
}
