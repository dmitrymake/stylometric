import { EditorialPanel, CodeBlock } from "@dmitrymake/rk-ui";

const NOTES = [
  {
    title: "Состав сравнения",
    accent: "var(--gold)",
    body: "Кандидаты перечислены в конфиге, исследуемое произведение выбирается явно. Его другие издания и тексты с тем же содержанием должны оставаться вне эталонов.",
  },
  {
    title: "Вес произведений",
    accent: "var(--icon-blue)",
    body: "При work_balanced выравнивается суммарный вес авторов, а внутри каждого автора — вес произведений. Вместе с этим меняются обучение словаря, частотные веса и нормировка частот.",
  },
  {
    title: "Повторное использование кэша",
    accent: "var(--success)",
    body: "Языковой разбор и базовые представления могут использоваться повторно при совпадении текста, модели и параметров обработки. Изменение этих условий требует нового расчёта; ускорение зависит от того, какая часть работы уже выполнена.",
  },
];

export default function Repro() {
  return (
    <section className="section" id="repro">
      <div className="wrap flow">
        <div className="section-head reveal">
          <p className="eyebrow">05 / Работа с программой</p>
          <h2>Как запустить своё сравнение</h2>
          <p className="prose lead muted">
            Для собственного отчёта нужны эталонные произведения, список кандидатов и
            явно выбранный исследуемый текст. Синтетический пример позволяет проверить
            этот путь без литературного корпуса. Повторение измерения качества
            требует отдельно восстановить его исходные данные и условия.
          </p>
        </div>

        <div className="module reveal" style={{ display: "grid", gap: 16 }}>
          <h3>Установка из корня проекта</h3>
          <CodeBlock language="bash" title="окружение и языковая модель">
            {`uv venv --python 3.11 --seed
uv pip install --constraint requirements.lock -e ".[dev]"
.venv/bin/python -m spacy download ru_core_news_lg`}
          </CodeBlock>
          <p className="prose muted">
            Проверенное окружение — CPython 3.11 на Linux. Параметр --seed добавляет pip,
            который нужен загрузчику модели spaCy. Тексты вместе с кодом не устанавливаются.
          </p>
        </div>

        <div className="module reveal" style={{ display: "grid", gap: 16 }}>
          <h3>Синтетический пример</h3>
          <CodeBlock language="bash" title="проверить путь до отчёта">
            {`.venv/bin/python scripts/make_demo_corpus.py --output research/local/demo
.venv/bin/stylo analyze --config research/local/demo/case.yaml --target-work unknown/target_periodic`}
          </CodeBlock>
          <p className="prose muted">
            Генератор создаёт оригинальные синтетические тексты вымышленных авторов и
            исследуемые цели. Каталог должен быть пустым: существующие файлы не
            перезаписываются. Этот запуск проверяет работу программы, а не точность на литературе.
          </p>
        </div>

        <div className="module reveal" style={{ display: "grid", gap: 16 }}>
          <h3>Свой корпус и конкретная цель</h3>
          <p className="prose muted">
            Эталонные UTF-8 тексты располагаются по пути input/&lt;author_id&gt;/&lt;work_id&gt;.txt:
            одно произведение в одном файле. Исследуемый текст — input/unknown/target.txt.
            В research/local/case.yaml укажите имена каталогов эталонных авторов:
          </p>
          <CodeBlock language="yaml" title="research/local/case.yaml">
            {`deployment:
  candidate_authors: [author_a, author_b]
evaluation:
  training_weighting: work_balanced
paths:
  data: research/local/case/data
  docs: research/local/case/results`}
          </CodeBlock>
          <CodeBlock language="bash" title="проанализировать выбранное произведение">
            {`.venv/bin/stylo analyze --config research/local/case.yaml --target-work unknown/target`}
          </CodeBlock>
          <p className="prose muted">
            Команда очищает и проверяет корпус, откладывает выбранное произведение,
            нарезает тексты, обучает модели и создаёт JSON/HTML-отчёт. LR и Delta
            показываются отдельно. Пути разрешаются относительно рабочего каталога запуска;
            оценки относятся к сравнению внутри заданной панели кандидатов.
          </p>
        </div>

        <div className="grid cols-3 module reveal">
          {NOTES.map((note) => (
            <EditorialPanel key={note.title} style={{ borderTop: `3px solid ${note.accent}` }}>
              <h3 style={{ margin: "0 0 8px", fontSize: "1.2rem", color: note.accent }}>{note.title}</h3>
              <p className="prose muted" style={{ margin: 0, fontSize: 17 }}>{note.body}</p>
            </EditorialPanel>
          ))}
        </div>

        <div className="module reveal prose">
          <h3>Что требуется для повторения измерения из статьи</h3>
          <p>
            Публичный агрегат содержит результаты и идентификаторы условий, но не исходные
            литературные тексты. Для повторения замера нужен тот же частный корпус,
            распределение произведений по проверкам, конфигурация и окружение.
            Загрузка другой подборки классики или запуск своего analyze создают другое сравнение.
          </p>
          <p className="note">
            Артефакт docs/repro_gates.json содержит результаты точных сверок для
            перечисленных в нём корпусов, настроек и окружений. Для другого запуска
            совпадение требуется проверять отдельно. Подробные команды и устройство программы — в README.
          </p>
        </div>
      </div>
    </section>
  );
}
