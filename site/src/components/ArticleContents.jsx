import { EditorialContents } from "@dmitrymake/rk-ui";

export default function ArticleContents({ chapter, items }) {
  return <div className="article-contents"><EditorialContents label="В этой главе" items={items.map(([id, label]) => ({ href: `#${chapter}/${id}`, label }))} /></div>;
}
