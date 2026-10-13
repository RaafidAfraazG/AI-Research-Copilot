import ReactMarkdown from "react-markdown";
import { safeExternalUrl } from "../lib/safeUrl";

/**
 * Renders LLM-produced Markdown safely: raw HTML is never interpreted (react-markdown
 * default, no rehype-raw), images and embeds are dropped, and links are restricted to
 * http(s) and opened with rel="noopener noreferrer".
 */
export function Markdown({ children, inline = false }: { children: string; inline?: boolean }) {
  return (
    <ReactMarkdown
      disallowedElements={["img", "script", "iframe", "style"]}
      unwrapDisallowed
      components={{
        a({ href, children: kids }) {
          const safe = safeExternalUrl(href);
          if (!safe) return <span>{kids}</span>;
          return (
            <a href={safe} target="_blank" rel="noopener noreferrer">
              {kids}
            </a>
          );
        },
        ...(inline ? { p: ({ children: kids }) => <>{kids}</> } : {}),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
