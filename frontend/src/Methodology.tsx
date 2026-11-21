import ReactMarkdown from "react-markdown";
import source from "./methodology.md?raw";

const methodology = source
  .replace(/^# Our Methodology\r?\n+/, "")
  .replace(/^\[\r?\n([\s\S]*?)^\]\s*$/gm, (_match, formula: string) => `\n\`\`\`equation\n${formula.trim()}\n\`\`\`\n`)
  .replace(
    /(A simplified model structure can be expressed as:\r?\n\r?\n)([\s\S]*?)\r?\n\](?=\r?\n\r?\nWhere:)/,
    (_match, introduction: string, formula: string) => `${introduction}\`\`\`equation\n${formula.trim()}\n\`\`\``,
  )
  .replace("[Insert Date]", "21 November 2026");

function Methodology() {
  return (
    <div className="longform-document">
      <ReactMarkdown>{methodology}</ReactMarkdown>
    </div>
  );
}

export default Methodology;
