import ReactMarkdown from "react-markdown";
import source from "./data-sources.md?raw";

const dataSources = source
  .replace(/^# Data & Data Sources\r?\n+/, "");

function DataSources() {
  return (
    <div className="longform-document">
      <ReactMarkdown
        components={{
          a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
        }}
      >
        {dataSources}
      </ReactMarkdown>
    </div>
  );
}

export default DataSources;
