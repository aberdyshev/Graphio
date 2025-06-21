import { c as create_ssr_component, v as validate_component } from './ssr-Cwm06D-i.js';
import { H as MarkdownCode } from './2-BgvxP-qP.js';
import './index-CoAj_-n5.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BlohB9Ds.js';

const css = {
  code: ".gallery.svelte-1ayixqk{padding:var(--size-1) var(--size-2)}",
  map: '{"version":3,"file":"Example.svelte","sources":["Example.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { MarkdownCode } from \\"@gradio/markdown-code\\";\\nexport let value;\\nexport let type;\\nexport let selected = false;\\nexport let sanitize_html;\\nexport let line_breaks;\\nexport let latex_delimiters;\\nfunction truncate_text(text, max_length = 60) {\\n    if (!text)\\n        return \\"\\";\\n    const str = String(text);\\n    if (str.length <= max_length)\\n        return str;\\n    return str.slice(0, max_length) + \\"...\\";\\n}\\n<\/script>\\n\\n<div\\n\\tclass:table={type === \\"table\\"}\\n\\tclass:gallery={type === \\"gallery\\"}\\n\\tclass:selected\\n\\tclass=\\"prose\\"\\n>\\n\\t<MarkdownCode\\n\\t\\tmessage={truncate_text(value)}\\n\\t\\t{latex_delimiters}\\n\\t\\t{sanitize_html}\\n\\t\\t{line_breaks}\\n\\t\\tchatbot={false}\\n\\t/>\\n</div>\\n\\n<style>\\n\\t.gallery {\\n\\t\\tpadding: var(--size-1) var(--size-2);\\n\\t}</style>\\n"],"names":[],"mappings":"AAiCC,uBAAS,CACR,OAAO,CAAE,IAAI,QAAQ,CAAC,CAAC,IAAI,QAAQ,CACpC"}'
};
function truncate_text(text, max_length = 60) {
  if (!text)
    return "";
  const str = String(text);
  if (str.length <= max_length)
    return str;
  return str.slice(0, max_length) + "...";
}
const Example = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  let { type } = $$props;
  let { selected = false } = $$props;
  let { sanitize_html } = $$props;
  let { line_breaks } = $$props;
  let { latex_delimiters } = $$props;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.type === void 0 && $$bindings.type && type !== void 0)
    $$bindings.type(type);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  if ($$props.sanitize_html === void 0 && $$bindings.sanitize_html && sanitize_html !== void 0)
    $$bindings.sanitize_html(sanitize_html);
  if ($$props.line_breaks === void 0 && $$bindings.line_breaks && line_breaks !== void 0)
    $$bindings.line_breaks(line_breaks);
  if ($$props.latex_delimiters === void 0 && $$bindings.latex_delimiters && latex_delimiters !== void 0)
    $$bindings.latex_delimiters(latex_delimiters);
  $$result.css.add(css);
  return `<div class="${[
    "prose svelte-1ayixqk",
    (type === "table" ? "table" : "") + " " + (type === "gallery" ? "gallery" : "") + " " + (selected ? "selected" : "")
  ].join(" ").trim()}">${validate_component(MarkdownCode, "MarkdownCode").$$render(
    $$result,
    {
      message: truncate_text(value),
      latex_delimiters,
      sanitize_html,
      line_breaks,
      chatbot: false
    },
    {},
    {}
  )} </div>`;
});

export { Example as default };
//# sourceMappingURL=Example18-C5rvYDkC.js.map
