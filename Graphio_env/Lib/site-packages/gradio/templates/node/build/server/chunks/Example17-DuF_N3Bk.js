import { c as create_ssr_component, v as validate_component } from './ssr-Cwm06D-i.js';
import { J as JSON$1 } from './JSON-DD7Naicy.js';
import './2-BgvxP-qP.js';
import './index-CoAj_-n5.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BlohB9Ds.js';

const css = {
  code: ".container.svelte-v7ph9u img{width:100%;height:100%}.container.selected.svelte-v7ph9u{border-color:var(--border-color-accent)}.border.table.svelte-v7ph9u{border:1px solid var(--border-color-primary)}.container.table.svelte-v7ph9u{margin:0 auto;border-radius:var(--radius-lg);overflow:hidden;width:100%;height:100%;max-width:var(--size-40);max-height:var(--size-20);object-fit:cover}.container.gallery.svelte-v7ph9u{width:100%;max-width:100%;object-fit:cover;max-width:var(--size-40);max-height:var(--size-20);overflow:hidden}",
  map: '{"version":3,"file":"Example.svelte","sources":["Example.svelte"],"sourcesContent":["<script lang=\\"ts\\">import JSON from \\"./shared/JSON.svelte\\";\\nexport let value;\\nexport let theme_mode = \\"system\\";\\nlet show_indices = false;\\nlet label_height = 0;\\nexport let type;\\nexport let selected = false;\\n<\/script>\\n\\n<div\\n\\tclass=\\"container\\"\\n\\tclass:table={type === \\"table\\"}\\n\\tclass:gallery={type === \\"gallery\\"}\\n\\tclass:selected\\n\\tclass:border={value}\\n>\\n\\t{#if value}\\n\\t\\t<JSON\\n\\t\\t\\t{value}\\n\\t\\t\\topen={true}\\n\\t\\t\\t{theme_mode}\\n\\t\\t\\t{show_indices}\\n\\t\\t\\t{label_height}\\n\\t\\t\\tinteractive={false}\\n\\t\\t\\tshow_copy_button={false}\\n\\t\\t/>\\n\\t{/if}\\n</div>\\n\\n<style>\\n\\t.container :global(img) {\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t}\\n\\n\\t.container.selected {\\n\\t\\tborder-color: var(--border-color-accent);\\n\\t}\\n\\t.border.table {\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t}\\n\\n\\t.container.table {\\n\\t\\tmargin: 0 auto;\\n\\t\\tborder-radius: var(--radius-lg);\\n\\t\\toverflow: hidden;\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t\\tmax-width: var(--size-40);\\n\\t\\tmax-height: var(--size-20);\\n\\t\\tobject-fit: cover;\\n\\t}\\n\\n\\t.container.gallery {\\n\\t\\twidth: 100%;\\n\\t\\tmax-width: 100%;\\n\\t\\tobject-fit: cover;\\n\\t\\tmax-width: var(--size-40);\\n\\t\\tmax-height: var(--size-20);\\n\\t\\toverflow: hidden;\\n\\t}</style>\\n"],"names":[],"mappings":"AA8BC,wBAAU,CAAS,GAAK,CACvB,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IACT,CAEA,UAAU,uBAAU,CACnB,YAAY,CAAE,IAAI,qBAAqB,CACxC,CACA,OAAO,oBAAO,CACb,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,sBAAsB,CAC7C,CAEA,UAAU,oBAAO,CAChB,MAAM,CAAE,CAAC,CAAC,IAAI,CACd,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,QAAQ,CAAE,MAAM,CAChB,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,UAAU,CAAE,IAAI,SAAS,CAAC,CAC1B,UAAU,CAAE,KACb,CAEA,UAAU,sBAAS,CAClB,KAAK,CAAE,IAAI,CACX,SAAS,CAAE,IAAI,CACf,UAAU,CAAE,KAAK,CACjB,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,UAAU,CAAE,IAAI,SAAS,CAAC,CAC1B,QAAQ,CAAE,MACX"}'
};
let show_indices = false;
let label_height = 0;
const Example = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  let { theme_mode = "system" } = $$props;
  let { type } = $$props;
  let { selected = false } = $$props;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props.type === void 0 && $$bindings.type && type !== void 0)
    $$bindings.type(type);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  $$result.css.add(css);
  return `<div class="${[
    "container svelte-v7ph9u",
    (type === "table" ? "table" : "") + " " + (type === "gallery" ? "gallery" : "") + " " + (selected ? "selected" : "") + " " + (value ? "border" : "")
  ].join(" ").trim()}">${value ? `${validate_component(JSON$1, "JSON").$$render(
    $$result,
    {
      value,
      open: true,
      theme_mode,
      show_indices,
      label_height,
      interactive: false,
      show_copy_button: false
    },
    {},
    {}
  )}` : ``} </div>`;
});

export { Example as default };
//# sourceMappingURL=Example17-DuF_N3Bk.js.map
