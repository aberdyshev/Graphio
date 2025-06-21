import { c as create_ssr_component, o as onDestroy, v as validate_component, e as escape, b as add_attribute } from './ssr-Cwm06D-i.js';
import { B as Block, W as BlockTitle, a0 as Calendar } from './2-BgvxP-qP.js';
export { default as BaseExample } from './Example-BzP15dgx.js';
import './index-CoAj_-n5.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BlohB9Ds.js';

const date_is_valid_format = (date, include_time) => {
  if (date === null || date === "")
    return true;
  const valid_regex = include_time ? /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/ : /^\d{4}-\d{2}-\d{2}$/;
  const is_valid_date = date.match(valid_regex) !== null;
  const is_valid_now = date.match(/^(?:\s*now\s*(?:-\s*\d+\s*[dmhs])?)?\s*$/) !== null;
  return is_valid_date || is_valid_now;
};
const parse_date_value = (entered_value, include_time) => {
  if (!entered_value || entered_value === "") {
    const now2 = /* @__PURE__ */ new Date();
    return {
      selected_date: now2,
      current_year: now2.getFullYear(),
      current_month: now2.getMonth(),
      selected_hour: now2.getHours(),
      selected_minute: now2.getMinutes(),
      selected_second: now2.getSeconds(),
      is_pm: now2.getHours() >= 12
    };
  }
  try {
    let date_to_parse = entered_value;
    if (!include_time && entered_value.match(/^\d{4}-\d{2}-\d{2}$/)) {
      date_to_parse += " 00:00:00";
    }
    const parsed = new Date(date_to_parse.replace(" ", "T"));
    if (!isNaN(parsed.getTime())) {
      return {
        selected_date: parsed,
        current_year: parsed.getFullYear(),
        current_month: parsed.getMonth(),
        selected_hour: parsed.getHours(),
        selected_minute: parsed.getMinutes(),
        selected_second: parsed.getSeconds(),
        is_pm: parsed.getHours() >= 12
      };
    }
  } catch (e) {
  }
  const now = /* @__PURE__ */ new Date();
  return {
    selected_date: now,
    current_year: now.getFullYear(),
    current_month: now.getMonth(),
    selected_hour: now.getHours(),
    selected_minute: now.getMinutes(),
    selected_second: now.getSeconds(),
    is_pm: now.getHours() >= 12
  };
};
const css = {
  code: ".label-content.svelte-ywg1ch{display:flex;justify-content:space-between;align-items:flex-start}button.svelte-ywg1ch{cursor:pointer;color:var(--body-text-color-subdued)}button.svelte-ywg1ch:hover{color:var(--body-text-color)}.svelte-ywg1ch::placeholder{color:var(--input-placeholder-color)}.timebox.svelte-ywg1ch{flex-grow:1;flex-shrink:1;display:flex;position:relative;background:var(--input-background-fill)}.timebox.svelte-ywg1ch svg{height:18px}.time.svelte-ywg1ch{padding:var(--input-padding);color:var(--body-text-color);font-weight:var(--input-text-weight);font-size:var(--input-text-size);line-height:var(--line-sm);outline:none;flex-grow:1;background:none;border:var(--input-border-width) solid var(--input-border-color);border-right:none;border-top-left-radius:var(--input-radius);border-bottom-left-radius:var(--input-radius);box-shadow:var(--input-shadow)}.time.svelte-ywg1ch:disabled{border-right:var(--input-border-width) solid var(--input-border-color);border-top-right-radius:var(--input-radius);border-bottom-right-radius:var(--input-radius)}.time.invalid.svelte-ywg1ch{color:var(--body-text-color-subdued)}.calendar.svelte-ywg1ch{display:inline-flex;justify-content:center;align-items:center;transition:var(--button-transition);box-shadow:var(--button-primary-shadow);text-align:center;background:var(--button-secondary-background-fill);color:var(--button-secondary-text-color);font-weight:var(--button-large-text-weight);font-size:var(--button-large-text-size);border-top-right-radius:var(--input-radius);border-bottom-right-radius:var(--input-radius);padding:var(--size-2);border:var(--input-border-width) solid var(--input-border-color)}.calendar.svelte-ywg1ch:hover{background:var(--button-secondary-background-fill-hover);box-shadow:var(--button-primary-shadow-hover)}.calendar.svelte-ywg1ch:active{box-shadow:var(--button-primary-shadow-active)}",
  map: '{"version":3,"file":"Index.svelte","sources":["Index.svelte"],"sourcesContent":["<script context=\\"module\\" lang=\\"ts\\">export { default as BaseExample } from \\"./Example.svelte\\";\\n<\/script>\\n\\n<script lang=\\"ts\\">import { Block, BlockTitle } from \\"@gradio/atoms\\";\\nimport { Calendar } from \\"@gradio/icons\\";\\nimport { onDestroy } from \\"svelte\\";\\nimport DateTimePicker from \\"./DateTimePicker.svelte\\";\\nimport { format_date, date_is_valid_format, parse_date_value } from \\"./utils\\";\\nexport let gradio;\\nexport let label = \\"Time\\";\\nexport let show_label = true;\\nexport let info = void 0;\\nexport let interactive;\\n$: disabled = !interactive;\\nexport let elem_id = \\"\\";\\nexport let elem_classes = [];\\nexport let visible = true;\\nexport let value = \\"\\";\\nlet old_value = value;\\nexport let scale = null;\\nexport let min_width = void 0;\\nexport let include_time = true;\\nlet show_picker = false;\\nlet picker_ref;\\nlet input_ref;\\nlet calendar_button_ref;\\nlet picker_position = { top: 0, left: 0 };\\n$: if (value !== old_value) {\\n    old_value = value;\\n    entered_value = value;\\n    update_picker_from_value();\\n    gradio.dispatch(\\"change\\");\\n}\\nlet entered_value = value;\\n$: valid = date_is_valid_format(entered_value, include_time);\\nconst submit_values = () => {\\n    if (entered_value === value)\\n        return;\\n    if (!date_is_valid_format(entered_value, include_time))\\n        return;\\n    old_value = value = entered_value;\\n    gradio.dispatch(\\"change\\");\\n};\\nlet current_year = ( /* @__PURE__ */new Date()).getFullYear();\\nlet current_month = ( /* @__PURE__ */new Date()).getMonth();\\nlet selected_date = /* @__PURE__ */ new Date();\\nlet selected_hour = ( /* @__PURE__ */new Date()).getHours();\\nlet selected_minute = ( /* @__PURE__ */new Date()).getMinutes();\\nlet selected_second = ( /* @__PURE__ */new Date()).getSeconds();\\nlet is_pm = selected_hour >= 12;\\nconst update_picker_from_value = () => {\\n    const parsed = parse_date_value(entered_value, include_time);\\n    selected_date = parsed.selected_date;\\n    current_year = parsed.current_year;\\n    current_month = parsed.current_month;\\n    selected_hour = parsed.selected_hour;\\n    selected_minute = parsed.selected_minute;\\n    selected_second = parsed.selected_second;\\n    is_pm = parsed.is_pm;\\n};\\nconst calculate_picker_position = () => {\\n    if (calendar_button_ref) {\\n        const rect = calendar_button_ref.getBoundingClientRect();\\n        picker_position = {\\n            top: rect.bottom + 4,\\n            left: rect.right - 280\\n        };\\n    }\\n};\\nconst toggle_picker = (event) => {\\n    if (!disabled) {\\n        event.stopPropagation();\\n        show_picker = !show_picker;\\n        if (show_picker) {\\n            update_picker_from_value();\\n            calculate_picker_position();\\n            setTimeout(() => {\\n                if (typeof window !== \\"undefined\\") {\\n                    window.addEventListener(\\"click\\", handle_click_outside);\\n                    window.addEventListener(\\"scroll\\", handle_scroll, true);\\n                }\\n            }, 10);\\n        }\\n        else if (typeof window !== \\"undefined\\") {\\n            window.removeEventListener(\\"click\\", handle_click_outside);\\n            window.removeEventListener(\\"scroll\\", handle_scroll, true);\\n        }\\n    }\\n};\\nconst close_picker = () => {\\n    show_picker = false;\\n    if (typeof window !== \\"undefined\\") {\\n        window.removeEventListener(\\"click\\", handle_click_outside);\\n        window.removeEventListener(\\"scroll\\", handle_scroll, true);\\n    }\\n};\\nconst handle_click_outside = (event) => {\\n    if (show_picker && picker_ref && !picker_ref.contains(event.target) && calendar_button_ref && !calendar_button_ref.contains(event.target)) {\\n        close_picker();\\n    }\\n};\\nconst handle_scroll = () => {\\n    if (show_picker) {\\n        calculate_picker_position();\\n    }\\n};\\nconst handle_picker_update = (event) => {\\n    entered_value = event.detail.formatted;\\n    submit_values();\\n};\\nconst handle_picker_clear = () => {\\n    entered_value = \\"\\";\\n    value = \\"\\";\\n    close_picker();\\n    gradio.dispatch(\\"change\\");\\n};\\nonDestroy(() => {\\n    if (typeof window !== \\"undefined\\") {\\n        window.removeEventListener(\\"click\\", handle_click_outside);\\n        window.removeEventListener(\\"scroll\\", handle_scroll, true);\\n    }\\n});\\nupdate_picker_from_value();\\n<\/script>\\n\\n<Block\\n\\t{visible}\\n\\t{elem_id}\\n\\t{elem_classes}\\n\\t{scale}\\n\\t{min_width}\\n\\tallow_overflow={false}\\n\\tpadding={true}\\n>\\n\\t<div class=\\"label-content\\">\\n\\t\\t<BlockTitle {show_label} {info}>{label}</BlockTitle>\\n\\t</div>\\n\\t<div class=\\"timebox\\">\\n\\t\\t<input\\n\\t\\t\\tbind:this={input_ref}\\n\\t\\t\\tclass=\\"time\\"\\n\\t\\t\\tbind:value={entered_value}\\n\\t\\t\\tclass:invalid={!valid}\\n\\t\\t\\ton:keydown={(evt) => {\\n\\t\\t\\t\\tif (evt.key === \\"Enter\\") {\\n\\t\\t\\t\\t\\tsubmit_values();\\n\\t\\t\\t\\t\\tgradio.dispatch(\\"submit\\");\\n\\t\\t\\t\\t}\\n\\t\\t\\t}}\\n\\t\\t\\ton:blur={submit_values}\\n\\t\\t\\t{disabled}\\n\\t\\t\\tplaceholder={include_time ? \\"YYYY-MM-DD HH:MM:SS\\" : \\"YYYY-MM-DD\\"}\\n\\t\\t/>\\n\\n\\t\\t{#if interactive}\\n\\t\\t\\t<button\\n\\t\\t\\t\\tbind:this={calendar_button_ref}\\n\\t\\t\\t\\tclass=\\"calendar\\"\\n\\t\\t\\t\\t{disabled}\\n\\t\\t\\t\\ton:click={toggle_picker}\\n\\t\\t\\t>\\n\\t\\t\\t\\t<Calendar />\\n\\t\\t\\t</button>\\n\\t\\t{/if}\\n\\t</div>\\n\\n\\t{#if show_picker}\\n\\t\\t<div bind:this={picker_ref}>\\n\\t\\t\\t<DateTimePicker\\n\\t\\t\\t\\tbind:selected_date\\n\\t\\t\\t\\tbind:current_year\\n\\t\\t\\t\\tbind:current_month\\n\\t\\t\\t\\tbind:selected_hour\\n\\t\\t\\t\\tbind:selected_minute\\n\\t\\t\\t\\tbind:selected_second\\n\\t\\t\\t\\tbind:is_pm\\n\\t\\t\\t\\t{include_time}\\n\\t\\t\\t\\tposition={picker_position}\\n\\t\\t\\t\\ton:update={handle_picker_update}\\n\\t\\t\\t\\ton:clear={handle_picker_clear}\\n\\t\\t\\t\\ton:close={close_picker}\\n\\t\\t\\t/>\\n\\t\\t</div>\\n\\t{/if}\\n</Block>\\n\\n<style>\\n\\t.label-content {\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: space-between;\\n\\t\\talign-items: flex-start;\\n\\t}\\n\\n\\tbutton {\\n\\t\\tcursor: pointer;\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t}\\n\\n\\tbutton:hover {\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t::placeholder {\\n\\t\\tcolor: var(--input-placeholder-color);\\n\\t}\\n\\n\\t.timebox {\\n\\t\\tflex-grow: 1;\\n\\t\\tflex-shrink: 1;\\n\\t\\tdisplay: flex;\\n\\t\\tposition: relative;\\n\\t\\tbackground: var(--input-background-fill);\\n\\t}\\n\\n\\t.timebox :global(svg) {\\n\\t\\theight: 18px;\\n\\t}\\n\\n\\t.time {\\n\\t\\tpadding: var(--input-padding);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-weight: var(--input-text-weight);\\n\\t\\tfont-size: var(--input-text-size);\\n\\t\\tline-height: var(--line-sm);\\n\\t\\toutline: none;\\n\\t\\tflex-grow: 1;\\n\\t\\tbackground: none;\\n\\t\\tborder: var(--input-border-width) solid var(--input-border-color);\\n\\t\\tborder-right: none;\\n\\t\\tborder-top-left-radius: var(--input-radius);\\n\\t\\tborder-bottom-left-radius: var(--input-radius);\\n\\t\\tbox-shadow: var(--input-shadow);\\n\\t}\\n\\n\\t.time:disabled {\\n\\t\\tborder-right: var(--input-border-width) solid var(--input-border-color);\\n\\t\\tborder-top-right-radius: var(--input-radius);\\n\\t\\tborder-bottom-right-radius: var(--input-radius);\\n\\t}\\n\\n\\t.time.invalid {\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t}\\n\\n\\t.calendar {\\n\\t\\tdisplay: inline-flex;\\n\\t\\tjustify-content: center;\\n\\t\\talign-items: center;\\n\\t\\ttransition: var(--button-transition);\\n\\t\\tbox-shadow: var(--button-primary-shadow);\\n\\t\\ttext-align: center;\\n\\t\\tbackground: var(--button-secondary-background-fill);\\n\\t\\tcolor: var(--button-secondary-text-color);\\n\\t\\tfont-weight: var(--button-large-text-weight);\\n\\t\\tfont-size: var(--button-large-text-size);\\n\\t\\tborder-top-right-radius: var(--input-radius);\\n\\t\\tborder-bottom-right-radius: var(--input-radius);\\n\\t\\tpadding: var(--size-2);\\n\\t\\tborder: var(--input-border-width) solid var(--input-border-color);\\n\\t}\\n\\n\\t.calendar:hover {\\n\\t\\tbackground: var(--button-secondary-background-fill-hover);\\n\\t\\tbox-shadow: var(--button-primary-shadow-hover);\\n\\t}\\n\\n\\t.calendar:active {\\n\\t\\tbox-shadow: var(--button-primary-shadow-active);\\n\\t}</style>\\n"],"names":[],"mappings":"AA2LC,4BAAe,CACd,OAAO,CAAE,IAAI,CACb,eAAe,CAAE,aAAa,CAC9B,WAAW,CAAE,UACd,CAEA,oBAAO,CACN,MAAM,CAAE,OAAO,CACf,KAAK,CAAE,IAAI,yBAAyB,CACrC,CAEA,oBAAM,MAAO,CACZ,KAAK,CAAE,IAAI,iBAAiB,CAC7B,eAEA,aAAc,CACb,KAAK,CAAE,IAAI,yBAAyB,CACrC,CAEA,sBAAS,CACR,SAAS,CAAE,CAAC,CACZ,WAAW,CAAE,CAAC,CACd,OAAO,CAAE,IAAI,CACb,QAAQ,CAAE,QAAQ,CAClB,UAAU,CAAE,IAAI,uBAAuB,CACxC,CAEA,sBAAQ,CAAS,GAAK,CACrB,MAAM,CAAE,IACT,CAEA,mBAAM,CACL,OAAO,CAAE,IAAI,eAAe,CAAC,CAC7B,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,WAAW,CAAE,IAAI,mBAAmB,CAAC,CACrC,SAAS,CAAE,IAAI,iBAAiB,CAAC,CACjC,WAAW,CAAE,IAAI,SAAS,CAAC,CAC3B,OAAO,CAAE,IAAI,CACb,SAAS,CAAE,CAAC,CACZ,UAAU,CAAE,IAAI,CAChB,MAAM,CAAE,IAAI,oBAAoB,CAAC,CAAC,KAAK,CAAC,IAAI,oBAAoB,CAAC,CACjE,YAAY,CAAE,IAAI,CAClB,sBAAsB,CAAE,IAAI,cAAc,CAAC,CAC3C,yBAAyB,CAAE,IAAI,cAAc,CAAC,CAC9C,UAAU,CAAE,IAAI,cAAc,CAC/B,CAEA,mBAAK,SAAU,CACd,YAAY,CAAE,IAAI,oBAAoB,CAAC,CAAC,KAAK,CAAC,IAAI,oBAAoB,CAAC,CACvE,uBAAuB,CAAE,IAAI,cAAc,CAAC,CAC5C,0BAA0B,CAAE,IAAI,cAAc,CAC/C,CAEA,KAAK,sBAAS,CACb,KAAK,CAAE,IAAI,yBAAyB,CACrC,CAEA,uBAAU,CACT,OAAO,CAAE,WAAW,CACpB,eAAe,CAAE,MAAM,CACvB,WAAW,CAAE,MAAM,CACnB,UAAU,CAAE,IAAI,mBAAmB,CAAC,CACpC,UAAU,CAAE,IAAI,uBAAuB,CAAC,CACxC,UAAU,CAAE,MAAM,CAClB,UAAU,CAAE,IAAI,kCAAkC,CAAC,CACnD,KAAK,CAAE,IAAI,6BAA6B,CAAC,CACzC,WAAW,CAAE,IAAI,0BAA0B,CAAC,CAC5C,SAAS,CAAE,IAAI,wBAAwB,CAAC,CACxC,uBAAuB,CAAE,IAAI,cAAc,CAAC,CAC5C,0BAA0B,CAAE,IAAI,cAAc,CAAC,CAC/C,OAAO,CAAE,IAAI,QAAQ,CAAC,CACtB,MAAM,CAAE,IAAI,oBAAoB,CAAC,CAAC,KAAK,CAAC,IAAI,oBAAoB,CACjE,CAEA,uBAAS,MAAO,CACf,UAAU,CAAE,IAAI,wCAAwC,CAAC,CACzD,UAAU,CAAE,IAAI,6BAA6B,CAC9C,CAEA,uBAAS,OAAQ,CAChB,UAAU,CAAE,IAAI,8BAA8B,CAC/C"}'
};
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let disabled;
  let valid;
  let { gradio } = $$props;
  let { label = "Time" } = $$props;
  let { show_label = true } = $$props;
  let { info = void 0 } = $$props;
  let { interactive } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = "" } = $$props;
  let old_value = value;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { include_time = true } = $$props;
  let input_ref;
  let calendar_button_ref;
  let entered_value = value;
  const update_picker_from_value = () => {
    const parsed = parse_date_value(entered_value, include_time);
    parsed.selected_date;
    parsed.current_year;
    parsed.current_month;
    parsed.selected_hour;
    parsed.selected_minute;
    parsed.selected_second;
    parsed.is_pm;
  };
  const handle_click_outside = (event) => {
  };
  const handle_scroll = () => {
  };
  onDestroy(() => {
    if (typeof window !== "undefined") {
      window.removeEventListener("click", handle_click_outside);
      window.removeEventListener("scroll", handle_scroll, true);
    }
  });
  update_picker_from_value();
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.info === void 0 && $$bindings.info && info !== void 0)
    $$bindings.info(info);
  if ($$props.interactive === void 0 && $$bindings.interactive && interactive !== void 0)
    $$bindings.interactive(interactive);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.include_time === void 0 && $$bindings.include_time && include_time !== void 0)
    $$bindings.include_time(include_time);
  $$result.css.add(css);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    disabled = !interactive;
    {
      if (value !== old_value) {
        old_value = value;
        entered_value = value;
        update_picker_from_value();
        gradio.dispatch("change");
      }
    }
    valid = date_is_valid_format(entered_value, include_time);
    $$rendered = `${validate_component(Block, "Block").$$render(
      $$result,
      {
        visible,
        elem_id,
        elem_classes,
        scale,
        min_width,
        allow_overflow: false,
        padding: true
      },
      {},
      {
        default: () => {
          return `<div class="label-content svelte-ywg1ch">${validate_component(BlockTitle, "BlockTitle").$$render($$result, { show_label, info }, {}, {
            default: () => {
              return `${escape(label)}`;
            }
          })}</div> <div class="timebox svelte-ywg1ch"><input class="${["time svelte-ywg1ch", !valid ? "invalid" : ""].join(" ").trim()}" ${disabled ? "disabled" : ""}${add_attribute("placeholder", include_time ? "YYYY-MM-DD HH:MM:SS" : "YYYY-MM-DD", 0)}${add_attribute("this", input_ref, 0)}${add_attribute("value", entered_value, 0)}> ${interactive ? `<button class="calendar svelte-ywg1ch" ${disabled ? "disabled" : ""}${add_attribute("this", calendar_button_ref, 0)}>${validate_component(Calendar, "Calendar").$$render($$result, {}, {}, {})}</button>` : ``}</div> ${``}`;
        }
      }
    )}`;
  } while (!$$settled);
  return $$rendered;
});

export { Index as default };
//# sourceMappingURL=Index54-Bp37rKfv.js.map
