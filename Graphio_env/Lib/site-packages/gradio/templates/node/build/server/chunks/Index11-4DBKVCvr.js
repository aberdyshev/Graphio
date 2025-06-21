import { c as create_ssr_component, v as validate_component } from './ssr-Cwm06D-i.js';
import { J as JSON$1$1 } from './JSON-DD7Naicy.js';
import { B as Block, j as BlockLabel, aq as JSON$1, S as Static } from './2-BgvxP-qP.js';
import './index-CoAj_-n5.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BlohB9Ds.js';

const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value } = $$props;
  let old_value;
  let { loading_status } = $$props;
  let { label } = $$props;
  let { show_label } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { gradio } = $$props;
  let { open = false } = $$props;
  let { theme_mode } = $$props;
  let { show_indices } = $$props;
  let { height } = $$props;
  let { min_height } = $$props;
  let { max_height } = $$props;
  let label_height = 0;
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.container === void 0 && $$bindings.container && container !== void 0)
    $$bindings.container(container);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.open === void 0 && $$bindings.open && open !== void 0)
    $$bindings.open(open);
  if ($$props.theme_mode === void 0 && $$bindings.theme_mode && theme_mode !== void 0)
    $$bindings.theme_mode(theme_mode);
  if ($$props.show_indices === void 0 && $$bindings.show_indices && show_indices !== void 0)
    $$bindings.show_indices(show_indices);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props.min_height === void 0 && $$bindings.min_height && min_height !== void 0)
    $$bindings.min_height(min_height);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  {
    {
      if (value !== old_value) {
        old_value = value;
        gradio.dispatch("change");
      }
    }
  }
  return `${validate_component(Block, "Block").$$render(
    $$result,
    {
      visible,
      test_id: "json",
      elem_id,
      elem_classes,
      container,
      scale,
      min_width,
      padding: false,
      allow_overflow: true,
      overflow_behavior: "auto",
      height,
      min_height,
      max_height
    },
    {},
    {
      default: () => {
        return `<div>${label ? `${validate_component(BlockLabel, "BlockLabel").$$render(
          $$result,
          {
            Icon: JSON$1,
            show_label,
            label,
            float: false,
            disable: container === false
          },
          {},
          {}
        )}` : ``}</div> ${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status), {}, {})} ${validate_component(JSON$1$1, "JSON").$$render(
          $$result,
          {
            value,
            open,
            theme_mode,
            show_indices,
            label_height
          },
          {},
          {}
        )}`;
      }
    }
  )}`;
});

export { JSON$1$1 as BaseJSON, Index as default };
//# sourceMappingURL=Index11-4DBKVCvr.js.map
