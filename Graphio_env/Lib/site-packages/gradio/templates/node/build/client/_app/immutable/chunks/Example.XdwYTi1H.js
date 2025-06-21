import { SvelteComponent, init, safe_not_equal, element, claim_element, children, detach, attr, toggle_class, insert_hydration, transition_in, group_outros, transition_out, check_outros, create_component, claim_component, mount_component, destroy_component } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { J as JSON } from "./JSON.CiuEKgJ_.js";
function create_if_block(ctx) {
  let json;
  let current;
  json = new JSON({
    props: {
      value: (
        /*value*/
        ctx[0]
      ),
      open: true,
      theme_mode: (
        /*theme_mode*/
        ctx[1]
      ),
      show_indices,
      label_height,
      interactive: false,
      show_copy_button: false
    }
  });
  return {
    c() {
      create_component(json.$$.fragment);
    },
    l(nodes) {
      claim_component(json.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(json, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const json_changes = {};
      if (dirty & /*value*/
      1)
        json_changes.value = /*value*/
        ctx2[0];
      if (dirty & /*theme_mode*/
      2)
        json_changes.theme_mode = /*theme_mode*/
        ctx2[1];
      json.$set(json_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(json.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(json.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(json, detaching);
    }
  };
}
function create_fragment(ctx) {
  let div;
  let current;
  let if_block = (
    /*value*/
    ctx[0] && create_if_block(ctx)
  );
  return {
    c() {
      div = element("div");
      if (if_block)
        if_block.c();
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      if (if_block)
        if_block.l(div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "container svelte-v7ph9u");
      toggle_class(
        div,
        "table",
        /*type*/
        ctx[2] === "table"
      );
      toggle_class(
        div,
        "gallery",
        /*type*/
        ctx[2] === "gallery"
      );
      toggle_class(
        div,
        "selected",
        /*selected*/
        ctx[3]
      );
      toggle_class(
        div,
        "border",
        /*value*/
        ctx[0]
      );
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      if (if_block)
        if_block.m(div, null);
      current = true;
    },
    p(ctx2, [dirty]) {
      if (
        /*value*/
        ctx2[0]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*value*/
          1) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(div, null);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      if (!current || dirty & /*type*/
      4) {
        toggle_class(
          div,
          "table",
          /*type*/
          ctx2[2] === "table"
        );
      }
      if (!current || dirty & /*type*/
      4) {
        toggle_class(
          div,
          "gallery",
          /*type*/
          ctx2[2] === "gallery"
        );
      }
      if (!current || dirty & /*selected*/
      8) {
        toggle_class(
          div,
          "selected",
          /*selected*/
          ctx2[3]
        );
      }
      if (!current || dirty & /*value*/
      1) {
        toggle_class(
          div,
          "border",
          /*value*/
          ctx2[0]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      if (if_block)
        if_block.d();
    }
  };
}
let show_indices = false;
let label_height = 0;
function instance($$self, $$props, $$invalidate) {
  let { value } = $$props;
  let { theme_mode = "system" } = $$props;
  let { type } = $$props;
  let { selected = false } = $$props;
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("theme_mode" in $$props2)
      $$invalidate(1, theme_mode = $$props2.theme_mode);
    if ("type" in $$props2)
      $$invalidate(2, type = $$props2.type);
    if ("selected" in $$props2)
      $$invalidate(3, selected = $$props2.selected);
  };
  return [value, theme_mode, type, selected];
}
class Example extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      value: 0,
      theme_mode: 1,
      type: 2,
      selected: 3
    });
  }
}
export {
  Example as default
};
