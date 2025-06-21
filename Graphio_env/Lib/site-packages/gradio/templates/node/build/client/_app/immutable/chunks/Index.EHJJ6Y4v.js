import { SvelteComponent, init, safe_not_equal, create_component, claim_component, mount_component, transition_in, transition_out, destroy_component, assign, element, space, claim_element, children, detach, claim_space, add_render_callback, insert_hydration, add_iframe_resize_listener, group_outros, check_outros, get_spread_update, get_spread_object } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { J as JSON, a as JSON$1 } from "./JSON.CiuEKgJ_.js";
import { B as Block, S as Static } from "./2.CWSJHAMp.js";
import { B as BlockLabel } from "./BlockLabel.Cft09z0E.js";
function create_if_block(ctx) {
  let blocklabel;
  let current;
  blocklabel = new BlockLabel({
    props: {
      Icon: JSON$1,
      show_label: (
        /*show_label*/
        ctx[6]
      ),
      label: (
        /*label*/
        ctx[5]
      ),
      float: false,
      disable: (
        /*container*/
        ctx[7] === false
      )
    }
  });
  return {
    c() {
      create_component(blocklabel.$$.fragment);
    },
    l(nodes) {
      claim_component(blocklabel.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(blocklabel, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const blocklabel_changes = {};
      if (dirty & /*show_label*/
      64)
        blocklabel_changes.show_label = /*show_label*/
        ctx2[6];
      if (dirty & /*label*/
      32)
        blocklabel_changes.label = /*label*/
        ctx2[5];
      if (dirty & /*container*/
      128)
        blocklabel_changes.disable = /*container*/
        ctx2[7] === false;
      blocklabel.$set(blocklabel_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(blocklabel.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(blocklabel.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(blocklabel, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let div;
  let div_resize_listener;
  let t0;
  let statustracker;
  let t1;
  let json;
  let current;
  let if_block = (
    /*label*/
    ctx[5] && create_if_block(ctx)
  );
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[10].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[10].i18n
    ) },
    /*loading_status*/
    ctx[4]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[20]
  );
  json = new JSON({
    props: {
      value: (
        /*value*/
        ctx[3]
      ),
      open: (
        /*open*/
        ctx[11]
      ),
      theme_mode: (
        /*theme_mode*/
        ctx[12]
      ),
      show_indices: (
        /*show_indices*/
        ctx[13]
      ),
      label_height: (
        /*label_height*/
        ctx[17]
      )
    }
  });
  return {
    c() {
      div = element("div");
      if (if_block)
        if_block.c();
      t0 = space();
      create_component(statustracker.$$.fragment);
      t1 = space();
      create_component(json.$$.fragment);
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", {});
      var div_nodes = children(div);
      if (if_block)
        if_block.l(div_nodes);
      div_nodes.forEach(detach);
      t0 = claim_space(nodes);
      claim_component(statustracker.$$.fragment, nodes);
      t1 = claim_space(nodes);
      claim_component(json.$$.fragment, nodes);
      this.h();
    },
    h() {
      add_render_callback(() => (
        /*div_elementresize_handler*/
        ctx[19].call(div)
      ));
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      if (if_block)
        if_block.m(div, null);
      div_resize_listener = add_iframe_resize_listener(
        div,
        /*div_elementresize_handler*/
        ctx[19].bind(div)
      );
      insert_hydration(target, t0, anchor);
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t1, anchor);
      mount_component(json, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*label*/
        ctx2[5]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*label*/
          32) {
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
      const statustracker_changes = dirty & /*gradio, loading_status*/
      1040 ? get_spread_update(statustracker_spread_levels, [
        dirty & /*gradio*/
        1024 && {
          autoscroll: (
            /*gradio*/
            ctx2[10].autoscroll
          )
        },
        dirty & /*gradio*/
        1024 && { i18n: (
          /*gradio*/
          ctx2[10].i18n
        ) },
        dirty & /*loading_status*/
        16 && get_spread_object(
          /*loading_status*/
          ctx2[4]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const json_changes = {};
      if (dirty & /*value*/
      8)
        json_changes.value = /*value*/
        ctx2[3];
      if (dirty & /*open*/
      2048)
        json_changes.open = /*open*/
        ctx2[11];
      if (dirty & /*theme_mode*/
      4096)
        json_changes.theme_mode = /*theme_mode*/
        ctx2[12];
      if (dirty & /*show_indices*/
      8192)
        json_changes.show_indices = /*show_indices*/
        ctx2[13];
      if (dirty & /*label_height*/
      131072)
        json_changes.label_height = /*label_height*/
        ctx2[17];
      json.$set(json_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      transition_in(statustracker.$$.fragment, local);
      transition_in(json.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      transition_out(statustracker.$$.fragment, local);
      transition_out(json.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
        detach(t0);
        detach(t1);
      }
      if (if_block)
        if_block.d();
      div_resize_listener();
      destroy_component(statustracker, detaching);
      destroy_component(json, detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let current;
  block = new Block({
    props: {
      visible: (
        /*visible*/
        ctx[2]
      ),
      test_id: "json",
      elem_id: (
        /*elem_id*/
        ctx[0]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[1]
      ),
      container: (
        /*container*/
        ctx[7]
      ),
      scale: (
        /*scale*/
        ctx[8]
      ),
      min_width: (
        /*min_width*/
        ctx[9]
      ),
      padding: false,
      allow_overflow: true,
      overflow_behavior: "auto",
      height: (
        /*height*/
        ctx[14]
      ),
      min_height: (
        /*min_height*/
        ctx[15]
      ),
      max_height: (
        /*max_height*/
        ctx[16]
      ),
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(block.$$.fragment);
    },
    l(nodes) {
      claim_component(block.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(block, target, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      const block_changes = {};
      if (dirty & /*visible*/
      4)
        block_changes.visible = /*visible*/
        ctx2[2];
      if (dirty & /*elem_id*/
      1)
        block_changes.elem_id = /*elem_id*/
        ctx2[0];
      if (dirty & /*elem_classes*/
      2)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[1];
      if (dirty & /*container*/
      128)
        block_changes.container = /*container*/
        ctx2[7];
      if (dirty & /*scale*/
      256)
        block_changes.scale = /*scale*/
        ctx2[8];
      if (dirty & /*min_width*/
      512)
        block_changes.min_width = /*min_width*/
        ctx2[9];
      if (dirty & /*height*/
      16384)
        block_changes.height = /*height*/
        ctx2[14];
      if (dirty & /*min_height*/
      32768)
        block_changes.min_height = /*min_height*/
        ctx2[15];
      if (dirty & /*max_height*/
      65536)
        block_changes.max_height = /*max_height*/
        ctx2[16];
      if (dirty & /*$$scope, value, open, theme_mode, show_indices, label_height, gradio, loading_status, show_label, label, container*/
      2243832) {
        block_changes.$$scope = { dirty, ctx: ctx2 };
      }
      block.$set(block_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(block.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(block.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(block, detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
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
  function div_elementresize_handler() {
    label_height = this.clientHeight;
    $$invalidate(17, label_height);
  }
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  $$self.$$set = ($$props2) => {
    if ("elem_id" in $$props2)
      $$invalidate(0, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(1, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(2, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(3, value = $$props2.value);
    if ("loading_status" in $$props2)
      $$invalidate(4, loading_status = $$props2.loading_status);
    if ("label" in $$props2)
      $$invalidate(5, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(6, show_label = $$props2.show_label);
    if ("container" in $$props2)
      $$invalidate(7, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(8, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(9, min_width = $$props2.min_width);
    if ("gradio" in $$props2)
      $$invalidate(10, gradio = $$props2.gradio);
    if ("open" in $$props2)
      $$invalidate(11, open = $$props2.open);
    if ("theme_mode" in $$props2)
      $$invalidate(12, theme_mode = $$props2.theme_mode);
    if ("show_indices" in $$props2)
      $$invalidate(13, show_indices = $$props2.show_indices);
    if ("height" in $$props2)
      $$invalidate(14, height = $$props2.height);
    if ("min_height" in $$props2)
      $$invalidate(15, min_height = $$props2.min_height);
    if ("max_height" in $$props2)
      $$invalidate(16, max_height = $$props2.max_height);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*value, old_value, gradio*/
    263176) {
      {
        if (value !== old_value) {
          $$invalidate(18, old_value = value);
          gradio.dispatch("change");
        }
      }
    }
  };
  return [
    elem_id,
    elem_classes,
    visible,
    value,
    loading_status,
    label,
    show_label,
    container,
    scale,
    min_width,
    gradio,
    open,
    theme_mode,
    show_indices,
    height,
    min_height,
    max_height,
    label_height,
    old_value,
    div_elementresize_handler,
    clear_status_handler
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      elem_id: 0,
      elem_classes: 1,
      visible: 2,
      value: 3,
      loading_status: 4,
      label: 5,
      show_label: 6,
      container: 7,
      scale: 8,
      min_width: 9,
      gradio: 10,
      open: 11,
      theme_mode: 12,
      show_indices: 13,
      height: 14,
      min_height: 15,
      max_height: 16
    });
  }
}
export {
  JSON as BaseJSON,
  Index as default
};
