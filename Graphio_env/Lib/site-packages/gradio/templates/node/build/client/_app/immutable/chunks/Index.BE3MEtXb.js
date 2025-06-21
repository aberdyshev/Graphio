import { SvelteComponent, init, safe_not_equal, create_component, claim_component, mount_component, transition_in, transition_out, destroy_component, afterUpdate, assign, binding_callbacks, bind, space, claim_space, insert_hydration, get_spread_update, get_spread_object, group_outros, check_outros, add_flush_callback, detach } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, S as Static, C as Checkbox, j as Info } from "./2.CWSJHAMp.js";
function create_if_block(ctx) {
  let info_1;
  let current;
  info_1 = new Info({ props: { info: (
    /*info*/
    ctx[5]
  ) } });
  return {
    c() {
      create_component(info_1.$$.fragment);
    },
    l(nodes) {
      claim_component(info_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(info_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const info_1_changes = {};
      if (dirty & /*info*/
      32)
        info_1_changes.info = /*info*/
        ctx2[5];
      info_1.$set(info_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(info_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(info_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(info_1, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let statustracker;
  let t0;
  let t1;
  let basecheckbox;
  let updating_value;
  let current;
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
    ctx[9]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[14]
  );
  let if_block = (
    /*info*/
    ctx[5] && create_if_block(ctx)
  );
  function basecheckbox_value_binding(value) {
    ctx[15](value);
  }
  let basecheckbox_props = {
    label: (
      /*label*/
      ctx[4]
    ),
    interactive: (
      /*interactive*/
      ctx[11]
    )
  };
  if (
    /*value*/
    ctx[0] !== void 0
  ) {
    basecheckbox_props.value = /*value*/
    ctx[0];
  }
  basecheckbox = new Checkbox({ props: basecheckbox_props });
  binding_callbacks.push(() => bind(basecheckbox, "value", basecheckbox_value_binding));
  basecheckbox.$on(
    "change",
    /*handle_change*/
    ctx[12]
  );
  basecheckbox.$on(
    "select",
    /*select_handler*/
    ctx[16]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t0 = space();
      if (if_block)
        if_block.c();
      t1 = space();
      create_component(basecheckbox.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t0 = claim_space(nodes);
      if (if_block)
        if_block.l(nodes);
      t1 = claim_space(nodes);
      claim_component(basecheckbox.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t0, anchor);
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t1, anchor);
      mount_component(basecheckbox, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty & /*gradio, loading_status*/
      1536 ? get_spread_update(statustracker_spread_levels, [
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
        512 && get_spread_object(
          /*loading_status*/
          ctx2[9]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      if (
        /*info*/
        ctx2[5]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*info*/
          32) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(t1.parentNode, t1);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      const basecheckbox_changes = {};
      if (dirty & /*label*/
      16)
        basecheckbox_changes.label = /*label*/
        ctx2[4];
      if (dirty & /*interactive*/
      2048)
        basecheckbox_changes.interactive = /*interactive*/
        ctx2[11];
      if (!updating_value && dirty & /*value*/
      1) {
        updating_value = true;
        basecheckbox_changes.value = /*value*/
        ctx2[0];
        add_flush_callback(() => updating_value = false);
      }
      basecheckbox.$set(basecheckbox_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(if_block);
      transition_in(basecheckbox.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(if_block);
      transition_out(basecheckbox.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t0);
        detach(t1);
      }
      destroy_component(statustracker, detaching);
      if (if_block)
        if_block.d(detaching);
      destroy_component(basecheckbox, detaching);
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
        ctx[3]
      ),
      elem_id: (
        /*elem_id*/
        ctx[1]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[2]
      ),
      container: (
        /*container*/
        ctx[6]
      ),
      scale: (
        /*scale*/
        ctx[7]
      ),
      min_width: (
        /*min_width*/
        ctx[8]
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
      8)
        block_changes.visible = /*visible*/
        ctx2[3];
      if (dirty & /*elem_id*/
      2)
        block_changes.elem_id = /*elem_id*/
        ctx2[1];
      if (dirty & /*elem_classes*/
      4)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[2];
      if (dirty & /*container*/
      64)
        block_changes.container = /*container*/
        ctx2[6];
      if (dirty & /*scale*/
      128)
        block_changes.scale = /*scale*/
        ctx2[7];
      if (dirty & /*min_width*/
      256)
        block_changes.min_width = /*min_width*/
        ctx2[8];
      if (dirty & /*$$scope, label, interactive, value, gradio, info, loading_status*/
      134705) {
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
  let { value = false } = $$props;
  let { value_is_output = false } = $$props;
  let { label = "Checkbox" } = $$props;
  let { info = void 0 } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { loading_status } = $$props;
  let { gradio } = $$props;
  let { interactive } = $$props;
  function handle_change() {
    gradio.dispatch("change");
    if (!value_is_output) {
      gradio.dispatch("input");
    }
  }
  afterUpdate(() => {
    $$invalidate(13, value_is_output = false);
  });
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  function basecheckbox_value_binding(value$1) {
    value = value$1;
    $$invalidate(0, value);
  }
  const select_handler = (e) => gradio.dispatch("select", e.detail);
  $$self.$$set = ($$props2) => {
    if ("elem_id" in $$props2)
      $$invalidate(1, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(2, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(3, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("value_is_output" in $$props2)
      $$invalidate(13, value_is_output = $$props2.value_is_output);
    if ("label" in $$props2)
      $$invalidate(4, label = $$props2.label);
    if ("info" in $$props2)
      $$invalidate(5, info = $$props2.info);
    if ("container" in $$props2)
      $$invalidate(6, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(7, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(8, min_width = $$props2.min_width);
    if ("loading_status" in $$props2)
      $$invalidate(9, loading_status = $$props2.loading_status);
    if ("gradio" in $$props2)
      $$invalidate(10, gradio = $$props2.gradio);
    if ("interactive" in $$props2)
      $$invalidate(11, interactive = $$props2.interactive);
  };
  return [
    value,
    elem_id,
    elem_classes,
    visible,
    label,
    info,
    container,
    scale,
    min_width,
    loading_status,
    gradio,
    interactive,
    handle_change,
    value_is_output,
    clear_status_handler,
    basecheckbox_value_binding,
    select_handler
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      elem_id: 1,
      elem_classes: 2,
      visible: 3,
      value: 0,
      value_is_output: 13,
      label: 4,
      info: 5,
      container: 6,
      scale: 7,
      min_width: 8,
      loading_status: 9,
      gradio: 10,
      interactive: 11
    });
  }
}
export {
  Checkbox as BaseCheckbox,
  Index as default
};
