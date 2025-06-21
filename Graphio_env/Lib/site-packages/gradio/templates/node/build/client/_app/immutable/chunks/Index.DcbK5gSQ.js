import { SvelteComponent, init, safe_not_equal, binding_callbacks, bind, create_component, claim_component, mount_component, add_flush_callback, transition_in, transition_out, destroy_component, assign, space, claim_space, insert_hydration, group_outros, check_outros, get_spread_update, get_spread_object, detach } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { P as Plot, a as Plot$1 } from "./Plot.D1qXL7qm.js";
import { B as Block, S as Static } from "./2.CWSJHAMp.js";
import { B as BlockLabel } from "./BlockLabel.Cft09z0E.js";
import { I as IconButtonWrapper } from "./IconButtonWrapper.DkpBG_NJ.js";
import { F as FullscreenButton } from "./FullscreenButton.C05mBOIB.js";
function create_if_block(ctx) {
  let iconbuttonwrapper;
  let current;
  iconbuttonwrapper = new IconButtonWrapper({
    props: {
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(iconbuttonwrapper.$$.fragment);
    },
    l(nodes) {
      claim_component(iconbuttonwrapper.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(iconbuttonwrapper, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const iconbuttonwrapper_changes = {};
      if (dirty & /*$$scope, fullscreen*/
      17039360) {
        iconbuttonwrapper_changes.$$scope = { dirty, ctx: ctx2 };
      }
      iconbuttonwrapper.$set(iconbuttonwrapper_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(iconbuttonwrapper.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(iconbuttonwrapper.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(iconbuttonwrapper, detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let fullscreenbutton;
  let current;
  fullscreenbutton = new FullscreenButton({
    props: { fullscreen: (
      /*fullscreen*/
      ctx[18]
    ) }
  });
  fullscreenbutton.$on(
    "fullscreen",
    /*fullscreen_handler*/
    ctx[19]
  );
  return {
    c() {
      create_component(fullscreenbutton.$$.fragment);
    },
    l(nodes) {
      claim_component(fullscreenbutton.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(fullscreenbutton, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const fullscreenbutton_changes = {};
      if (dirty & /*fullscreen*/
      262144)
        fullscreenbutton_changes.fullscreen = /*fullscreen*/
        ctx2[18];
      fullscreenbutton.$set(fullscreenbutton_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(fullscreenbutton.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(fullscreenbutton.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(fullscreenbutton, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let blocklabel;
  let t0;
  let t1;
  let statustracker;
  let t2;
  let plot;
  let current;
  blocklabel = new BlockLabel({
    props: {
      show_label: (
        /*show_label*/
        ctx[6]
      ),
      label: (
        /*label*/
        ctx[5] || /*gradio*/
        ctx[13].i18n("plot.plot")
      ),
      Icon: Plot
    }
  });
  let if_block = (
    /*show_fullscreen_button*/
    ctx[17] && create_if_block(ctx)
  );
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[13].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[13].i18n
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
  plot = new Plot$1({
    props: {
      value: (
        /*value*/
        ctx[0]
      ),
      theme_mode: (
        /*theme_mode*/
        ctx[10]
      ),
      caption: (
        /*caption*/
        ctx[11]
      ),
      bokeh_version: (
        /*bokeh_version*/
        ctx[12]
      ),
      show_actions_button: (
        /*show_actions_button*/
        ctx[14]
      ),
      gradio: (
        /*gradio*/
        ctx[13]
      ),
      show_label: (
        /*show_label*/
        ctx[6]
      ),
      _selectable: (
        /*_selectable*/
        ctx[15]
      ),
      x_lim: (
        /*x_lim*/
        ctx[16]
      )
    }
  });
  plot.$on(
    "change",
    /*change_handler*/
    ctx[21]
  );
  plot.$on(
    "select",
    /*select_handler*/
    ctx[22]
  );
  return {
    c() {
      create_component(blocklabel.$$.fragment);
      t0 = space();
      if (if_block)
        if_block.c();
      t1 = space();
      create_component(statustracker.$$.fragment);
      t2 = space();
      create_component(plot.$$.fragment);
    },
    l(nodes) {
      claim_component(blocklabel.$$.fragment, nodes);
      t0 = claim_space(nodes);
      if (if_block)
        if_block.l(nodes);
      t1 = claim_space(nodes);
      claim_component(statustracker.$$.fragment, nodes);
      t2 = claim_space(nodes);
      claim_component(plot.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(blocklabel, target, anchor);
      insert_hydration(target, t0, anchor);
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t1, anchor);
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t2, anchor);
      mount_component(plot, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const blocklabel_changes = {};
      if (dirty & /*show_label*/
      64)
        blocklabel_changes.show_label = /*show_label*/
        ctx2[6];
      if (dirty & /*label, gradio*/
      8224)
        blocklabel_changes.label = /*label*/
        ctx2[5] || /*gradio*/
        ctx2[13].i18n("plot.plot");
      blocklabel.$set(blocklabel_changes);
      if (
        /*show_fullscreen_button*/
        ctx2[17]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*show_fullscreen_button*/
          131072) {
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
      const statustracker_changes = dirty & /*gradio, loading_status*/
      8208 ? get_spread_update(statustracker_spread_levels, [
        dirty & /*gradio*/
        8192 && {
          autoscroll: (
            /*gradio*/
            ctx2[13].autoscroll
          )
        },
        dirty & /*gradio*/
        8192 && { i18n: (
          /*gradio*/
          ctx2[13].i18n
        ) },
        dirty & /*loading_status*/
        16 && get_spread_object(
          /*loading_status*/
          ctx2[4]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const plot_changes = {};
      if (dirty & /*value*/
      1)
        plot_changes.value = /*value*/
        ctx2[0];
      if (dirty & /*theme_mode*/
      1024)
        plot_changes.theme_mode = /*theme_mode*/
        ctx2[10];
      if (dirty & /*caption*/
      2048)
        plot_changes.caption = /*caption*/
        ctx2[11];
      if (dirty & /*bokeh_version*/
      4096)
        plot_changes.bokeh_version = /*bokeh_version*/
        ctx2[12];
      if (dirty & /*show_actions_button*/
      16384)
        plot_changes.show_actions_button = /*show_actions_button*/
        ctx2[14];
      if (dirty & /*gradio*/
      8192)
        plot_changes.gradio = /*gradio*/
        ctx2[13];
      if (dirty & /*show_label*/
      64)
        plot_changes.show_label = /*show_label*/
        ctx2[6];
      if (dirty & /*_selectable*/
      32768)
        plot_changes._selectable = /*_selectable*/
        ctx2[15];
      if (dirty & /*x_lim*/
      65536)
        plot_changes.x_lim = /*x_lim*/
        ctx2[16];
      plot.$set(plot_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(blocklabel.$$.fragment, local);
      transition_in(if_block);
      transition_in(statustracker.$$.fragment, local);
      transition_in(plot.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(blocklabel.$$.fragment, local);
      transition_out(if_block);
      transition_out(statustracker.$$.fragment, local);
      transition_out(plot.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t0);
        detach(t1);
        detach(t2);
      }
      destroy_component(blocklabel, detaching);
      if (if_block)
        if_block.d(detaching);
      destroy_component(statustracker, detaching);
      destroy_component(plot, detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let updating_fullscreen;
  let current;
  function block_fullscreen_binding(value) {
    ctx[23](value);
  }
  let block_props = {
    padding: false,
    elem_id: (
      /*elem_id*/
      ctx[1]
    ),
    elem_classes: (
      /*elem_classes*/
      ctx[2]
    ),
    visible: (
      /*visible*/
      ctx[3]
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
    allow_overflow: false,
    $$slots: { default: [create_default_slot] },
    $$scope: { ctx }
  };
  if (
    /*fullscreen*/
    ctx[18] !== void 0
  ) {
    block_props.fullscreen = /*fullscreen*/
    ctx[18];
  }
  block = new Block({ props: block_props });
  binding_callbacks.push(() => bind(block, "fullscreen", block_fullscreen_binding));
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
      if (dirty & /*elem_id*/
      2)
        block_changes.elem_id = /*elem_id*/
        ctx2[1];
      if (dirty & /*elem_classes*/
      4)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[2];
      if (dirty & /*visible*/
      8)
        block_changes.visible = /*visible*/
        ctx2[3];
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
      if (dirty & /*$$scope, value, theme_mode, caption, bokeh_version, show_actions_button, gradio, show_label, _selectable, x_lim, loading_status, fullscreen, show_fullscreen_button, label*/
      17300593) {
        block_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_fullscreen && dirty & /*fullscreen*/
      262144) {
        updating_fullscreen = true;
        block_changes.fullscreen = /*fullscreen*/
        ctx2[18];
        add_flush_callback(() => updating_fullscreen = false);
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
  let { value = null } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { loading_status } = $$props;
  let { label } = $$props;
  let { show_label } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { theme_mode } = $$props;
  let { caption } = $$props;
  let { bokeh_version } = $$props;
  let { gradio } = $$props;
  let { show_actions_button = false } = $$props;
  let { _selectable = false } = $$props;
  let { x_lim = null } = $$props;
  let { show_fullscreen_button = false } = $$props;
  let fullscreen = false;
  const fullscreen_handler = ({ detail }) => {
    $$invalidate(18, fullscreen = detail);
  };
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const change_handler = () => gradio.dispatch("change");
  const select_handler = (e) => gradio.dispatch("select", e.detail);
  function block_fullscreen_binding(value2) {
    fullscreen = value2;
    $$invalidate(18, fullscreen);
  }
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("elem_id" in $$props2)
      $$invalidate(1, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(2, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(3, visible = $$props2.visible);
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
    if ("theme_mode" in $$props2)
      $$invalidate(10, theme_mode = $$props2.theme_mode);
    if ("caption" in $$props2)
      $$invalidate(11, caption = $$props2.caption);
    if ("bokeh_version" in $$props2)
      $$invalidate(12, bokeh_version = $$props2.bokeh_version);
    if ("gradio" in $$props2)
      $$invalidate(13, gradio = $$props2.gradio);
    if ("show_actions_button" in $$props2)
      $$invalidate(14, show_actions_button = $$props2.show_actions_button);
    if ("_selectable" in $$props2)
      $$invalidate(15, _selectable = $$props2._selectable);
    if ("x_lim" in $$props2)
      $$invalidate(16, x_lim = $$props2.x_lim);
    if ("show_fullscreen_button" in $$props2)
      $$invalidate(17, show_fullscreen_button = $$props2.show_fullscreen_button);
  };
  return [
    value,
    elem_id,
    elem_classes,
    visible,
    loading_status,
    label,
    show_label,
    container,
    scale,
    min_width,
    theme_mode,
    caption,
    bokeh_version,
    gradio,
    show_actions_button,
    _selectable,
    x_lim,
    show_fullscreen_button,
    fullscreen,
    fullscreen_handler,
    clear_status_handler,
    change_handler,
    select_handler,
    block_fullscreen_binding
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      value: 0,
      elem_id: 1,
      elem_classes: 2,
      visible: 3,
      loading_status: 4,
      label: 5,
      show_label: 6,
      container: 7,
      scale: 8,
      min_width: 9,
      theme_mode: 10,
      caption: 11,
      bokeh_version: 12,
      gradio: 13,
      show_actions_button: 14,
      _selectable: 15,
      x_lim: 16,
      show_fullscreen_button: 17
    });
  }
}
export {
  Plot$1 as BasePlot,
  Index as default
};
