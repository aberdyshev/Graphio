const __vite__fileDeps=["./vega-embed.module.GWvvoORA.js","./vega-tooltip.module.DwwcB-rx.js","./2.CWSJHAMp.js","./preload-helper.DpQnamwV.js","./stores.BS_SVy0_.js","./client.DUstWztb.js","../assets/2.CVnuSI86.css","./time.yxtfVXXq.js","./step.B19URGTo.js","./linear.DiiEIHGV.js","./init.B8eB4uja.js","./dsv.CKxpOsHB.js","./range.Bs5oKbOs.js","./ordinal.Bkur82De.js","./arc.CI3-aWN2.js","./dispatch.C88ITL0L.js"],__vite__mapDeps=i=>i.map(i=>__vite__fileDeps[i]);
import { _ as __vitePreload } from "./preload-helper.DpQnamwV.js";
import { SvelteComponent, init, safe_not_equal, binding_callbacks, bind, create_component, claim_component, mount_component, add_flush_callback, transition_in, transition_out, destroy_component, onMount, space, empty, claim_space, insert_hydration, group_outros, check_outros, detach, assign, get_spread_update, get_spread_object, text, claim_text, set_data, element, claim_element, children, attr, noop, append_hydration } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, z as BlockTitle, S as Static } from "./2.CWSJHAMp.js";
import { E as Empty } from "./Empty.BC4b_-zA.js";
import { L as LineChart } from "./LineChart.D-OPS8mj.js";
import { I as IconButtonWrapper } from "./IconButtonWrapper.DkpBG_NJ.js";
import { F as FullscreenButton } from "./FullscreenButton.C05mBOIB.js";
function create_if_block_3(ctx) {
  let statustracker;
  let current;
  const statustracker_spread_levels = [
    { autoscroll: (
      /*gradio*/
      ctx[3].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      ctx[3].i18n
    ) },
    /*loading_status*/
    ctx[11]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[48]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      2056 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        8 && { autoscroll: (
          /*gradio*/
          ctx2[3].autoscroll
        ) },
        dirty[0] & /*gradio*/
        8 && { i18n: (
          /*gradio*/
          ctx2[3].i18n
        ) },
        dirty[0] & /*loading_status*/
        2048 && get_spread_object(
          /*loading_status*/
          ctx2[11]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(statustracker, detaching);
    }
  };
}
function create_if_block_2(ctx) {
  let iconbuttonwrapper;
  let current;
  iconbuttonwrapper = new IconButtonWrapper({
    props: {
      $$slots: { default: [create_default_slot_3] },
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
      if (dirty[0] & /*fullscreen*/
      8192 | dirty[2] & /*$$scope*/
      16) {
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
function create_default_slot_3(ctx) {
  let fullscreenbutton;
  let current;
  fullscreenbutton = new FullscreenButton({
    props: { fullscreen: (
      /*fullscreen*/
      ctx[13]
    ) }
  });
  fullscreenbutton.$on(
    "fullscreen",
    /*fullscreen_handler*/
    ctx[49]
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
      if (dirty[0] & /*fullscreen*/
      8192)
        fullscreenbutton_changes.fullscreen = /*fullscreen*/
        ctx2[13];
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
function create_default_slot_2(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*label*/
        ctx[4]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*label*/
        ctx[4]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      16)
        set_data(
          t,
          /*label*/
          ctx2[4]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_else_block(ctx) {
  let empty_1;
  let current;
  empty_1 = new Empty({
    props: {
      unpadded_box: true,
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(empty_1.$$.fragment);
    },
    l(nodes) {
      claim_component(empty_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(empty_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const empty_1_changes = {};
      if (dirty[2] & /*$$scope*/
      16) {
        empty_1_changes.$$scope = { dirty, ctx: ctx2 };
      }
      empty_1.$set(empty_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(empty_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(empty_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(empty_1, detaching);
    }
  };
}
function create_if_block(ctx) {
  let div;
  let t;
  let if_block_anchor;
  let if_block = (
    /*caption*/
    ctx[1] && create_if_block_1(ctx)
  );
  return {
    c() {
      div = element("div");
      t = space();
      if (if_block)
        if_block.c();
      if_block_anchor = empty();
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      children(div).forEach(detach);
      t = claim_space(nodes);
      if (if_block)
        if_block.l(nodes);
      if_block_anchor = empty();
      this.h();
    },
    h() {
      attr(div, "class", "svelte-19qacdz");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      ctx[50](div);
      insert_hydration(target, t, anchor);
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
    },
    p(ctx2, dirty) {
      if (
        /*caption*/
        ctx2[1]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
        } else {
          if_block = create_if_block_1(ctx2);
          if_block.c();
          if_block.m(if_block_anchor.parentNode, if_block_anchor);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(div);
        detach(t);
        detach(if_block_anchor);
      }
      ctx[50](null);
      if (if_block)
        if_block.d(detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let labelicon;
  let current;
  labelicon = new LineChart({});
  return {
    c() {
      create_component(labelicon.$$.fragment);
    },
    l(nodes) {
      claim_component(labelicon.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(labelicon, target, anchor);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(labelicon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(labelicon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(labelicon, detaching);
    }
  };
}
function create_if_block_1(ctx) {
  let p;
  let t;
  return {
    c() {
      p = element("p");
      t = text(
        /*caption*/
        ctx[1]
      );
      this.h();
    },
    l(nodes) {
      p = claim_element(nodes, "P", { class: true });
      var p_nodes = children(p);
      t = claim_text(
        p_nodes,
        /*caption*/
        ctx[1]
      );
      p_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(p, "class", "caption svelte-19qacdz");
    },
    m(target, anchor) {
      insert_hydration(target, p, anchor);
      append_hydration(p, t);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*caption*/
      2)
        set_data(
          t,
          /*caption*/
          ctx2[1]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(p);
      }
    }
  };
}
function create_default_slot(ctx) {
  let t0;
  let t1;
  let blocktitle;
  let t2;
  let current_block_type_index;
  let if_block2;
  let if_block2_anchor;
  let current;
  let if_block0 = (
    /*loading_status*/
    ctx[11] && create_if_block_3(ctx)
  );
  let if_block1 = (
    /*show_fullscreen_button*/
    ctx[2] && create_if_block_2(ctx)
  );
  blocktitle = new BlockTitle({
    props: {
      show_label: (
        /*show_label*/
        ctx[8]
      ),
      info: void 0,
      $$slots: { default: [create_default_slot_2] },
      $$scope: { ctx }
    }
  });
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*value*/
      ctx2[0] && /*is_browser*/
      ctx2[15]
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block2 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      if (if_block0)
        if_block0.c();
      t0 = space();
      if (if_block1)
        if_block1.c();
      t1 = space();
      create_component(blocktitle.$$.fragment);
      t2 = space();
      if_block2.c();
      if_block2_anchor = empty();
    },
    l(nodes) {
      if (if_block0)
        if_block0.l(nodes);
      t0 = claim_space(nodes);
      if (if_block1)
        if_block1.l(nodes);
      t1 = claim_space(nodes);
      claim_component(blocktitle.$$.fragment, nodes);
      t2 = claim_space(nodes);
      if_block2.l(nodes);
      if_block2_anchor = empty();
    },
    m(target, anchor) {
      if (if_block0)
        if_block0.m(target, anchor);
      insert_hydration(target, t0, anchor);
      if (if_block1)
        if_block1.m(target, anchor);
      insert_hydration(target, t1, anchor);
      mount_component(blocktitle, target, anchor);
      insert_hydration(target, t2, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block2_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*loading_status*/
        ctx2[11]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty[0] & /*loading_status*/
          2048) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_3(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(t0.parentNode, t0);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*show_fullscreen_button*/
        ctx2[2]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty[0] & /*show_fullscreen_button*/
          4) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block_2(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(t1.parentNode, t1);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
      const blocktitle_changes = {};
      if (dirty[0] & /*show_label*/
      256)
        blocktitle_changes.show_label = /*show_label*/
        ctx2[8];
      if (dirty[0] & /*label*/
      16 | dirty[2] & /*$$scope*/
      16) {
        blocktitle_changes.$$scope = { dirty, ctx: ctx2 };
      }
      blocktitle.$set(blocktitle_changes);
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx2, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block2 = if_blocks[current_block_type_index];
        if (!if_block2) {
          if_block2 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block2.c();
        } else {
          if_block2.p(ctx2, dirty);
        }
        transition_in(if_block2, 1);
        if_block2.m(if_block2_anchor.parentNode, if_block2_anchor);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block0);
      transition_in(if_block1);
      transition_in(blocktitle.$$.fragment, local);
      transition_in(if_block2);
      current = true;
    },
    o(local) {
      transition_out(if_block0);
      transition_out(if_block1);
      transition_out(blocktitle.$$.fragment, local);
      transition_out(if_block2);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t0);
        detach(t1);
        detach(t2);
        detach(if_block2_anchor);
      }
      if (if_block0)
        if_block0.d(detaching);
      if (if_block1)
        if_block1.d(detaching);
      destroy_component(blocktitle, detaching);
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let updating_fullscreen;
  let current;
  function block_fullscreen_binding(value) {
    ctx[51](value);
  }
  let block_props = {
    visible: (
      /*visible*/
      ctx[7]
    ),
    elem_id: (
      /*elem_id*/
      ctx[5]
    ),
    elem_classes: (
      /*elem_classes*/
      ctx[6]
    ),
    scale: (
      /*scale*/
      ctx[9]
    ),
    min_width: (
      /*min_width*/
      ctx[10]
    ),
    allow_overflow: false,
    padding: true,
    height: (
      /*height*/
      ctx[12]
    ),
    $$slots: { default: [create_default_slot] },
    $$scope: { ctx }
  };
  if (
    /*fullscreen*/
    ctx[13] !== void 0
  ) {
    block_props.fullscreen = /*fullscreen*/
    ctx[13];
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
    p(ctx2, dirty) {
      const block_changes = {};
      if (dirty[0] & /*visible*/
      128)
        block_changes.visible = /*visible*/
        ctx2[7];
      if (dirty[0] & /*elem_id*/
      32)
        block_changes.elem_id = /*elem_id*/
        ctx2[5];
      if (dirty[0] & /*elem_classes*/
      64)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[6];
      if (dirty[0] & /*scale*/
      512)
        block_changes.scale = /*scale*/
        ctx2[9];
      if (dirty[0] & /*min_width*/
      1024)
        block_changes.min_width = /*min_width*/
        ctx2[10];
      if (dirty[0] & /*height*/
      4096)
        block_changes.height = /*height*/
        ctx2[12];
      if (dirty[0] & /*caption, chart_element, value, show_label, label, fullscreen, show_fullscreen_button, gradio, loading_status*/
      26911 | dirty[2] & /*$$scope*/
      16) {
        block_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_fullscreen && dirty[0] & /*fullscreen*/
      8192) {
        updating_fullscreen = true;
        block_changes.fullscreen = /*fullscreen*/
        ctx2[13];
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
  let unique_colors;
  let x_start;
  let x_end;
  let y_start;
  let y_end;
  let _sort;
  let x_temporal;
  let _x_lim;
  let computed_style;
  let _color_map;
  let { value } = $$props;
  let { x } = $$props;
  let { y } = $$props;
  let { color = null } = $$props;
  let { title = null } = $$props;
  let { x_title = null } = $$props;
  let { y_title = null } = $$props;
  let { color_title = null } = $$props;
  let { x_bin = null } = $$props;
  let { y_aggregate = void 0 } = $$props;
  let { color_map = null } = $$props;
  let { x_lim = null } = $$props;
  let { y_lim = null } = $$props;
  let { x_label_angle = null } = $$props;
  let { y_label_angle = null } = $$props;
  let { x_axis_labels_visible = true } = $$props;
  let { caption = null } = $$props;
  let { sort = null } = $$props;
  let { tooltip = "axis" } = $$props;
  let { show_fullscreen_button = false } = $$props;
  let fullscreen = false;
  function reformat_sort(_sort2) {
    if (_sort2 === "x") {
      return "ascending";
    } else if (_sort2 === "-x") {
      return "descending";
    } else if (_sort2 === "y") {
      return { field: y, order: "ascending" };
    } else if (_sort2 === "-y") {
      return { field: y, order: "descending" };
    } else if (_sort2 === null) {
      return null;
    } else if (Array.isArray(_sort2)) {
      return _sort2;
    }
  }
  let { _selectable = false } = $$props;
  let _data;
  let { gradio } = $$props;
  let _x_bin;
  let mouse_down_on_chart = false;
  const SUFFIX_DURATION = { s: 1, m: 60, h: 60 * 60, d: 24 * 60 * 60 };
  let _y_aggregate;
  let aggregating;
  function reformat_data(data) {
    if (tooltip == "all" || Array.isArray(tooltip)) {
      return data.data.map((row) => {
        const obj = {};
        data.columns.forEach((col, i) => {
          obj[col] = row[i];
        });
        return obj;
      });
    }
    let x_index = data.columns.indexOf(x);
    let y_index = data.columns.indexOf(y);
    let color_index = color ? data.columns.indexOf(color) : null;
    return data.data.map((row) => {
      const obj = { [x]: row[x_index], [y]: row[y_index] };
      if (color && color_index !== null) {
        obj[color] = row[color_index];
      }
      return obj;
    });
  }
  let old_value = value;
  const is_browser = typeof window !== "undefined";
  let chart_element;
  let view;
  let mounted = false;
  let old_width;
  let old_height;
  let resizeObserver;
  let vegaEmbed;
  async function load_chart() {
    if (mouse_down_on_chart) {
      refresh_pending = true;
      return;
    }
    if (view) {
      view.finalize();
    }
    if (!value || !chart_element)
      return;
    old_width = chart_element.offsetWidth;
    old_height = chart_element.offsetHeight;
    const spec = create_vega_lite_spec();
    if (!spec)
      return;
    resizeObserver = new ResizeObserver((el) => {
      if (!el[0].target || !(el[0].target instanceof HTMLElement))
        return;
      if (old_width === 0 && chart_element.offsetWidth !== 0 && value.datatypes[x] === "nominal") {
        load_chart();
      } else {
        view.signal("width", el[0].target.offsetWidth).run();
      }
      if (old_height !== el[0].target.offsetHeight && fullscreen) {
        view.signal("height", el[0].target.offsetHeight).run();
        old_height = el[0].target.offsetHeight;
      }
    });
    if (!vegaEmbed) {
      vegaEmbed = (await __vitePreload(() => import("./vega-embed.module.GWvvoORA.js"), true ? __vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) : void 0, import.meta.url)).default;
    }
    vegaEmbed(chart_element, spec, { actions: false }).then(function(result) {
      $$invalidate(39, view = result.view);
      resizeObserver.observe(chart_element);
      var debounceTimeout;
      view.addEventListener("dblclick", () => {
        gradio.dispatch("double_click");
      });
      chart_element.addEventListener(
        "mousedown",
        function(e) {
          if (e.detail > 1) {
            e.preventDefault();
          }
        },
        false
      );
      if (_selectable) {
        view.addSignalListener("brush", function(_, value2) {
          mouse_down_on_chart = true;
          if (Object.keys(value2).length === 0)
            return;
          clearTimeout(debounceTimeout);
          let range = value2[Object.keys(value2)[0]];
          if (x_temporal) {
            range = [range[0] / 1e3, range[1] / 1e3];
          }
          debounceTimeout = setTimeout(
            function() {
              mouse_down_on_chart = false;
              gradio.dispatch("select", {
                value: range,
                index: range,
                selected: true
              });
              if (refresh_pending) {
                refresh_pending = false;
                load_chart();
              }
            },
            250
          );
        });
      }
    });
  }
  let refresh_pending = false;
  onMount(() => {
    $$invalidate(40, mounted = true);
    return () => {
      $$invalidate(40, mounted = false);
      if (view) {
        view.finalize();
      }
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  });
  function create_vega_lite_spec() {
    if (!value || !computed_style)
      return null;
    let accent_color = computed_style.getPropertyValue("--color-accent");
    let body_text_color = computed_style.getPropertyValue("--body-text-color");
    let borderColorPrimary = computed_style.getPropertyValue("--border-color-primary");
    let font_family = computed_style.fontFamily;
    let title_weight = computed_style.getPropertyValue("--block-title-text-weight");
    const font_to_px_val = (font) => {
      return font.endsWith("px") ? parseFloat(font.slice(0, -2)) : 12;
    };
    let text_size_md = font_to_px_val(computed_style.getPropertyValue("--text-md"));
    let text_size_sm = font_to_px_val(computed_style.getPropertyValue("--text-sm"));
    return {
      $schema: "https://vega.github.io/schema/vega-lite/v5.17.0.json",
      background: "transparent",
      config: {
        autosize: { type: "fit", contains: "padding" },
        axis: {
          labelFont: font_family,
          labelColor: body_text_color,
          titleFont: font_family,
          titleColor: body_text_color,
          titlePadding: 8,
          tickColor: borderColorPrimary,
          labelFontSize: text_size_sm,
          gridColor: borderColorPrimary,
          titleFontWeight: "normal",
          titleFontSize: text_size_sm,
          labelFontWeight: "normal",
          domain: false,
          labelAngle: 0
        },
        legend: {
          labelColor: body_text_color,
          labelFont: font_family,
          titleColor: body_text_color,
          titleFont: font_family,
          titleFontWeight: "normal",
          titleFontSize: text_size_sm,
          labelFontWeight: "normal",
          offset: 2
        },
        title: {
          color: body_text_color,
          font: font_family,
          fontSize: text_size_md,
          fontWeight: title_weight,
          anchor: "middle"
        },
        view: { stroke: borderColorPrimary },
        mark: {
          stroke: value.mark !== "bar" ? accent_color : void 0,
          fill: value.mark === "bar" ? accent_color : void 0,
          cursor: "crosshair"
        }
      },
      data: { name: "data" },
      datasets: { data: _data },
      layer: ["plot", ...value.mark === "line" ? ["hover"] : []].map((mode) => {
        return {
          encoding: {
            size: value.mark === "line" ? mode == "plot" ? {
              condition: {
                empty: false,
                param: "hoverPlot",
                value: 3
              },
              value: 2
            } : {
              condition: { empty: false, param: "hover", value: 100 },
              value: 0
            } : void 0,
            opacity: mode === "plot" ? void 0 : {
              condition: { empty: false, param: "hover", value: 1 },
              value: 0
            },
            x: {
              axis: {
                ...x_label_angle !== null && { labelAngle: x_label_angle },
                labels: x_axis_labels_visible,
                ticks: x_axis_labels_visible
              },
              field: x,
              title: x_title || x,
              type: value.datatypes[x],
              scale: _x_lim ? { domain: _x_lim } : void 0,
              bin: _x_bin ? { step: _x_bin } : void 0,
              sort: _sort
            },
            y: {
              axis: y_label_angle ? { labelAngle: y_label_angle } : {},
              field: y,
              title: y_title || y,
              type: value.datatypes[y],
              scale: y_lim ? { domain: y_lim } : void 0,
              aggregate: aggregating ? _y_aggregate : void 0
            },
            color: color ? {
              field: color,
              legend: { orient: "bottom", title: color_title },
              scale: value.datatypes[color] === "nominal" ? {
                domain: unique_colors,
                range: color_map ? unique_colors.map((c) => color_map[c]) : void 0
              } : {
                range: [100, 200, 300, 400, 500, 600, 700, 800, 900].map((n) => computed_style.getPropertyValue("--primary-" + n)),
                interpolate: "hsl"
              },
              type: value.datatypes[color]
            } : void 0,
            tooltip: tooltip == "none" ? void 0 : [
              {
                field: y,
                type: value.datatypes[y],
                aggregate: aggregating ? _y_aggregate : void 0,
                title: y_title || y
              },
              {
                field: x,
                type: value.datatypes[x],
                title: x_title || x,
                format: x_temporal ? "%Y-%m-%d %H:%M:%S" : void 0,
                bin: _x_bin ? { step: _x_bin } : void 0
              },
              ...color ? [
                {
                  field: color,
                  type: value.datatypes[color]
                }
              ] : [],
              ...tooltip === "axis" ? [] : value == null ? void 0 : value.columns.filter((col) => col !== x && col !== y && col !== color && (tooltip === "all" || tooltip.includes(col))).map((column) => ({
                field: column,
                type: value.datatypes[column]
              }))
            ]
          },
          strokeDash: {},
          mark: {
            clip: true,
            type: mode === "hover" ? "point" : value.mark
          },
          name: mode
        };
      }),
      // @ts-ignore
      params: [
        ...value.mark === "line" ? [
          {
            name: "hoverPlot",
            select: {
              clear: "mouseout",
              fields: color ? [color] : [],
              nearest: true,
              on: "mouseover",
              type: "point"
            },
            views: ["hover"]
          },
          {
            name: "hover",
            select: {
              clear: "mouseout",
              nearest: true,
              on: "mouseover",
              type: "point"
            },
            views: ["hover"]
          }
        ] : [],
        ..._selectable ? [
          {
            name: "brush",
            select: {
              encodings: ["x"],
              mark: {
                fill: "gray",
                fillOpacity: 0.3,
                stroke: "none"
              },
              type: "interval"
            },
            views: ["plot"]
          }
        ] : []
      ],
      width: chart_element.offsetWidth,
      height: height || fullscreen ? "container" : void 0,
      title: title || void 0
    };
  }
  let { label = "Textbox" } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { show_label } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { loading_status = void 0 } = $$props;
  let { height = void 0 } = $$props;
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const fullscreen_handler = ({ detail }) => {
    $$invalidate(13, fullscreen = detail);
  };
  function div_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      chart_element = $$value;
      $$invalidate(14, chart_element);
    });
  }
  function block_fullscreen_binding(value2) {
    fullscreen = value2;
    $$invalidate(13, fullscreen);
  }
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("x" in $$props2)
      $$invalidate(18, x = $$props2.x);
    if ("y" in $$props2)
      $$invalidate(19, y = $$props2.y);
    if ("color" in $$props2)
      $$invalidate(20, color = $$props2.color);
    if ("title" in $$props2)
      $$invalidate(21, title = $$props2.title);
    if ("x_title" in $$props2)
      $$invalidate(22, x_title = $$props2.x_title);
    if ("y_title" in $$props2)
      $$invalidate(23, y_title = $$props2.y_title);
    if ("color_title" in $$props2)
      $$invalidate(24, color_title = $$props2.color_title);
    if ("x_bin" in $$props2)
      $$invalidate(25, x_bin = $$props2.x_bin);
    if ("y_aggregate" in $$props2)
      $$invalidate(26, y_aggregate = $$props2.y_aggregate);
    if ("color_map" in $$props2)
      $$invalidate(27, color_map = $$props2.color_map);
    if ("x_lim" in $$props2)
      $$invalidate(16, x_lim = $$props2.x_lim);
    if ("y_lim" in $$props2)
      $$invalidate(17, y_lim = $$props2.y_lim);
    if ("x_label_angle" in $$props2)
      $$invalidate(28, x_label_angle = $$props2.x_label_angle);
    if ("y_label_angle" in $$props2)
      $$invalidate(29, y_label_angle = $$props2.y_label_angle);
    if ("x_axis_labels_visible" in $$props2)
      $$invalidate(30, x_axis_labels_visible = $$props2.x_axis_labels_visible);
    if ("caption" in $$props2)
      $$invalidate(1, caption = $$props2.caption);
    if ("sort" in $$props2)
      $$invalidate(31, sort = $$props2.sort);
    if ("tooltip" in $$props2)
      $$invalidate(32, tooltip = $$props2.tooltip);
    if ("show_fullscreen_button" in $$props2)
      $$invalidate(2, show_fullscreen_button = $$props2.show_fullscreen_button);
    if ("_selectable" in $$props2)
      $$invalidate(33, _selectable = $$props2._selectable);
    if ("gradio" in $$props2)
      $$invalidate(3, gradio = $$props2.gradio);
    if ("label" in $$props2)
      $$invalidate(4, label = $$props2.label);
    if ("elem_id" in $$props2)
      $$invalidate(5, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(6, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(7, visible = $$props2.visible);
    if ("show_label" in $$props2)
      $$invalidate(8, show_label = $$props2.show_label);
    if ("scale" in $$props2)
      $$invalidate(9, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(10, min_width = $$props2.min_width);
    if ("loading_status" in $$props2)
      $$invalidate(11, loading_status = $$props2.loading_status);
    if ("height" in $$props2)
      $$invalidate(12, height = $$props2.height);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*value*/
    1) {
      $$invalidate(34, _data = value ? reformat_data(value) : []);
    }
    if ($$self.$$.dirty[0] & /*color, value*/
    1048577 | $$self.$$.dirty[1] & /*_data*/
    8) {
      unique_colors = color && value && value.datatypes[color] === "nominal" ? Array.from(new Set(_data.map((d) => d[color]))) : [];
    }
    if ($$self.$$.dirty[0] & /*x_lim*/
    65536) {
      $$invalidate(16, x_lim = x_lim || null);
    }
    if ($$self.$$.dirty[0] & /*y_lim*/
    131072) {
      $$invalidate(17, y_lim = y_lim || null);
    }
    if ($$self.$$.dirty[0] & /*x_lim*/
    65536) {
      $$invalidate(46, [x_start, x_end] = x_lim === null ? [void 0, void 0] : x_lim, x_start, ($$invalidate(45, x_end), $$invalidate(16, x_lim)));
    }
    if ($$self.$$.dirty[0] & /*y_lim*/
    131072) {
      $$invalidate(44, [y_start, y_end] = y_lim || [void 0, void 0], y_start, ($$invalidate(43, y_end), $$invalidate(17, y_lim)));
    }
    if ($$self.$$.dirty[1] & /*sort*/
    1) {
      _sort = reformat_sort(sort);
    }
    if ($$self.$$.dirty[0] & /*value, x*/
    262145) {
      $$invalidate(41, x_temporal = value && value.datatypes[x] === "temporal");
    }
    if ($$self.$$.dirty[0] & /*x_lim*/
    65536 | $$self.$$.dirty[1] & /*x_temporal*/
    1024) {
      _x_lim = x_lim && x_temporal ? [x_lim[0] * 1e3, x_lim[1] * 1e3] : x_lim;
    }
    if ($$self.$$.dirty[0] & /*x_bin*/
    33554432) {
      $$invalidate(35, _x_bin = x_bin ? typeof x_bin === "string" ? 1e3 * parseInt(x_bin.substring(0, x_bin.length - 1)) * SUFFIX_DURATION[x_bin[x_bin.length - 1]] : x_bin : void 0);
    }
    if ($$self.$$.dirty[0] & /*value, y_aggregate, x*/
    67371009 | $$self.$$.dirty[1] & /*_x_bin, aggregating*/
    80) {
      {
        if (value) {
          if (value.mark === "point") {
            $$invalidate(37, aggregating = _x_bin !== void 0);
            $$invalidate(36, _y_aggregate = y_aggregate || aggregating ? "sum" : void 0);
          } else {
            $$invalidate(37, aggregating = _x_bin !== void 0 || value.datatypes[x] === "nominal");
            $$invalidate(36, _y_aggregate = y_aggregate ? y_aggregate : "sum");
          }
        }
      }
    }
    if ($$self.$$.dirty[0] & /*value*/
    1 | $$self.$$.dirty[1] & /*old_value, view, _data*/
    392) {
      if (old_value !== value && view) {
        $$invalidate(38, old_value = value);
        view.data("data", _data).runAsync();
      }
    }
    if ($$self.$$.dirty[0] & /*chart_element*/
    16384) {
      $$invalidate(42, computed_style = chart_element ? window.getComputedStyle(chart_element) : null);
    }
    if ($$self.$$.dirty[0] & /*color_map*/
    134217728) {
      $$invalidate(47, _color_map = JSON.stringify(color_map));
    }
    if ($$self.$$.dirty[0] & /*title, x_title, y_title, color_title, x, y, color, x_bin, caption, chart_element, fullscreen*/
    66871298 | $$self.$$.dirty[1] & /*_y_aggregate, _color_map, x_start, x_end, y_start, y_end, sort, mounted, computed_style*/
    129569) {
      computed_style && requestAnimationFrame(load_chart);
    }
  };
  return [
    value,
    caption,
    show_fullscreen_button,
    gradio,
    label,
    elem_id,
    elem_classes,
    visible,
    show_label,
    scale,
    min_width,
    loading_status,
    height,
    fullscreen,
    chart_element,
    is_browser,
    x_lim,
    y_lim,
    x,
    y,
    color,
    title,
    x_title,
    y_title,
    color_title,
    x_bin,
    y_aggregate,
    color_map,
    x_label_angle,
    y_label_angle,
    x_axis_labels_visible,
    sort,
    tooltip,
    _selectable,
    _data,
    _x_bin,
    _y_aggregate,
    aggregating,
    old_value,
    view,
    mounted,
    x_temporal,
    computed_style,
    y_end,
    y_start,
    x_end,
    x_start,
    _color_map,
    clear_status_handler,
    fullscreen_handler,
    div_binding,
    block_fullscreen_binding
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance,
      create_fragment,
      safe_not_equal,
      {
        value: 0,
        x: 18,
        y: 19,
        color: 20,
        title: 21,
        x_title: 22,
        y_title: 23,
        color_title: 24,
        x_bin: 25,
        y_aggregate: 26,
        color_map: 27,
        x_lim: 16,
        y_lim: 17,
        x_label_angle: 28,
        y_label_angle: 29,
        x_axis_labels_visible: 30,
        caption: 1,
        sort: 31,
        tooltip: 32,
        show_fullscreen_button: 2,
        _selectable: 33,
        gradio: 3,
        label: 4,
        elem_id: 5,
        elem_classes: 6,
        visible: 7,
        show_label: 8,
        scale: 9,
        min_width: 10,
        loading_status: 11,
        height: 12
      },
      null,
      [-1, -1, -1]
    );
  }
}
export {
  Index as default
};
