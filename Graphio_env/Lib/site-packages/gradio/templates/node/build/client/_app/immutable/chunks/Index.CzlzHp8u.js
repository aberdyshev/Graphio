import { SvelteComponent, init, safe_not_equal, binding_callbacks, bind, create_component, claim_component, mount_component, add_flush_callback, transition_in, transition_out, destroy_component, createEventDispatcher, assign, space, empty, claim_space, insert_hydration, get_spread_update, get_spread_object, group_outros, check_outros, detach } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, S as Static } from "./2.CWSJHAMp.js";
import { U as UploadText } from "./UploadText.r-ece2mU.js";
import Gallery from "./Gallery.ClDBYQib.js";
import { B as BaseFileUpload } from "./FileUpload.cQ2FySjZ.js";
/* empty css                                               */
function create_else_block(ctx) {
  let gallery;
  let updating_selected_index;
  let updating_value;
  let current;
  function gallery_selected_index_binding(value) {
    ctx[32](value);
  }
  function gallery_value_binding(value) {
    ctx[33](value);
  }
  let gallery_props = {
    label: (
      /*label*/
      ctx[5]
    ),
    show_label: (
      /*show_label*/
      ctx[4]
    ),
    columns: (
      /*columns*/
      ctx[14]
    ),
    rows: (
      /*rows*/
      ctx[15]
    ),
    height: (
      /*height*/
      ctx[16]
    ),
    preview: (
      /*preview*/
      ctx[17]
    ),
    object_fit: (
      /*object_fit*/
      ctx[19]
    ),
    interactive: (
      /*interactive*/
      ctx[21]
    ),
    allow_preview: (
      /*allow_preview*/
      ctx[18]
    ),
    show_share_button: (
      /*show_share_button*/
      ctx[20]
    ),
    show_download_button: (
      /*show_download_button*/
      ctx[22]
    ),
    i18n: (
      /*gradio*/
      ctx[23].i18n
    ),
    _fetch: (
      /*func_2*/
      ctx[31]
    ),
    show_fullscreen_button: (
      /*show_fullscreen_button*/
      ctx[24]
    ),
    fullscreen: (
      /*fullscreen*/
      ctx[3]
    )
  };
  if (
    /*selected_index*/
    ctx[1] !== void 0
  ) {
    gallery_props.selected_index = /*selected_index*/
    ctx[1];
  }
  if (
    /*value*/
    ctx[0] !== void 0
  ) {
    gallery_props.value = /*value*/
    ctx[0];
  }
  gallery = new Gallery({ props: gallery_props });
  binding_callbacks.push(() => bind(gallery, "selected_index", gallery_selected_index_binding));
  binding_callbacks.push(() => bind(gallery, "value", gallery_value_binding));
  gallery.$on(
    "change",
    /*change_handler*/
    ctx[34]
  );
  gallery.$on(
    "select",
    /*select_handler*/
    ctx[35]
  );
  gallery.$on(
    "share",
    /*share_handler*/
    ctx[36]
  );
  gallery.$on(
    "error",
    /*error_handler_1*/
    ctx[37]
  );
  gallery.$on(
    "preview_open",
    /*preview_open_handler*/
    ctx[38]
  );
  gallery.$on(
    "preview_close",
    /*preview_close_handler*/
    ctx[39]
  );
  gallery.$on(
    "fullscreen",
    /*fullscreen_handler*/
    ctx[40]
  );
  return {
    c() {
      create_component(gallery.$$.fragment);
    },
    l(nodes) {
      claim_component(gallery.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(gallery, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const gallery_changes = {};
      if (dirty[0] & /*label*/
      32)
        gallery_changes.label = /*label*/
        ctx2[5];
      if (dirty[0] & /*show_label*/
      16)
        gallery_changes.show_label = /*show_label*/
        ctx2[4];
      if (dirty[0] & /*columns*/
      16384)
        gallery_changes.columns = /*columns*/
        ctx2[14];
      if (dirty[0] & /*rows*/
      32768)
        gallery_changes.rows = /*rows*/
        ctx2[15];
      if (dirty[0] & /*height*/
      65536)
        gallery_changes.height = /*height*/
        ctx2[16];
      if (dirty[0] & /*preview*/
      131072)
        gallery_changes.preview = /*preview*/
        ctx2[17];
      if (dirty[0] & /*object_fit*/
      524288)
        gallery_changes.object_fit = /*object_fit*/
        ctx2[19];
      if (dirty[0] & /*interactive*/
      2097152)
        gallery_changes.interactive = /*interactive*/
        ctx2[21];
      if (dirty[0] & /*allow_preview*/
      262144)
        gallery_changes.allow_preview = /*allow_preview*/
        ctx2[18];
      if (dirty[0] & /*show_share_button*/
      1048576)
        gallery_changes.show_share_button = /*show_share_button*/
        ctx2[20];
      if (dirty[0] & /*show_download_button*/
      4194304)
        gallery_changes.show_download_button = /*show_download_button*/
        ctx2[22];
      if (dirty[0] & /*gradio*/
      8388608)
        gallery_changes.i18n = /*gradio*/
        ctx2[23].i18n;
      if (dirty[0] & /*gradio*/
      8388608)
        gallery_changes._fetch = /*func_2*/
        ctx2[31];
      if (dirty[0] & /*show_fullscreen_button*/
      16777216)
        gallery_changes.show_fullscreen_button = /*show_fullscreen_button*/
        ctx2[24];
      if (dirty[0] & /*fullscreen*/
      8)
        gallery_changes.fullscreen = /*fullscreen*/
        ctx2[3];
      if (!updating_selected_index && dirty[0] & /*selected_index*/
      2) {
        updating_selected_index = true;
        gallery_changes.selected_index = /*selected_index*/
        ctx2[1];
        add_flush_callback(() => updating_selected_index = false);
      }
      if (!updating_value && dirty[0] & /*value*/
      1) {
        updating_value = true;
        gallery_changes.value = /*value*/
        ctx2[0];
        add_flush_callback(() => updating_value = false);
      }
      gallery.$set(gallery_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(gallery.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(gallery.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(gallery, detaching);
    }
  };
}
function create_if_block(ctx) {
  let basefileupload;
  let current;
  basefileupload = new BaseFileUpload({
    props: {
      value: null,
      root: (
        /*root*/
        ctx[6]
      ),
      label: (
        /*label*/
        ctx[5]
      ),
      max_file_size: (
        /*gradio*/
        ctx[23].max_file_size
      ),
      file_count: "multiple",
      file_types: (
        /*file_types*/
        ctx[10]
      ),
      i18n: (
        /*gradio*/
        ctx[23].i18n
      ),
      upload: (
        /*func*/
        ctx[27]
      ),
      stream_handler: (
        /*func_1*/
        ctx[28]
      ),
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  basefileupload.$on(
    "upload",
    /*upload_handler*/
    ctx[29]
  );
  basefileupload.$on(
    "error",
    /*error_handler*/
    ctx[30]
  );
  return {
    c() {
      create_component(basefileupload.$$.fragment);
    },
    l(nodes) {
      claim_component(basefileupload.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(basefileupload, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const basefileupload_changes = {};
      if (dirty[0] & /*root*/
      64)
        basefileupload_changes.root = /*root*/
        ctx2[6];
      if (dirty[0] & /*label*/
      32)
        basefileupload_changes.label = /*label*/
        ctx2[5];
      if (dirty[0] & /*gradio*/
      8388608)
        basefileupload_changes.max_file_size = /*gradio*/
        ctx2[23].max_file_size;
      if (dirty[0] & /*file_types*/
      1024)
        basefileupload_changes.file_types = /*file_types*/
        ctx2[10];
      if (dirty[0] & /*gradio*/
      8388608)
        basefileupload_changes.i18n = /*gradio*/
        ctx2[23].i18n;
      if (dirty[0] & /*gradio*/
      8388608)
        basefileupload_changes.upload = /*func*/
        ctx2[27];
      if (dirty[0] & /*gradio*/
      8388608)
        basefileupload_changes.stream_handler = /*func_1*/
        ctx2[28];
      if (dirty[0] & /*gradio*/
      8388608 | dirty[1] & /*$$scope*/
      4096) {
        basefileupload_changes.$$scope = { dirty, ctx: ctx2 };
      }
      basefileupload.$set(basefileupload_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(basefileupload.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(basefileupload.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(basefileupload, detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let uploadtext;
  let current;
  uploadtext = new UploadText({
    props: {
      i18n: (
        /*gradio*/
        ctx[23].i18n
      ),
      type: "gallery"
    }
  });
  return {
    c() {
      create_component(uploadtext.$$.fragment);
    },
    l(nodes) {
      claim_component(uploadtext.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(uploadtext, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const uploadtext_changes = {};
      if (dirty[0] & /*gradio*/
      8388608)
        uploadtext_changes.i18n = /*gradio*/
        ctx2[23].i18n;
      uploadtext.$set(uploadtext_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(uploadtext.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(uploadtext.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(uploadtext, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let statustracker;
  let t;
  let current_block_type_index;
  let if_block;
  let if_block_anchor;
  let current;
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[23].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[23].i18n
    ) },
    /*loading_status*/
    ctx[2]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[26]
  );
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*interactive*/
      ctx2[21] && /*no_value*/
      ctx2[25]
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t = space();
      if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t = claim_space(nodes);
      if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      8388612 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        8388608 && {
          autoscroll: (
            /*gradio*/
            ctx2[23].autoscroll
          )
        },
        dirty[0] & /*gradio*/
        8388608 && { i18n: (
          /*gradio*/
          ctx2[23].i18n
        ) },
        dirty[0] & /*loading_status*/
        4 && get_spread_object(
          /*loading_status*/
          ctx2[2]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
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
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        } else {
          if_block.p(ctx2, dirty);
        }
        transition_in(if_block, 1);
        if_block.m(if_block_anchor.parentNode, if_block_anchor);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(if_block_anchor);
      }
      destroy_component(statustracker, detaching);
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let updating_fullscreen;
  let current;
  function block_fullscreen_binding(value) {
    ctx[41](value);
  }
  let block_props = {
    visible: (
      /*visible*/
      ctx[9]
    ),
    variant: "solid",
    padding: false,
    elem_id: (
      /*elem_id*/
      ctx[7]
    ),
    elem_classes: (
      /*elem_classes*/
      ctx[8]
    ),
    container: (
      /*container*/
      ctx[11]
    ),
    scale: (
      /*scale*/
      ctx[12]
    ),
    min_width: (
      /*min_width*/
      ctx[13]
    ),
    allow_overflow: false,
    height: typeof /*height*/
    ctx[16] === "number" ? (
      /*height*/
      ctx[16]
    ) : void 0,
    $$slots: { default: [create_default_slot] },
    $$scope: { ctx }
  };
  if (
    /*fullscreen*/
    ctx[3] !== void 0
  ) {
    block_props.fullscreen = /*fullscreen*/
    ctx[3];
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
      512)
        block_changes.visible = /*visible*/
        ctx2[9];
      if (dirty[0] & /*elem_id*/
      128)
        block_changes.elem_id = /*elem_id*/
        ctx2[7];
      if (dirty[0] & /*elem_classes*/
      256)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[8];
      if (dirty[0] & /*container*/
      2048)
        block_changes.container = /*container*/
        ctx2[11];
      if (dirty[0] & /*scale*/
      4096)
        block_changes.scale = /*scale*/
        ctx2[12];
      if (dirty[0] & /*min_width*/
      8192)
        block_changes.min_width = /*min_width*/
        ctx2[13];
      if (dirty[0] & /*height*/
      65536)
        block_changes.height = typeof /*height*/
        ctx2[16] === "number" ? (
          /*height*/
          ctx2[16]
        ) : void 0;
      if (dirty[0] & /*root, label, gradio, file_types, value, loading_status, interactive, no_value, show_label, columns, rows, height, preview, object_fit, allow_preview, show_share_button, show_download_button, show_fullscreen_button, fullscreen, selected_index*/
      67093631 | dirty[1] & /*$$scope*/
      4096) {
        block_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_fullscreen && dirty[0] & /*fullscreen*/
      8) {
        updating_fullscreen = true;
        block_changes.fullscreen = /*fullscreen*/
        ctx2[3];
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
async function process_upload_files(files) {
  const processed_files = await Promise.all(files.map(async (x) => {
    var _a;
    if (((_a = x.path) == null ? void 0 : _a.toLowerCase().endsWith(".svg")) && x.url) {
      const response = await fetch(x.url);
      const svgContent = await response.text();
      return {
        ...x,
        url: `data:image/svg+xml,${encodeURIComponent(svgContent)}`
      };
    }
    return x;
  }));
  return processed_files.map((x) => {
    var _a;
    return ((_a = x.mime_type) == null ? void 0 : _a.includes("video")) ? { video: x, caption: null } : { image: x, caption: null };
  });
}
function instance($$self, $$props, $$invalidate) {
  let no_value;
  let { loading_status } = $$props;
  let { show_label } = $$props;
  let { label } = $$props;
  let { root } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = null } = $$props;
  let { file_types = ["image", "video"] } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { columns = [2] } = $$props;
  let { rows = void 0 } = $$props;
  let { height = "auto" } = $$props;
  let { preview } = $$props;
  let { allow_preview = true } = $$props;
  let { selected_index = null } = $$props;
  let { object_fit = "cover" } = $$props;
  let { show_share_button = false } = $$props;
  let { interactive } = $$props;
  let { show_download_button = false } = $$props;
  let { gradio } = $$props;
  let { show_fullscreen_button = true } = $$props;
  let { fullscreen = false } = $$props;
  const dispatch = createEventDispatcher();
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const func = (...args) => gradio.client.upload(...args);
  const func_1 = (...args) => gradio.client.stream(...args);
  const upload_handler = async (e) => {
    const files = Array.isArray(e.detail) ? e.detail : [e.detail];
    $$invalidate(0, value = await process_upload_files(files));
    gradio.dispatch("upload", value);
    gradio.dispatch("change", value);
  };
  const error_handler = ({ detail }) => {
    $$invalidate(2, loading_status = loading_status || {});
    $$invalidate(2, loading_status.status = "error", loading_status);
    gradio.dispatch("error", detail);
  };
  const func_2 = (...args) => gradio.client.fetch(...args);
  function gallery_selected_index_binding(value2) {
    selected_index = value2;
    $$invalidate(1, selected_index);
  }
  function gallery_value_binding(value$1) {
    value = value$1;
    $$invalidate(0, value);
  }
  const change_handler = () => gradio.dispatch("change", value);
  const select_handler = (e) => gradio.dispatch("select", e.detail);
  const share_handler = (e) => gradio.dispatch("share", e.detail);
  const error_handler_1 = (e) => gradio.dispatch("error", e.detail);
  const preview_open_handler = () => gradio.dispatch("preview_open");
  const preview_close_handler = () => gradio.dispatch("preview_close");
  const fullscreen_handler = ({ detail }) => {
    $$invalidate(3, fullscreen = detail);
  };
  function block_fullscreen_binding(value2) {
    fullscreen = value2;
    $$invalidate(3, fullscreen);
  }
  $$self.$$set = ($$props2) => {
    if ("loading_status" in $$props2)
      $$invalidate(2, loading_status = $$props2.loading_status);
    if ("show_label" in $$props2)
      $$invalidate(4, show_label = $$props2.show_label);
    if ("label" in $$props2)
      $$invalidate(5, label = $$props2.label);
    if ("root" in $$props2)
      $$invalidate(6, root = $$props2.root);
    if ("elem_id" in $$props2)
      $$invalidate(7, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(8, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(9, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("file_types" in $$props2)
      $$invalidate(10, file_types = $$props2.file_types);
    if ("container" in $$props2)
      $$invalidate(11, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(12, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(13, min_width = $$props2.min_width);
    if ("columns" in $$props2)
      $$invalidate(14, columns = $$props2.columns);
    if ("rows" in $$props2)
      $$invalidate(15, rows = $$props2.rows);
    if ("height" in $$props2)
      $$invalidate(16, height = $$props2.height);
    if ("preview" in $$props2)
      $$invalidate(17, preview = $$props2.preview);
    if ("allow_preview" in $$props2)
      $$invalidate(18, allow_preview = $$props2.allow_preview);
    if ("selected_index" in $$props2)
      $$invalidate(1, selected_index = $$props2.selected_index);
    if ("object_fit" in $$props2)
      $$invalidate(19, object_fit = $$props2.object_fit);
    if ("show_share_button" in $$props2)
      $$invalidate(20, show_share_button = $$props2.show_share_button);
    if ("interactive" in $$props2)
      $$invalidate(21, interactive = $$props2.interactive);
    if ("show_download_button" in $$props2)
      $$invalidate(22, show_download_button = $$props2.show_download_button);
    if ("gradio" in $$props2)
      $$invalidate(23, gradio = $$props2.gradio);
    if ("show_fullscreen_button" in $$props2)
      $$invalidate(24, show_fullscreen_button = $$props2.show_fullscreen_button);
    if ("fullscreen" in $$props2)
      $$invalidate(3, fullscreen = $$props2.fullscreen);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*value*/
    1) {
      $$invalidate(25, no_value = value === null ? true : value.length === 0);
    }
    if ($$self.$$.dirty[0] & /*selected_index*/
    2) {
      dispatch("prop_change", { selected_index });
    }
  };
  return [
    value,
    selected_index,
    loading_status,
    fullscreen,
    show_label,
    label,
    root,
    elem_id,
    elem_classes,
    visible,
    file_types,
    container,
    scale,
    min_width,
    columns,
    rows,
    height,
    preview,
    allow_preview,
    object_fit,
    show_share_button,
    interactive,
    show_download_button,
    gradio,
    show_fullscreen_button,
    no_value,
    clear_status_handler,
    func,
    func_1,
    upload_handler,
    error_handler,
    func_2,
    gallery_selected_index_binding,
    gallery_value_binding,
    change_handler,
    select_handler,
    share_handler,
    error_handler_1,
    preview_open_handler,
    preview_close_handler,
    fullscreen_handler,
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
        loading_status: 2,
        show_label: 4,
        label: 5,
        root: 6,
        elem_id: 7,
        elem_classes: 8,
        visible: 9,
        value: 0,
        file_types: 10,
        container: 11,
        scale: 12,
        min_width: 13,
        columns: 14,
        rows: 15,
        height: 16,
        preview: 17,
        allow_preview: 18,
        selected_index: 1,
        object_fit: 19,
        show_share_button: 20,
        interactive: 21,
        show_download_button: 22,
        gradio: 23,
        show_fullscreen_button: 24,
        fullscreen: 3
      },
      null,
      [-1, -1]
    );
  }
}
export {
  Gallery as BaseGallery,
  Index as default
};
