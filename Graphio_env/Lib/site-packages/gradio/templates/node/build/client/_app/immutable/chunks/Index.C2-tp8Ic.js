import { SvelteComponent, init, safe_not_equal, svg_element, claim_svg_element, children, detach, attr, insert_hydration, append_hydration, noop, ensure_array_like, element, space, text, claim_element, get_svelte_dataset, claim_space, claim_text, set_style, listen, set_data, destroy_each, run_all, createEventDispatcher, toggle_class, set_input_value, to_number, create_component, claim_component, mount_component, transition_in, transition_out, destroy_component, onDestroy, empty, group_outros, check_outros, binding_callbacks, bind, add_flush_callback } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { B as Block, z as BlockTitle } from "./2.CWSJHAMp.js";
import { default as default2 } from "./Example.CYe6rnxa.js";
function create_fragment$2(ctx) {
  let svg;
  let rect;
  let line0;
  let line1;
  let line2;
  return {
    c() {
      svg = svg_element("svg");
      rect = svg_element("rect");
      line0 = svg_element("line");
      line1 = svg_element("line");
      line2 = svg_element("line");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", {
        xmlns: true,
        width: true,
        height: true,
        viewBox: true
      });
      var svg_nodes = children(svg);
      rect = claim_svg_element(svg_nodes, "rect", {
        x: true,
        y: true,
        width: true,
        height: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(rect).forEach(detach);
      line0 = claim_svg_element(svg_nodes, "line", {
        x1: true,
        y1: true,
        x2: true,
        y2: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(line0).forEach(detach);
      line1 = claim_svg_element(svg_nodes, "line", {
        x1: true,
        y1: true,
        x2: true,
        y2: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(line1).forEach(detach);
      line2 = claim_svg_element(svg_nodes, "line", {
        x1: true,
        y1: true,
        x2: true,
        y2: true,
        stroke: true,
        "stroke-width": true,
        "stroke-linecap": true,
        "stroke-linejoin": true,
        fill: true
      });
      children(line2).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(rect, "x", "2");
      attr(rect, "y", "4");
      attr(rect, "width", "20");
      attr(rect, "height", "18");
      attr(rect, "stroke", "currentColor");
      attr(rect, "stroke-width", "2");
      attr(rect, "stroke-linecap", "round");
      attr(rect, "stroke-linejoin", "round");
      attr(rect, "fill", "none");
      attr(line0, "x1", "2");
      attr(line0, "y1", "9");
      attr(line0, "x2", "22");
      attr(line0, "y2", "9");
      attr(line0, "stroke", "currentColor");
      attr(line0, "stroke-width", "2");
      attr(line0, "stroke-linecap", "round");
      attr(line0, "stroke-linejoin", "round");
      attr(line0, "fill", "none");
      attr(line1, "x1", "7");
      attr(line1, "y1", "2");
      attr(line1, "x2", "7");
      attr(line1, "y2", "6");
      attr(line1, "stroke", "currentColor");
      attr(line1, "stroke-width", "2");
      attr(line1, "stroke-linecap", "round");
      attr(line1, "stroke-linejoin", "round");
      attr(line1, "fill", "none");
      attr(line2, "x1", "17");
      attr(line2, "y1", "2");
      attr(line2, "x2", "17");
      attr(line2, "y2", "6");
      attr(line2, "stroke", "currentColor");
      attr(line2, "stroke-width", "2");
      attr(line2, "stroke-linecap", "round");
      attr(line2, "stroke-linejoin", "round");
      attr(line2, "fill", "none");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
      attr(svg, "width", "24px");
      attr(svg, "height", "24px");
      attr(svg, "viewBox", "0 0 24 24");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, rect);
      append_hydration(svg, line0);
      append_hydration(svg, line1);
      append_hydration(svg, line2);
    },
    p: noop,
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
class Calendar extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, null, create_fragment$2, safe_not_equal, {});
  }
}
const format_date = (date, include_time) => {
  if (date.toJSON() === null)
    return "";
  const pad = (num) => num.toString().padStart(2, "0");
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  const date_str = `${year}-${month}-${day}`;
  const time_str = `${hours}:${minutes}:${seconds}`;
  if (include_time) {
    return `${date_str} ${time_str}`;
  }
  return date_str;
};
const date_is_valid_format = (date, include_time) => {
  if (date === null || date === "")
    return true;
  const valid_regex = include_time ? /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/ : /^\d{4}-\d{2}-\d{2}$/;
  const is_valid_date = date.match(valid_regex) !== null;
  const is_valid_now = date.match(/^(?:\s*now\s*(?:-\s*\d+\s*[dmhs])?)?\s*$/) !== null;
  return is_valid_date || is_valid_now;
};
const get_days_in_month = (year, month) => {
  return new Date(year, month + 1, 0).getDate();
};
const get_first_day_of_month = (year, month) => {
  return new Date(year, month, 1).getDay();
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
const generate_calendar_days = (current_year, current_month) => {
  const days_in_month = get_days_in_month(current_year, current_month);
  const first_day = get_first_day_of_month(current_year, current_month);
  const days = [];
  const prev_month = current_month === 0 ? 11 : current_month - 1;
  const prev_year = current_month === 0 ? current_year - 1 : current_year;
  const days_in_prev_month = get_days_in_month(prev_year, prev_month);
  for (let i = first_day - 1; i >= 0; i--) {
    days.push({
      day: days_in_prev_month - i,
      is_current_month: false,
      is_next_month: false
    });
  }
  for (let day = 1; day <= days_in_month; day++) {
    days.push({
      day,
      is_current_month: true,
      is_next_month: false
    });
  }
  const remaining_slots = 42 - days.length;
  for (let day = 1; day <= remaining_slots; day++) {
    days.push({
      day,
      is_current_month: false,
      is_next_month: true
    });
  }
  return days;
};
const calculate_display_hour = (selected_hour, is_pm) => {
  return is_pm ? selected_hour === 0 ? 12 : selected_hour > 12 ? selected_hour - 12 : selected_hour : selected_hour === 0 ? 12 : selected_hour;
};
const convert_display_hour_to_24h = (display_hour, is_pm) => {
  if (is_pm) {
    return display_hour === 12 ? 12 : display_hour + 12;
  }
  return display_hour === 12 ? 0 : display_hour;
};
const month_names = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December"
];
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[27] = list[i].day;
  child_ctx[28] = list[i].is_current_month;
  child_ctx[29] = list[i].is_next_month;
  return child_ctx;
}
function create_each_block(ctx) {
  let button;
  let t0_value = (
    /*day*/
    ctx[27] + ""
  );
  let t0;
  let t1;
  let mounted;
  let dispose;
  function click_handler() {
    return (
      /*click_handler*/
      ctx[19](
        /*is_current_month*/
        ctx[28],
        /*day*/
        ctx[27],
        /*is_next_month*/
        ctx[29]
      )
    );
  }
  return {
    c() {
      button = element("button");
      t0 = text(t0_value);
      t1 = space();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { type: true, class: true });
      var button_nodes = children(button);
      t0 = claim_text(button_nodes, t0_value);
      t1 = claim_space(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "type", "button");
      attr(button, "class", "day svelte-12ypm2m");
      toggle_class(button, "other-month", !/*is_current_month*/
      ctx[28]);
      toggle_class(
        button,
        "selected",
        /*is_current_month*/
        ctx[28] && /*day*/
        ctx[27] === /*selected_date*/
        ctx[3].getDate() && /*current_month*/
        ctx[1] === /*selected_date*/
        ctx[3].getMonth() && /*current_year*/
        ctx[0] === /*selected_date*/
        ctx[3].getFullYear()
      );
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      append_hydration(button, t0);
      append_hydration(button, t1);
      if (!mounted) {
        dispose = listen(button, "click", click_handler);
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (dirty[0] & /*calendar_days*/
      256 && t0_value !== (t0_value = /*day*/
      ctx[27] + ""))
        set_data(t0, t0_value);
      if (dirty[0] & /*calendar_days*/
      256) {
        toggle_class(button, "other-month", !/*is_current_month*/
        ctx[28]);
      }
      if (dirty[0] & /*calendar_days, selected_date, current_month, current_year*/
      267) {
        toggle_class(
          button,
          "selected",
          /*is_current_month*/
          ctx[28] && /*day*/
          ctx[27] === /*selected_date*/
          ctx[3].getDate() && /*current_month*/
          ctx[1] === /*selected_date*/
          ctx[3].getMonth() && /*current_year*/
          ctx[0] === /*selected_date*/
          ctx[3].getFullYear()
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      mounted = false;
      dispose();
    }
  };
}
function create_if_block$1(ctx) {
  let div5;
  let div4;
  let div0;
  let label0;
  let textContent = "Hour";
  let t1;
  let input0;
  let t2;
  let div1;
  let label1;
  let textContent_1 = "Min";
  let t4;
  let input1;
  let t5;
  let div2;
  let label2;
  let textContent_2 = "Sec";
  let t7;
  let input2;
  let t8;
  let div3;
  let span;
  let textContent_3 = "Period";
  let t10;
  let button;
  let t11_value = (
    /*is_pm*/
    ctx[2] ? "PM" : "AM"
  );
  let t11;
  let mounted;
  let dispose;
  return {
    c() {
      div5 = element("div");
      div4 = element("div");
      div0 = element("div");
      label0 = element("label");
      label0.textContent = textContent;
      t1 = space();
      input0 = element("input");
      t2 = space();
      div1 = element("div");
      label1 = element("label");
      label1.textContent = textContent_1;
      t4 = space();
      input1 = element("input");
      t5 = space();
      div2 = element("div");
      label2 = element("label");
      label2.textContent = textContent_2;
      t7 = space();
      input2 = element("input");
      t8 = space();
      div3 = element("div");
      span = element("span");
      span.textContent = textContent_3;
      t10 = space();
      button = element("button");
      t11 = text(t11_value);
      this.h();
    },
    l(nodes) {
      div5 = claim_element(nodes, "DIV", { class: true });
      var div5_nodes = children(div5);
      div4 = claim_element(div5_nodes, "DIV", { class: true });
      var div4_nodes = children(div4);
      div0 = claim_element(div4_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      label0 = claim_element(div0_nodes, "LABEL", {
        for: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(label0) !== "svelte-1n9kdhy")
        label0.textContent = textContent;
      t1 = claim_space(div0_nodes);
      input0 = claim_element(div0_nodes, "INPUT", {
        id: true,
        type: true,
        min: true,
        max: true,
        class: true
      });
      div0_nodes.forEach(detach);
      t2 = claim_space(div4_nodes);
      div1 = claim_element(div4_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      label1 = claim_element(div1_nodes, "LABEL", {
        for: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(label1) !== "svelte-lzzwja")
        label1.textContent = textContent_1;
      t4 = claim_space(div1_nodes);
      input1 = claim_element(div1_nodes, "INPUT", {
        id: true,
        type: true,
        min: true,
        max: true,
        class: true
      });
      div1_nodes.forEach(detach);
      t5 = claim_space(div4_nodes);
      div2 = claim_element(div4_nodes, "DIV", { class: true });
      var div2_nodes = children(div2);
      label2 = claim_element(div2_nodes, "LABEL", {
        for: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(label2) !== "svelte-1x71o3x")
        label2.textContent = textContent_2;
      t7 = claim_space(div2_nodes);
      input2 = claim_element(div2_nodes, "INPUT", {
        id: true,
        type: true,
        min: true,
        max: true,
        class: true
      });
      div2_nodes.forEach(detach);
      t8 = claim_space(div4_nodes);
      div3 = claim_element(div4_nodes, "DIV", { class: true });
      var div3_nodes = children(div3);
      span = claim_element(div3_nodes, "SPAN", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(span) !== "svelte-f0vg1h")
        span.textContent = textContent_3;
      t10 = claim_space(div3_nodes);
      button = claim_element(div3_nodes, "BUTTON", {
        type: true,
        class: true,
        "aria-label": true
      });
      var button_nodes = children(button);
      t11 = claim_text(button_nodes, t11_value);
      button_nodes.forEach(detach);
      div3_nodes.forEach(detach);
      div4_nodes.forEach(detach);
      div5_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(label0, "for", "hour");
      attr(label0, "class", "svelte-12ypm2m");
      attr(input0, "id", "hour");
      attr(input0, "type", "number");
      attr(input0, "min", "1");
      attr(input0, "max", "12");
      attr(input0, "class", "svelte-12ypm2m");
      attr(div0, "class", "time-input-group svelte-12ypm2m");
      attr(label1, "for", "minute");
      attr(label1, "class", "svelte-12ypm2m");
      attr(input1, "id", "minute");
      attr(input1, "type", "number");
      attr(input1, "min", "0");
      attr(input1, "max", "59");
      attr(input1, "class", "svelte-12ypm2m");
      attr(div1, "class", "time-input-group svelte-12ypm2m");
      attr(label2, "for", "second");
      attr(label2, "class", "svelte-12ypm2m");
      attr(input2, "id", "second");
      attr(input2, "type", "number");
      attr(input2, "min", "0");
      attr(input2, "max", "59");
      attr(input2, "class", "svelte-12ypm2m");
      attr(div2, "class", "time-input-group svelte-12ypm2m");
      attr(span, "class", "am-pm-label svelte-12ypm2m");
      attr(button, "type", "button");
      attr(button, "class", "am-pm-toggle svelte-12ypm2m");
      attr(button, "aria-label", "Toggle AM/PM");
      attr(div3, "class", "time-input-group svelte-12ypm2m");
      attr(div4, "class", "time-inputs svelte-12ypm2m");
      attr(div5, "class", "time-picker svelte-12ypm2m");
    },
    m(target, anchor) {
      insert_hydration(target, div5, anchor);
      append_hydration(div5, div4);
      append_hydration(div4, div0);
      append_hydration(div0, label0);
      append_hydration(div0, t1);
      append_hydration(div0, input0);
      set_input_value(
        input0,
        /*display_hour*/
        ctx[9]
      );
      append_hydration(div4, t2);
      append_hydration(div4, div1);
      append_hydration(div1, label1);
      append_hydration(div1, t4);
      append_hydration(div1, input1);
      set_input_value(
        input1,
        /*selected_minute*/
        ctx[4]
      );
      append_hydration(div4, t5);
      append_hydration(div4, div2);
      append_hydration(div2, label2);
      append_hydration(div2, t7);
      append_hydration(div2, input2);
      set_input_value(
        input2,
        /*selected_second*/
        ctx[5]
      );
      append_hydration(div4, t8);
      append_hydration(div4, div3);
      append_hydration(div3, span);
      append_hydration(div3, t10);
      append_hydration(div3, button);
      append_hydration(button, t11);
      if (!mounted) {
        dispose = [
          listen(
            input0,
            "input",
            /*input0_input_handler*/
            ctx[20]
          ),
          listen(
            input0,
            "input",
            /*input_handler*/
            ctx[21]
          ),
          listen(
            input1,
            "input",
            /*input1_input_handler*/
            ctx[22]
          ),
          listen(
            input1,
            "input",
            /*update_time*/
            ctx[12]
          ),
          listen(
            input2,
            "input",
            /*input2_input_handler*/
            ctx[23]
          ),
          listen(
            input2,
            "input",
            /*update_time*/
            ctx[12]
          ),
          listen(
            button,
            "click",
            /*toggle_am_pm*/
            ctx[15]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*display_hour*/
      512 && to_number(input0.value) !== /*display_hour*/
      ctx2[9]) {
        set_input_value(
          input0,
          /*display_hour*/
          ctx2[9]
        );
      }
      if (dirty[0] & /*selected_minute*/
      16 && to_number(input1.value) !== /*selected_minute*/
      ctx2[4]) {
        set_input_value(
          input1,
          /*selected_minute*/
          ctx2[4]
        );
      }
      if (dirty[0] & /*selected_second*/
      32 && to_number(input2.value) !== /*selected_second*/
      ctx2[5]) {
        set_input_value(
          input2,
          /*selected_second*/
          ctx2[5]
        );
      }
      if (dirty[0] & /*is_pm*/
      4 && t11_value !== (t11_value = /*is_pm*/
      ctx2[2] ? "PM" : "AM"))
        set_data(t11, t11_value);
    },
    d(detaching) {
      if (detaching) {
        detach(div5);
      }
      mounted = false;
      run_all(dispose);
    }
  };
}
function create_fragment$1(ctx) {
  let div15;
  let div14;
  let div1;
  let button0;
  let textContent = "‹";
  let t1;
  let div0;
  let t2_value = month_names[
    /*current_month*/
    ctx[1]
  ] + "";
  let t2;
  let t3;
  let t4;
  let t5;
  let button1;
  let textContent_1 = "›";
  let t7;
  let div11;
  let div9;
  let textContent_2 = `<div class="weekday svelte-12ypm2m">Su</div> <div class="weekday svelte-12ypm2m">Mo</div> <div class="weekday svelte-12ypm2m">Tu</div> <div class="weekday svelte-12ypm2m">We</div> <div class="weekday svelte-12ypm2m">Th</div> <div class="weekday svelte-12ypm2m">Fr</div> <div class="weekday svelte-12ypm2m">Sa</div>`;
  let t21;
  let div10;
  let t22;
  let t23;
  let div13;
  let button2;
  let textContent_3 = "Clear";
  let t25;
  let div12;
  let button3;
  let textContent_4 = "Now";
  let t27;
  let button4;
  let textContent_5 = "Done";
  let mounted;
  let dispose;
  let each_value = ensure_array_like(
    /*calendar_days*/
    ctx[8]
  );
  let each_blocks = [];
  for (let i = 0; i < each_value.length; i += 1) {
    each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
  }
  let if_block = (
    /*include_time*/
    ctx[6] && create_if_block$1(ctx)
  );
  return {
    c() {
      div15 = element("div");
      div14 = element("div");
      div1 = element("div");
      button0 = element("button");
      button0.textContent = textContent;
      t1 = space();
      div0 = element("div");
      t2 = text(t2_value);
      t3 = space();
      t4 = text(
        /*current_year*/
        ctx[0]
      );
      t5 = space();
      button1 = element("button");
      button1.textContent = textContent_1;
      t7 = space();
      div11 = element("div");
      div9 = element("div");
      div9.innerHTML = textContent_2;
      t21 = space();
      div10 = element("div");
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      t22 = space();
      if (if_block)
        if_block.c();
      t23 = space();
      div13 = element("div");
      button2 = element("button");
      button2.textContent = textContent_3;
      t25 = space();
      div12 = element("div");
      button3 = element("button");
      button3.textContent = textContent_4;
      t27 = space();
      button4 = element("button");
      button4.textContent = textContent_5;
      this.h();
    },
    l(nodes) {
      div15 = claim_element(nodes, "DIV", { class: true, style: true });
      var div15_nodes = children(div15);
      div14 = claim_element(div15_nodes, "DIV", { class: true });
      var div14_nodes = children(div14);
      div1 = claim_element(div14_nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      button0 = claim_element(div1_nodes, "BUTTON", {
        type: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button0) !== "svelte-1fir2s8")
        button0.textContent = textContent;
      t1 = claim_space(div1_nodes);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      t2 = claim_text(div0_nodes, t2_value);
      t3 = claim_space(div0_nodes);
      t4 = claim_text(
        div0_nodes,
        /*current_year*/
        ctx[0]
      );
      div0_nodes.forEach(detach);
      t5 = claim_space(div1_nodes);
      button1 = claim_element(div1_nodes, "BUTTON", {
        type: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button1) !== "svelte-l526mg")
        button1.textContent = textContent_1;
      div1_nodes.forEach(detach);
      t7 = claim_space(div14_nodes);
      div11 = claim_element(div14_nodes, "DIV", { class: true });
      var div11_nodes = children(div11);
      div9 = claim_element(div11_nodes, "DIV", { class: true, ["data-svelte-h"]: true });
      if (get_svelte_dataset(div9) !== "svelte-10nwkhe")
        div9.innerHTML = textContent_2;
      t21 = claim_space(div11_nodes);
      div10 = claim_element(div11_nodes, "DIV", { class: true });
      var div10_nodes = children(div10);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(div10_nodes);
      }
      div10_nodes.forEach(detach);
      div11_nodes.forEach(detach);
      t22 = claim_space(div14_nodes);
      if (if_block)
        if_block.l(div14_nodes);
      t23 = claim_space(div14_nodes);
      div13 = claim_element(div14_nodes, "DIV", { class: true });
      var div13_nodes = children(div13);
      button2 = claim_element(div13_nodes, "BUTTON", {
        type: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button2) !== "svelte-1rfyr2i")
        button2.textContent = textContent_3;
      t25 = claim_space(div13_nodes);
      div12 = claim_element(div13_nodes, "DIV", { class: true });
      var div12_nodes = children(div12);
      button3 = claim_element(div12_nodes, "BUTTON", {
        type: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button3) !== "svelte-e0esoh")
        button3.textContent = textContent_4;
      t27 = claim_space(div12_nodes);
      button4 = claim_element(div12_nodes, "BUTTON", {
        type: true,
        class: true,
        ["data-svelte-h"]: true
      });
      if (get_svelte_dataset(button4) !== "svelte-16hiibm")
        button4.textContent = textContent_5;
      div12_nodes.forEach(detach);
      div13_nodes.forEach(detach);
      div14_nodes.forEach(detach);
      div15_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button0, "type", "button");
      attr(button0, "class", "nav-button svelte-12ypm2m");
      attr(div0, "class", "month-year svelte-12ypm2m");
      attr(button1, "type", "button");
      attr(button1, "class", "nav-button svelte-12ypm2m");
      attr(div1, "class", "picker-header svelte-12ypm2m");
      attr(div9, "class", "weekdays svelte-12ypm2m");
      attr(div10, "class", "days svelte-12ypm2m");
      attr(div11, "class", "calendar-grid svelte-12ypm2m");
      attr(button2, "type", "button");
      attr(button2, "class", "action-button svelte-12ypm2m");
      attr(button3, "type", "button");
      attr(button3, "class", "action-button svelte-12ypm2m");
      attr(button4, "type", "button");
      attr(button4, "class", "action-button svelte-12ypm2m");
      attr(div12, "class", "picker-actions-right svelte-12ypm2m");
      attr(div13, "class", "picker-actions svelte-12ypm2m");
      attr(div14, "class", "picker svelte-12ypm2m");
      attr(div15, "class", "picker-container svelte-12ypm2m");
      set_style(
        div15,
        "top",
        /*position*/
        ctx[7].top + "px"
      );
      set_style(
        div15,
        "left",
        /*position*/
        ctx[7].left + "px"
      );
    },
    m(target, anchor) {
      insert_hydration(target, div15, anchor);
      append_hydration(div15, div14);
      append_hydration(div14, div1);
      append_hydration(div1, button0);
      append_hydration(div1, t1);
      append_hydration(div1, div0);
      append_hydration(div0, t2);
      append_hydration(div0, t3);
      append_hydration(div0, t4);
      append_hydration(div1, t5);
      append_hydration(div1, button1);
      append_hydration(div14, t7);
      append_hydration(div14, div11);
      append_hydration(div11, div9);
      append_hydration(div11, t21);
      append_hydration(div11, div10);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(div10, null);
        }
      }
      append_hydration(div14, t22);
      if (if_block)
        if_block.m(div14, null);
      append_hydration(div14, t23);
      append_hydration(div14, div13);
      append_hydration(div13, button2);
      append_hydration(div13, t25);
      append_hydration(div13, div12);
      append_hydration(div12, button3);
      append_hydration(div12, t27);
      append_hydration(div12, button4);
      if (!mounted) {
        dispose = [
          listen(
            button0,
            "click",
            /*previous_month*/
            ctx[13]
          ),
          listen(
            button1,
            "click",
            /*next_month*/
            ctx[14]
          ),
          listen(
            button2,
            "click",
            /*click_handler_1*/
            ctx[24]
          ),
          listen(
            button3,
            "click",
            /*handle_now*/
            ctx[17]
          ),
          listen(
            button4,
            "click",
            /*click_handler_2*/
            ctx[25]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*current_month*/
      2 && t2_value !== (t2_value = month_names[
        /*current_month*/
        ctx2[1]
      ] + ""))
        set_data(t2, t2_value);
      if (dirty[0] & /*current_year*/
      1)
        set_data(
          t4,
          /*current_year*/
          ctx2[0]
        );
      if (dirty[0] & /*calendar_days, selected_date, current_month, current_year, select_date, next_month, previous_month*/
      26891) {
        each_value = ensure_array_like(
          /*calendar_days*/
          ctx2[8]
        );
        let i;
        for (i = 0; i < each_value.length; i += 1) {
          const child_ctx = get_each_context(ctx2, each_value, i);
          if (each_blocks[i]) {
            each_blocks[i].p(child_ctx, dirty);
          } else {
            each_blocks[i] = create_each_block(child_ctx);
            each_blocks[i].c();
            each_blocks[i].m(div10, null);
          }
        }
        for (; i < each_blocks.length; i += 1) {
          each_blocks[i].d(1);
        }
        each_blocks.length = each_value.length;
      }
      if (
        /*include_time*/
        ctx2[6]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
        } else {
          if_block = create_if_block$1(ctx2);
          if_block.c();
          if_block.m(div14, t23);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (dirty[0] & /*position*/
      128) {
        set_style(
          div15,
          "top",
          /*position*/
          ctx2[7].top + "px"
        );
      }
      if (dirty[0] & /*position*/
      128) {
        set_style(
          div15,
          "left",
          /*position*/
          ctx2[7].left + "px"
        );
      }
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(div15);
      }
      destroy_each(each_blocks, detaching);
      if (if_block)
        if_block.d();
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance$1($$self, $$props, $$invalidate) {
  let display_hour;
  let calendar_days;
  let { selected_date } = $$props;
  let { current_year } = $$props;
  let { current_month } = $$props;
  let { selected_hour } = $$props;
  let { selected_minute } = $$props;
  let { selected_second } = $$props;
  let { is_pm } = $$props;
  let { include_time } = $$props;
  let { position } = $$props;
  const dispatch = createEventDispatcher();
  const select_date = (day) => {
    $$invalidate(3, selected_date = new Date(current_year, current_month, day, selected_hour, selected_minute, selected_second));
    update_value();
  };
  const update_value = () => {
    const formatted = format_date(selected_date, include_time);
    dispatch("update", { date: selected_date, formatted });
  };
  const update_time = () => {
    $$invalidate(3, selected_date = new Date(current_year, current_month, selected_date.getDate(), selected_hour, selected_minute, selected_second));
    update_value();
  };
  const previous_month = () => {
    if (current_month === 0) {
      $$invalidate(1, current_month = 11);
      $$invalidate(0, current_year--, current_year);
    } else {
      $$invalidate(1, current_month--, current_month);
    }
  };
  const next_month = () => {
    if (current_month === 11) {
      $$invalidate(1, current_month = 0);
      $$invalidate(0, current_year++, current_year);
    } else {
      $$invalidate(1, current_month++, current_month);
    }
  };
  const toggle_am_pm = () => {
    $$invalidate(2, is_pm = !is_pm);
    if (is_pm && selected_hour < 12) {
      $$invalidate(18, selected_hour += 12);
    } else if (!is_pm && selected_hour >= 12) {
      $$invalidate(18, selected_hour -= 12);
    }
    update_time();
  };
  const update_display_hour = (new_hour) => {
    $$invalidate(18, selected_hour = convert_display_hour_to_24h(new_hour, is_pm));
    update_time();
  };
  const handle_now = () => {
    const now = /* @__PURE__ */ new Date();
    $$invalidate(3, selected_date = now);
    $$invalidate(0, current_year = now.getFullYear());
    $$invalidate(1, current_month = now.getMonth());
    $$invalidate(18, selected_hour = now.getHours());
    $$invalidate(4, selected_minute = now.getMinutes());
    $$invalidate(5, selected_second = now.getSeconds());
    $$invalidate(2, is_pm = selected_hour >= 12);
    update_value();
  };
  const click_handler = (is_current_month, day, is_next_month) => {
    if (is_current_month) {
      select_date(day);
    } else if (is_next_month) {
      next_month();
      select_date(day);
    } else {
      previous_month();
      select_date(day);
    }
  };
  function input0_input_handler() {
    display_hour = to_number(this.value);
    $$invalidate(9, display_hour), $$invalidate(18, selected_hour), $$invalidate(2, is_pm);
  }
  const input_handler = () => update_display_hour(display_hour);
  function input1_input_handler() {
    selected_minute = to_number(this.value);
    $$invalidate(4, selected_minute);
  }
  function input2_input_handler() {
    selected_second = to_number(this.value);
    $$invalidate(5, selected_second);
  }
  const click_handler_1 = () => dispatch("clear");
  const click_handler_2 = () => dispatch("close");
  $$self.$$set = ($$props2) => {
    if ("selected_date" in $$props2)
      $$invalidate(3, selected_date = $$props2.selected_date);
    if ("current_year" in $$props2)
      $$invalidate(0, current_year = $$props2.current_year);
    if ("current_month" in $$props2)
      $$invalidate(1, current_month = $$props2.current_month);
    if ("selected_hour" in $$props2)
      $$invalidate(18, selected_hour = $$props2.selected_hour);
    if ("selected_minute" in $$props2)
      $$invalidate(4, selected_minute = $$props2.selected_minute);
    if ("selected_second" in $$props2)
      $$invalidate(5, selected_second = $$props2.selected_second);
    if ("is_pm" in $$props2)
      $$invalidate(2, is_pm = $$props2.is_pm);
    if ("include_time" in $$props2)
      $$invalidate(6, include_time = $$props2.include_time);
    if ("position" in $$props2)
      $$invalidate(7, position = $$props2.position);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*selected_hour, is_pm*/
    262148) {
      $$invalidate(9, display_hour = calculate_display_hour(selected_hour, is_pm));
    }
    if ($$self.$$.dirty[0] & /*current_year, current_month*/
    3) {
      $$invalidate(8, calendar_days = generate_calendar_days(current_year, current_month));
    }
  };
  return [
    current_year,
    current_month,
    is_pm,
    selected_date,
    selected_minute,
    selected_second,
    include_time,
    position,
    calendar_days,
    display_hour,
    dispatch,
    select_date,
    update_time,
    previous_month,
    next_month,
    toggle_am_pm,
    update_display_hour,
    handle_now,
    selected_hour,
    click_handler,
    input0_input_handler,
    input_handler,
    input1_input_handler,
    input2_input_handler,
    click_handler_1,
    click_handler_2
  ];
}
class DateTimePicker extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance$1,
      create_fragment$1,
      safe_not_equal,
      {
        selected_date: 3,
        current_year: 0,
        current_month: 1,
        selected_hour: 18,
        selected_minute: 4,
        selected_second: 5,
        is_pm: 2,
        include_time: 6,
        position: 7
      },
      null,
      [-1, -1]
    );
  }
}
function create_default_slot_1(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*label*/
        ctx[1]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*label*/
        ctx[1]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      2)
        set_data(
          t,
          /*label*/
          ctx2[1]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_1(ctx) {
  let button;
  let calendar;
  let current;
  let mounted;
  let dispose;
  calendar = new Calendar({});
  return {
    c() {
      button = element("button");
      create_component(calendar.$$.fragment);
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      claim_component(calendar.$$.fragment, button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "calendar svelte-ywg1ch");
      button.disabled = /*disabled*/
      ctx[24];
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      mount_component(calendar, button, null);
      ctx[36](button);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*toggle_picker*/
          ctx[27]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (!current || dirty[0] & /*disabled*/
      16777216) {
        button.disabled = /*disabled*/
        ctx2[24];
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(calendar.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(calendar.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      destroy_component(calendar);
      ctx[36](null);
      mounted = false;
      dispose();
    }
  };
}
function create_if_block(ctx) {
  let div;
  let datetimepicker;
  let updating_selected_date;
  let updating_current_year;
  let updating_current_month;
  let updating_selected_hour;
  let updating_selected_minute;
  let updating_selected_second;
  let updating_is_pm;
  let current;
  function datetimepicker_selected_date_binding(value) {
    ctx[37](value);
  }
  function datetimepicker_current_year_binding(value) {
    ctx[38](value);
  }
  function datetimepicker_current_month_binding(value) {
    ctx[39](value);
  }
  function datetimepicker_selected_hour_binding(value) {
    ctx[40](value);
  }
  function datetimepicker_selected_minute_binding(value) {
    ctx[41](value);
  }
  function datetimepicker_selected_second_binding(value) {
    ctx[42](value);
  }
  function datetimepicker_is_pm_binding(value) {
    ctx[43](value);
  }
  let datetimepicker_props = {
    include_time: (
      /*include_time*/
      ctx[10]
    ),
    position: (
      /*picker_position*/
      ctx[16]
    )
  };
  if (
    /*selected_date*/
    ctx[19] !== void 0
  ) {
    datetimepicker_props.selected_date = /*selected_date*/
    ctx[19];
  }
  if (
    /*current_year*/
    ctx[17] !== void 0
  ) {
    datetimepicker_props.current_year = /*current_year*/
    ctx[17];
  }
  if (
    /*current_month*/
    ctx[18] !== void 0
  ) {
    datetimepicker_props.current_month = /*current_month*/
    ctx[18];
  }
  if (
    /*selected_hour*/
    ctx[20] !== void 0
  ) {
    datetimepicker_props.selected_hour = /*selected_hour*/
    ctx[20];
  }
  if (
    /*selected_minute*/
    ctx[21] !== void 0
  ) {
    datetimepicker_props.selected_minute = /*selected_minute*/
    ctx[21];
  }
  if (
    /*selected_second*/
    ctx[22] !== void 0
  ) {
    datetimepicker_props.selected_second = /*selected_second*/
    ctx[22];
  }
  if (
    /*is_pm*/
    ctx[23] !== void 0
  ) {
    datetimepicker_props.is_pm = /*is_pm*/
    ctx[23];
  }
  datetimepicker = new DateTimePicker({ props: datetimepicker_props });
  binding_callbacks.push(() => bind(datetimepicker, "selected_date", datetimepicker_selected_date_binding));
  binding_callbacks.push(() => bind(datetimepicker, "current_year", datetimepicker_current_year_binding));
  binding_callbacks.push(() => bind(datetimepicker, "current_month", datetimepicker_current_month_binding));
  binding_callbacks.push(() => bind(datetimepicker, "selected_hour", datetimepicker_selected_hour_binding));
  binding_callbacks.push(() => bind(datetimepicker, "selected_minute", datetimepicker_selected_minute_binding));
  binding_callbacks.push(() => bind(datetimepicker, "selected_second", datetimepicker_selected_second_binding));
  binding_callbacks.push(() => bind(datetimepicker, "is_pm", datetimepicker_is_pm_binding));
  datetimepicker.$on(
    "update",
    /*handle_picker_update*/
    ctx[29]
  );
  datetimepicker.$on(
    "clear",
    /*handle_picker_clear*/
    ctx[30]
  );
  datetimepicker.$on(
    "close",
    /*close_picker*/
    ctx[28]
  );
  return {
    c() {
      div = element("div");
      create_component(datetimepicker.$$.fragment);
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", { class: true });
      var div_nodes = children(div);
      claim_component(datetimepicker.$$.fragment, div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "svelte-ywg1ch");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      mount_component(datetimepicker, div, null);
      ctx[44](div);
      current = true;
    },
    p(ctx2, dirty) {
      const datetimepicker_changes = {};
      if (dirty[0] & /*include_time*/
      1024)
        datetimepicker_changes.include_time = /*include_time*/
        ctx2[10];
      if (dirty[0] & /*picker_position*/
      65536)
        datetimepicker_changes.position = /*picker_position*/
        ctx2[16];
      if (!updating_selected_date && dirty[0] & /*selected_date*/
      524288) {
        updating_selected_date = true;
        datetimepicker_changes.selected_date = /*selected_date*/
        ctx2[19];
        add_flush_callback(() => updating_selected_date = false);
      }
      if (!updating_current_year && dirty[0] & /*current_year*/
      131072) {
        updating_current_year = true;
        datetimepicker_changes.current_year = /*current_year*/
        ctx2[17];
        add_flush_callback(() => updating_current_year = false);
      }
      if (!updating_current_month && dirty[0] & /*current_month*/
      262144) {
        updating_current_month = true;
        datetimepicker_changes.current_month = /*current_month*/
        ctx2[18];
        add_flush_callback(() => updating_current_month = false);
      }
      if (!updating_selected_hour && dirty[0] & /*selected_hour*/
      1048576) {
        updating_selected_hour = true;
        datetimepicker_changes.selected_hour = /*selected_hour*/
        ctx2[20];
        add_flush_callback(() => updating_selected_hour = false);
      }
      if (!updating_selected_minute && dirty[0] & /*selected_minute*/
      2097152) {
        updating_selected_minute = true;
        datetimepicker_changes.selected_minute = /*selected_minute*/
        ctx2[21];
        add_flush_callback(() => updating_selected_minute = false);
      }
      if (!updating_selected_second && dirty[0] & /*selected_second*/
      4194304) {
        updating_selected_second = true;
        datetimepicker_changes.selected_second = /*selected_second*/
        ctx2[22];
        add_flush_callback(() => updating_selected_second = false);
      }
      if (!updating_is_pm && dirty[0] & /*is_pm*/
      8388608) {
        updating_is_pm = true;
        datetimepicker_changes.is_pm = /*is_pm*/
        ctx2[23];
        add_flush_callback(() => updating_is_pm = false);
      }
      datetimepicker.$set(datetimepicker_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(datetimepicker.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(datetimepicker.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_component(datetimepicker);
      ctx[44](null);
    }
  };
}
function create_default_slot(ctx) {
  let div0;
  let blocktitle;
  let t0;
  let div1;
  let input;
  let input_placeholder_value;
  let t1;
  let t2;
  let if_block1_anchor;
  let current;
  let mounted;
  let dispose;
  blocktitle = new BlockTitle({
    props: {
      show_label: (
        /*show_label*/
        ctx[2]
      ),
      info: (
        /*info*/
        ctx[3]
      ),
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  let if_block0 = (
    /*interactive*/
    ctx[4] && create_if_block_1(ctx)
  );
  let if_block1 = (
    /*show_picker*/
    ctx[12] && create_if_block(ctx)
  );
  return {
    c() {
      div0 = element("div");
      create_component(blocktitle.$$.fragment);
      t0 = space();
      div1 = element("div");
      input = element("input");
      t1 = space();
      if (if_block0)
        if_block0.c();
      t2 = space();
      if (if_block1)
        if_block1.c();
      if_block1_anchor = empty();
      this.h();
    },
    l(nodes) {
      div0 = claim_element(nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      claim_component(blocktitle.$$.fragment, div0_nodes);
      div0_nodes.forEach(detach);
      t0 = claim_space(nodes);
      div1 = claim_element(nodes, "DIV", { class: true });
      var div1_nodes = children(div1);
      input = claim_element(div1_nodes, "INPUT", { class: true, placeholder: true });
      t1 = claim_space(div1_nodes);
      if (if_block0)
        if_block0.l(div1_nodes);
      div1_nodes.forEach(detach);
      t2 = claim_space(nodes);
      if (if_block1)
        if_block1.l(nodes);
      if_block1_anchor = empty();
      this.h();
    },
    h() {
      attr(div0, "class", "label-content svelte-ywg1ch");
      attr(input, "class", "time svelte-ywg1ch");
      input.disabled = /*disabled*/
      ctx[24];
      attr(input, "placeholder", input_placeholder_value = /*include_time*/
      ctx[10] ? "YYYY-MM-DD HH:MM:SS" : "YYYY-MM-DD");
      toggle_class(input, "invalid", !/*valid*/
      ctx[25]);
      attr(div1, "class", "timebox svelte-ywg1ch");
    },
    m(target, anchor) {
      insert_hydration(target, div0, anchor);
      mount_component(blocktitle, div0, null);
      insert_hydration(target, t0, anchor);
      insert_hydration(target, div1, anchor);
      append_hydration(div1, input);
      ctx[33](input);
      set_input_value(
        input,
        /*entered_value*/
        ctx[11]
      );
      append_hydration(div1, t1);
      if (if_block0)
        if_block0.m(div1, null);
      insert_hydration(target, t2, anchor);
      if (if_block1)
        if_block1.m(target, anchor);
      insert_hydration(target, if_block1_anchor, anchor);
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            input,
            "input",
            /*input_input_handler*/
            ctx[34]
          ),
          listen(
            input,
            "keydown",
            /*keydown_handler*/
            ctx[35]
          ),
          listen(
            input,
            "blur",
            /*submit_values*/
            ctx[26]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      const blocktitle_changes = {};
      if (dirty[0] & /*show_label*/
      4)
        blocktitle_changes.show_label = /*show_label*/
        ctx2[2];
      if (dirty[0] & /*info*/
      8)
        blocktitle_changes.info = /*info*/
        ctx2[3];
      if (dirty[0] & /*label*/
      2 | dirty[1] & /*$$scope*/
      262144) {
        blocktitle_changes.$$scope = { dirty, ctx: ctx2 };
      }
      blocktitle.$set(blocktitle_changes);
      if (!current || dirty[0] & /*disabled*/
      16777216) {
        input.disabled = /*disabled*/
        ctx2[24];
      }
      if (!current || dirty[0] & /*include_time*/
      1024 && input_placeholder_value !== (input_placeholder_value = /*include_time*/
      ctx2[10] ? "YYYY-MM-DD HH:MM:SS" : "YYYY-MM-DD")) {
        attr(input, "placeholder", input_placeholder_value);
      }
      if (dirty[0] & /*entered_value*/
      2048 && input.value !== /*entered_value*/
      ctx2[11]) {
        set_input_value(
          input,
          /*entered_value*/
          ctx2[11]
        );
      }
      if (!current || dirty[0] & /*valid*/
      33554432) {
        toggle_class(input, "invalid", !/*valid*/
        ctx2[25]);
      }
      if (
        /*interactive*/
        ctx2[4]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty[0] & /*interactive*/
          16) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_1(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(div1, null);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*show_picker*/
        ctx2[12]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty[0] & /*show_picker*/
          4096) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(if_block1_anchor.parentNode, if_block1_anchor);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(blocktitle.$$.fragment, local);
      transition_in(if_block0);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(blocktitle.$$.fragment, local);
      transition_out(if_block0);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div0);
        detach(t0);
        detach(div1);
        detach(t2);
        detach(if_block1_anchor);
      }
      destroy_component(blocktitle);
      ctx[33](null);
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d(detaching);
      mounted = false;
      run_all(dispose);
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
        ctx[8]
      ),
      min_width: (
        /*min_width*/
        ctx[9]
      ),
      allow_overflow: false,
      padding: true,
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
      256)
        block_changes.scale = /*scale*/
        ctx2[8];
      if (dirty[0] & /*min_width*/
      512)
        block_changes.min_width = /*min_width*/
        ctx2[9];
      if (dirty[0] & /*picker_ref, include_time, picker_position, selected_date, current_year, current_month, selected_hour, selected_minute, selected_second, is_pm, show_picker, disabled, calendar_button_ref, interactive, input_ref, entered_value, valid, gradio, show_label, info, label*/
      67107871 | dirty[1] & /*$$scope*/
      262144) {
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
  let show_picker = false;
  let picker_ref;
  let input_ref;
  let calendar_button_ref;
  let picker_position = { top: 0, left: 0 };
  let entered_value = value;
  const submit_values = () => {
    if (entered_value === value)
      return;
    if (!date_is_valid_format(entered_value, include_time))
      return;
    $$invalidate(32, old_value = $$invalidate(31, value = entered_value));
    gradio.dispatch("change");
  };
  let current_year = /* @__PURE__ */ (/* @__PURE__ */ new Date()).getFullYear();
  let current_month = /* @__PURE__ */ (/* @__PURE__ */ new Date()).getMonth();
  let selected_date = /* @__PURE__ */ new Date();
  let selected_hour = /* @__PURE__ */ (/* @__PURE__ */ new Date()).getHours();
  let selected_minute = /* @__PURE__ */ (/* @__PURE__ */ new Date()).getMinutes();
  let selected_second = /* @__PURE__ */ (/* @__PURE__ */ new Date()).getSeconds();
  let is_pm = selected_hour >= 12;
  const update_picker_from_value = () => {
    const parsed = parse_date_value(entered_value, include_time);
    $$invalidate(19, selected_date = parsed.selected_date);
    $$invalidate(17, current_year = parsed.current_year);
    $$invalidate(18, current_month = parsed.current_month);
    $$invalidate(20, selected_hour = parsed.selected_hour);
    $$invalidate(21, selected_minute = parsed.selected_minute);
    $$invalidate(22, selected_second = parsed.selected_second);
    $$invalidate(23, is_pm = parsed.is_pm);
  };
  const calculate_picker_position = () => {
    if (calendar_button_ref) {
      const rect = calendar_button_ref.getBoundingClientRect();
      $$invalidate(16, picker_position = {
        top: rect.bottom + 4,
        left: rect.right - 280
      });
    }
  };
  const toggle_picker = (event) => {
    if (!disabled) {
      event.stopPropagation();
      $$invalidate(12, show_picker = !show_picker);
      if (show_picker) {
        update_picker_from_value();
        calculate_picker_position();
        setTimeout(
          () => {
            if (typeof window !== "undefined") {
              window.addEventListener("click", handle_click_outside);
              window.addEventListener("scroll", handle_scroll, true);
            }
          },
          10
        );
      } else if (typeof window !== "undefined") {
        window.removeEventListener("click", handle_click_outside);
        window.removeEventListener("scroll", handle_scroll, true);
      }
    }
  };
  const close_picker = () => {
    $$invalidate(12, show_picker = false);
    if (typeof window !== "undefined") {
      window.removeEventListener("click", handle_click_outside);
      window.removeEventListener("scroll", handle_scroll, true);
    }
  };
  const handle_click_outside = (event) => {
    if (show_picker && picker_ref && !picker_ref.contains(event.target) && calendar_button_ref && !calendar_button_ref.contains(event.target)) {
      close_picker();
    }
  };
  const handle_scroll = () => {
    if (show_picker) {
      calculate_picker_position();
    }
  };
  const handle_picker_update = (event) => {
    $$invalidate(11, entered_value = event.detail.formatted);
    submit_values();
  };
  const handle_picker_clear = () => {
    $$invalidate(11, entered_value = "");
    $$invalidate(31, value = "");
    close_picker();
    gradio.dispatch("change");
  };
  onDestroy(() => {
    if (typeof window !== "undefined") {
      window.removeEventListener("click", handle_click_outside);
      window.removeEventListener("scroll", handle_scroll, true);
    }
  });
  update_picker_from_value();
  function input_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      input_ref = $$value;
      $$invalidate(14, input_ref);
    });
  }
  function input_input_handler() {
    entered_value = this.value;
    $$invalidate(11, entered_value), $$invalidate(31, value), $$invalidate(32, old_value), $$invalidate(0, gradio);
  }
  const keydown_handler = (evt) => {
    if (evt.key === "Enter") {
      submit_values();
      gradio.dispatch("submit");
    }
  };
  function button_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      calendar_button_ref = $$value;
      $$invalidate(15, calendar_button_ref);
    });
  }
  function datetimepicker_selected_date_binding(value2) {
    selected_date = value2;
    $$invalidate(19, selected_date);
  }
  function datetimepicker_current_year_binding(value2) {
    current_year = value2;
    $$invalidate(17, current_year);
  }
  function datetimepicker_current_month_binding(value2) {
    current_month = value2;
    $$invalidate(18, current_month);
  }
  function datetimepicker_selected_hour_binding(value2) {
    selected_hour = value2;
    $$invalidate(20, selected_hour);
  }
  function datetimepicker_selected_minute_binding(value2) {
    selected_minute = value2;
    $$invalidate(21, selected_minute);
  }
  function datetimepicker_selected_second_binding(value2) {
    selected_second = value2;
    $$invalidate(22, selected_second);
  }
  function datetimepicker_is_pm_binding(value2) {
    is_pm = value2;
    $$invalidate(23, is_pm);
  }
  function div_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      picker_ref = $$value;
      $$invalidate(13, picker_ref);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("gradio" in $$props2)
      $$invalidate(0, gradio = $$props2.gradio);
    if ("label" in $$props2)
      $$invalidate(1, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(2, show_label = $$props2.show_label);
    if ("info" in $$props2)
      $$invalidate(3, info = $$props2.info);
    if ("interactive" in $$props2)
      $$invalidate(4, interactive = $$props2.interactive);
    if ("elem_id" in $$props2)
      $$invalidate(5, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(6, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(7, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(31, value = $$props2.value);
    if ("scale" in $$props2)
      $$invalidate(8, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(9, min_width = $$props2.min_width);
    if ("include_time" in $$props2)
      $$invalidate(10, include_time = $$props2.include_time);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*interactive*/
    16) {
      $$invalidate(24, disabled = !interactive);
    }
    if ($$self.$$.dirty[0] & /*gradio*/
    1 | $$self.$$.dirty[1] & /*value, old_value*/
    3) {
      if (value !== old_value) {
        $$invalidate(32, old_value = value);
        $$invalidate(11, entered_value = value);
        update_picker_from_value();
        gradio.dispatch("change");
      }
    }
    if ($$self.$$.dirty[0] & /*entered_value, include_time*/
    3072) {
      $$invalidate(25, valid = date_is_valid_format(entered_value, include_time));
    }
  };
  return [
    gradio,
    label,
    show_label,
    info,
    interactive,
    elem_id,
    elem_classes,
    visible,
    scale,
    min_width,
    include_time,
    entered_value,
    show_picker,
    picker_ref,
    input_ref,
    calendar_button_ref,
    picker_position,
    current_year,
    current_month,
    selected_date,
    selected_hour,
    selected_minute,
    selected_second,
    is_pm,
    disabled,
    valid,
    submit_values,
    toggle_picker,
    close_picker,
    handle_picker_update,
    handle_picker_clear,
    value,
    old_value,
    input_binding,
    input_input_handler,
    keydown_handler,
    button_binding,
    datetimepicker_selected_date_binding,
    datetimepicker_current_year_binding,
    datetimepicker_current_month_binding,
    datetimepicker_selected_hour_binding,
    datetimepicker_selected_minute_binding,
    datetimepicker_selected_second_binding,
    datetimepicker_is_pm_binding,
    div_binding
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
        gradio: 0,
        label: 1,
        show_label: 2,
        info: 3,
        interactive: 4,
        elem_id: 5,
        elem_classes: 6,
        visible: 7,
        value: 31,
        scale: 8,
        min_width: 9,
        include_time: 10
      },
      null,
      [-1, -1]
    );
  }
}
export {
  default2 as BaseExample,
  Index as default
};
