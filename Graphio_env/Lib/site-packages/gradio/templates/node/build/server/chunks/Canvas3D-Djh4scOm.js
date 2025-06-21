import { c as create_ssr_component, b as add_attribute } from './ssr-Cwm06D-i.js';
import { r as resolve_wasm_src } from './2-BgvxP-qP.js';
import './index-CoAj_-n5.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BlohB9Ds.js';

const Canvas3D = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let url;
  let { value } = $$props;
  let { display_mode } = $$props;
  let { clear_color } = $$props;
  let { camera_position } = $$props;
  let { zoom_speed } = $$props;
  let { pan_speed } = $$props;
  let { resolved_url = void 0 } = $$props;
  let latest_url;
  let canvas;
  let viewerDetails;
  function update_camera(camera_position2, zoom_speed2, pan_speed2) {
    const camera = viewerDetails.camera;
    if (camera_position2[0] !== null) {
      camera.alpha = camera_position2[0] * Math.PI / 180;
    }
    if (camera_position2[1] !== null) {
      camera.beta = camera_position2[1] * Math.PI / 180;
    }
    if (camera_position2[2] !== null) {
      camera.radius = camera_position2[2];
    }
    camera.lowerRadiusLimit = 0.1;
    const updateCameraSensibility = () => {
      camera.wheelPrecision = 250 / (camera.radius * zoom_speed2);
      camera.panningSensibility = 1e4 * pan_speed2 / camera.radius;
    };
    updateCameraSensibility();
    camera.onAfterCheckInputsObservable.add(updateCameraSensibility);
  }
  function reset_camera_position() {
  }
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.display_mode === void 0 && $$bindings.display_mode && display_mode !== void 0)
    $$bindings.display_mode(display_mode);
  if ($$props.clear_color === void 0 && $$bindings.clear_color && clear_color !== void 0)
    $$bindings.clear_color(clear_color);
  if ($$props.camera_position === void 0 && $$bindings.camera_position && camera_position !== void 0)
    $$bindings.camera_position(camera_position);
  if ($$props.zoom_speed === void 0 && $$bindings.zoom_speed && zoom_speed !== void 0)
    $$bindings.zoom_speed(zoom_speed);
  if ($$props.pan_speed === void 0 && $$bindings.pan_speed && pan_speed !== void 0)
    $$bindings.pan_speed(pan_speed);
  if ($$props.resolved_url === void 0 && $$bindings.resolved_url && resolved_url !== void 0)
    $$bindings.resolved_url(resolved_url);
  if ($$props.update_camera === void 0 && $$bindings.update_camera && update_camera !== void 0)
    $$bindings.update_camera(update_camera);
  if ($$props.reset_camera_position === void 0 && $$bindings.reset_camera_position && reset_camera_position !== void 0)
    $$bindings.reset_camera_position(reset_camera_position);
  url = value.url;
  {
    {
      resolved_url = url;
      if (url) {
        latest_url = url;
        const resolving_url = url;
        resolve_wasm_src(url).then((resolved) => {
          if (latest_url === resolving_url) {
            resolved_url = resolved ?? void 0;
          } else {
            resolved && URL.revokeObjectURL(resolved);
          }
        });
      }
    }
  }
  return `<canvas${add_attribute("this", canvas, 0)}></canvas>`;
});

export { Canvas3D as default };
//# sourceMappingURL=Canvas3D-Djh4scOm.js.map
