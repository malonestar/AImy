# AImy/services/yolo11x_trigger_service.py
import cv2, time
from pathlib import Path
from adapters.axera_utils import Detector
from notify import send_discord_message, send_discord_image
from services.vision.camera_pipeline import CameraPipeline
from services.vision.roi_manager import ROIManager
from services.vision.presence_trigger import PresenceTrigger
from services.vision.viewer import Viewer
from core.event_names import (
    VISION_ROI_DETECT_MODE_ON, VISION_ROI_SAVED, VISION_ROI_CANCELED,
    VISION_PERSON_PERSISTED, VISION_INFER_PAUSED, VISION_INFER_RESUMED
)
from services.vision.frame_broadcast import publish_frame
from services.vision import vision_state


def run_yolo11x_trigger_loop(detector, cap_width: int, cap_height: int, bus=None):

    cam = CameraPipeline(detector, cap_width, cap_height)
    cam.start()

    roi = ROIManager()
    presence = PresenceTrigger(target_label="person", min_seconds=5, conf_thresh=0.5)

    viewer = None
    callback_attached = False

    output_path = Path("output")
    output_path.mkdir(exist_ok=True)

    print("[INFO] Starting feed. Press 'd' to draw ROI, 'q' to quit.")
    prev_time = time.time()

    # Local pause flag (owned by this loop)
    state = {"pause_infer": False}

    # Local flag that the bus callback can set
    want_roi_editor = False

    # react to pause/resume events
    if bus:
        def _on_pause(evt):
            state["pause_infer"] = True

        def _on_resume(evt):
            state["pause_infer"] = False
            presence.reset()
            print("[INFO] Vision inference resumed.")

        bus.subscribe(VISION_INFER_PAUSED, _on_pause)
        bus.subscribe(VISION_INFER_RESUMED, _on_resume)

        # callback only sets intent/flags
        def _on_roi_edit(evt):
            nonlocal want_roi_editor
            want_roi_editor = True
            vision_state.set_mode(vision_state.ROI_EDIT)
            roi.enable_detect_mode()

        bus.subscribe(VISION_ROI_DETECT_MODE_ON, _on_roi_edit)

    try:
        while True:
            try:
                original_frame, preprocessed_frame, ratio, pad_w, pad_h = cam.get(timeout=1)
            except Exception:
                continue

            #  Create the OpenCV window inside the vision thread
            if want_roi_editor and viewer is None:
                viewer = Viewer(
                    window_name="LLM 8850 Object Detection",
                    screen_size=(1024, 600)
                )
                callback_attached = False
                want_roi_editor = False
                print("[INFO] OpenCV viewer opened for ROI edit")

            # Inference
            if not state["pause_infer"]:
                detections = detector.infer_single(
                    preprocessed_frame, original_frame.shape[:2], ratio, pad_w, pad_h
                )
            else:
                detections = []

            result_frame = detector.draw_detections(original_frame.copy(), detections)

            # Presence logic
            person_fire, seconds_in = (False, 0.0)
            if roi.roi_defined and roi.roi_box:
                person_fire, seconds_in = presence.check(detections, detector.labels, roi.roi_box)

            if person_fire:
                if bus:
                    bus.publish(VISION_INFER_PAUSED, None)

                print(f"[ALERT] Person detected in ROI for {presence.min_seconds} seconds!")
                timestamp = int(time.time())
                screenshot_path = output_path / f"alert_{timestamp}.jpg"

                image_to_save_with_boxes = detector.draw_detections(original_frame.copy(), detections)
                cv2.imwrite(str(screenshot_path), image_to_save_with_boxes)

                send_discord_message("Person detected in Region of Interest!")
                send_discord_image(str(screenshot_path), message="Detection snapshot attached.")

                if bus:
                    bus.publish(VISION_PERSON_PERSISTED, {"roi": roi.roi_box, "image": str(screenshot_path)})

            # Overlays / HUD
            result_frame = roi.overlay(result_frame)
            result_frame = roi.hud_text(result_frame, paused=state["pause_infer"])

            # In-ROI timer text (only while counting)
            if roi.roi_defined and seconds_in and not person_fire:
                cv2.putText(
                    result_frame, f"Person in ROI: {seconds_in:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )

            # FPS
            now = time.time()
            fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now
            cv2.putText(
                result_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Publish to MJPEG (frame_broadcast gates this by vision_state)
            publish_frame(result_frame)

            # If ROI editor window is open, display it
            if viewer:
                viewer.show(result_frame)

                # Attach mouse cb after first frame shows
                if not callback_attached:
                    roi.attach_to_window(viewer.window)
                    callback_attached = True

            # Keys only work if viewer exists
            key = viewer.wait_key(1) if viewer else -1

            if key == ord("q"):
                break

            elif key == ord("d"):
                if bus:
                    bus.publish(VISION_ROI_DETECT_MODE_ON, None)
                else:
                    vision_state.set_mode(vision_state.ROI_EDIT)
                    roi.enable_detect_mode()
                    want_roi_editor = True

            elif key == ord("s") and roi.detect_mode:
                if roi.save_roi():
                    presence.reset()
                    if bus:
                        bus.publish(VISION_ROI_SAVED, {"roi": roi.roi_box})

                # close ROI editor + resume stream
                if viewer:
                    viewer.close()
                    viewer = None
                    callback_attached = False

                vision_state.set_mode(vision_state.STREAM)

            elif key == ord("c") and roi.detect_mode:
                roi.cancel_roi()
                presence.reset()

                if viewer:
                    viewer.close()
                    viewer = None
                    callback_attached = False

                if bus:
                    bus.publish(VISION_ROI_CANCELED, None)

                vision_state.set_mode(vision_state.STREAM)

            elif key == ord("r"):
                if bus:
                    bus.publish(VISION_INFER_RESUMED, None)
                else:
                    state["pause_infer"] = False
                    presence.reset()
                    print("[INFO] Vision inference resumed.")

    finally:
        vision_state.set_mode(vision_state.STREAM)
        if bus:
            bus.unsubscribe(VISION_INFER_PAUSED, _on_pause)
            bus.unsubscribe(VISION_INFER_RESUMED, _on_resume)
            bus.unsubscribe(VISION_ROI_DETECT_MODE_ON, _on_roi_edit)

        print("[INFO] Shutting down...")
        cam.stop()

        if viewer:
            viewer.close()
            viewer = None
            callback_attached = False

        try:
            del detector
        except Exception:
            pass
