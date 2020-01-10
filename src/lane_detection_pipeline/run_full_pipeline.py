import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from src.lane_detection_pipeline.utils import GeneralUtils
from src.lane_detection_pipeline.image_processing_pipeline import ImageProcessingPipeline
from src.lane_detection_pipeline.lane_detection_pipeline import LaneDetectionPipeline, SavedPolynomialValues

def run_full_pipeline(image):
    global left_previous
    global right_previous

    utils = GeneralUtils()
    calib_params = utils.read_yaml("/home/charlie/repos/self_driving_car_nanodegree/lane_detection_node/src/lane_detection_pipeline/config/calibration_parameters.yaml")
    img_params = utils.read_yaml("/home/charlie/repos/self_driving_car_nanodegree/lane_detection_node/src/lane_detection_pipeline/config/image_processing_parameters.yaml")
    lane_params = utils.read_yaml("/home/charlie/repos/self_driving_car_nanodegree/lane_detection_node/src/lane_detection_pipeline/config/lane_detection_parameters.yaml")

    # Instanctiate our pipelines
    image_pipeline = ImageProcessingPipeline(calib_params, img_params)
    lane_pipeline = LaneDetectionPipeline(lane_params)

    # Image processing steps
    dst = image_pipeline.undistorted_image(image, draw_roi=False)
    warped, minv = image_pipeline.perspective_warp(dst)
    hls_image, l_channel, s_channel = image_pipeline.convert_image_to_hls(warped)
    x_grad = image_pipeline.abs_sobel_thresh(l_channel, orient='x')
    y_grad = image_pipeline.abs_sobel_thresh(l_channel, orient='y')
    mag_grad = image_pipeline.mag_thresh(l_channel)
    dir_grad = image_pipeline.dir_thresh(l_channel)
    combined_grad = image_pipeline.combine_gradients(x_grad, y_grad, mag_grad, dir_grad)
    extracted_colour = image_pipeline.extract_colour_channel(s_channel)
    processed_image = image_pipeline.combine_colour_gradient(combined_grad, extracted_colour)

    # Carry out lane detection on binary image.
    image_shape = processed_image.shape

    if left_previous.detected or right_previous.detected == False:

        leftx_base, rightx_base, hist, out_img = lane_pipeline.histogram_image(processed_image, debug_img=False)

        leftx, lefty, rightx, righty, out_img = lane_pipeline.determine_lane_pixels(
            processed_image, leftx_base, rightx_base, debug_img=True, out_img=out_img
        )


        left_fit, right_fit, left_fit_m, right_fit_m, left_fitx, right_fitx, ploty, out_img = lane_pipeline.fit_poly(
            image_shape, leftx, lefty, rightx, righty, debug_img=True, out_img=out_img
        )

        rad_curve_l, rad_curve_r, curvature = lane_pipeline.find_radius_of_curvature(left_fit_m, right_fit_m, ploty)
        off_centre = lane_pipeline.find_position_off_centre(image_shape, left_fit_m, right_fit_m)

    else:
        left_fit, right_fit, left_fit_m, right_fit_m, left_fitx, right_fitx, ploty, result = \
            lane_pipeline.search_around_poly(processed_image, left_previous.current_fit, right_previous.current_fit, debug_img=True)

        rad_curve_l, rad_curve_r, curvature = lane_pipeline.find_radius_of_curvature(left_fit_m, right_fit_m, ploty)
        off_centre = lane_pipeline.find_position_off_centre(image_shape, left_fit_m, right_fit_m)

    # Smooth fitting
    if left_previous.initialised and right_previous.initialised:
        left_fitx = (np.array(left_fitx)+np.array(left_previous.bestx))/2
        right_fitx = (np.array(right_fitx)+np.array(right_previous.bestx))/2

    # Store our values
    if len(left_fit) and len(right_fit) > 0:

        left_previous.initialised = True
        left_previous.detected = True
        left_previous.recent_xfitted = left_fitx
        if left_previous.bestx is not None:
            left_previous.bestx = (np.array(left_previous.bestx) + np.array(left_fitx)) / 2.0
        else:
            left_previous.bestx = left_fitx

        left_previous.radius_of_curvature = curvature
        left_previous.line_base_pos = off_centre
        if left_previous.current_fit is not None:
            left_previous.diffs = left_previous.current_fit - left_fit
        left_previous.current_fit = left_fit

        right_previous.initialised = True
        right_previous.detected = True
        right_previous.recent_xfitted = right_fitx
        if right_previous.bestx is not None:
            right_previous.bestx = (np.array(right_previous.bestx) + np.array(right_fitx)) / 2.0
        else:
            right_previous.bestx = right_fitx

        right_previous.radius_of_curvature = curvature
        right_previous.line_base_pos = off_centre
        if right_previous.current_fit is not None:
            right_previous.diffs = right_previous.current_fit - right_fit
        right_previous.current_fit = right_fit

    # Create drawing image
    warp_zero = np.zeros_like(processed_image).astype(np.uint8)
    colour_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Add overlays to image
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(colour_warp, np.asarray([pts], np.int32), (0, 255, 0))
    cv2.polylines(colour_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(colour_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
    unwarp = cv2.warpPerspective(colour_warp, minv, (image_shape[1], image_shape[0]))

    # Overlay lane
    lane_overlay = cv2.addWeighted(dst, 1, unwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    rad_text = "Radius of Curvature: " + "{:04.2f}".format(curvature) + "(m)"
    cv2.putText(lane_overlay, rad_text, (40, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    offset_text = "Ego Pose vs Centre Line: {:04.3f}".format(off_centre) + "(m)"
    cv2.putText(lane_overlay, offset_text, (40, 120), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return lane_overlay

def run_on_image(image):
    # Instanciate Left and Right Lane Holders
    left_previous = SavedPolynomialValues()
    right_previous = SavedPolynomialValues()

    pipeline = run_full_pipeline(image)
    return pipeline

def run_on_video(input_video=None, output_path=None):
    left_previous = SavedPolynomialValues()
    right_previous = SavedPolynomialValues()

    input = VideoFileClip(input_video)
    process_video = input.fl_image(run_full_pipeline)






