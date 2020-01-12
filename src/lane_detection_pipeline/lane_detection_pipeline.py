import glob
from typing import List

import cv2

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from src.lane_detection_pipeline.image_processing_pipeline import \
    ImageProcessingPipeline
from src.lane_detection_pipeline.utils import GeneralUtils


@dataclass
class SavedPolynomialValues:
    initialised: bool = False

    # was the line detected in the last iteration?
    detected: bool = False

    # x values of the last n fits of the line
    recent_xfitted: List = None

    # average x values of the fitted line over the last n iterations
    bestx = None

    # polynomial coefficients averaged over the last n iterations
    best_fit = None

    # polynomial coefficients for the most recent fit
    current_fit = None

    # radius of curvature of the line in some units
    radius_of_curvature = None

    # distance in meters of vehicle center from the line
    line_base_pos = None

    # difference in fit coefficients between last and new fits
    diffs = None


class LaneDetectionPipeline:
    def __init__(self, parameters):
        self.params = parameters

    @staticmethod
    def histogram_image(binary_warped, debug_img=False):
        """
        :param binary_warped: Expects the binary image output of the image processing pipeline.
        :param debug_img: If true return a visual image for debugging purposes.
        :return: Take a histogram of the lower half of the image and split it into the left and right base
                (leftx_base and rightx_base). The reasoning behind taking the lower half of the image is that our lane
                lines will be less curved in the lower half of the image, therefore it will be clearer for peak finding
                to isolate our lanes.
        """

        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        return leftx_base, rightx_base, histogram, out_img

    def determine_lane_pixels(
        self, binary_warped, leftx_base, rightx_base, debug_img=False, out_img=None
    ):
        """
        :param binary_warped: Expected the binary warped image produced by the image processing pipeline
        :param leftx_base: The left bottom half of the histogram
        :param rightx_base: The right bottom half of the histogram
        :param debug_img: If true return a debug image showing our sliding windows.
        :param out_img: If debug_img is true we also need the output debug image produced by histogram_image()
        :return: The pixel positions of those determined to be our lane split into left and right x and y. This is done
                 by using the sliding windows algorithm to determine the mean in each window and adjusting if below the
                 min_pix threshold.
        """
        nwindows = self.params["sliding_windows"]["nwindows"]
        margin = self.params["sliding_windows"]["margin"]
        min_pix = self.params["sliding_windows"]["min_pix"]

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if debug_img:
                # Draw the windows on the visualization image
                cv2.rectangle(
                    out_img,
                    (win_xleft_low, win_y_low),
                    (win_xleft_high, win_y_high),
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(
                    out_img,
                    (win_xright_low, win_y_low),
                    (win_xright_high, win_y_high),
                    (0, 255, 0),
                    2,
                )

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > min_pix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > min_pix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(
        self, img_shape, leftx, lefty, rightx, righty, debug_img=False, out_img=None
    ):
        """
        :param img_shape: Shape of the image we are fitting a polynomial to, used to determine y-values.
        :param leftx: Our LH non zero x points within the window.
        :param lefty: Our LH non zero y points within the window.
        :param rightx: Out RH non zero x points within the window.
        :param righty: Out RH non zero y points within the window.
        :param debug_img: If true return a debug image.
        :param out_img: The output image to write the debug image to.
        :return: left_fit: Our returned LH lane polynomial coefficients.
                 right_fit: Our returned RH lane polynomial coefficients.
                 left_fitx: Our LH x fitted points  based on polynomial coefficients.
                 right_fitx Our RH x fitted points based on the polynomial coefficients.
                 ploty: Our y points for both x and y
                 out_img: Our output debug image if debug true, else None value.
        """
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # In meters
        ym_per_pix = self.params["radius_of_curvature"][
            "ym_per_pix"
        ]  # meters per pixel in y dimension
        xm_per_pix = self.params["radius_of_curvature"][
            "xm_per_pix"
        ]  # meters per pixel in y dimension
        left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if debug_img:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            draw_points_l = (np.asarray([left_fitx, ploty]).T).astype(np.int32)
            draw_points_r = (np.asarray([right_fitx, ploty]).T).astype(np.int32)

            cv2.polylines(out_img, [draw_points_l], False, (0, 255, 255), thickness=3)
            cv2.polylines(out_img, [draw_points_r], False, (0, 255, 255), thickness=3)

        return (
            left_fit,
            right_fit,
            left_fit_m,
            right_fit_m,
            left_fitx,
            right_fitx,
            ploty,
            out_img,
        )

    def search_around_poly(
        self, binary_warped, previous_left_fit, previous_right_fit, debug_img=False
    ):
        """
        :param binary_warped: Our image processing pipeline output image.
        :param previous_left_fit: Previous LH polynomial coefficients.
        :param previous_right_fit: Previous RH polynomial coefficients.
        :param debug_img: Bool, if True return a debug image.
        :return: new_left_fit: New LH polynomial coefficients.
                 new_right_fit: New Rh polynomial coefficients.
                 left_fitx: Left fitted x points based on the polynomial coefficients.
                 right_fitx: Right fitted x points based on the polynomial coefficients.
                 ploty: Y points for both lines.
                 result: If debug_img True then return a debug image else return None.
        """

        margin = self.params["sliding_windows"]["margin"]

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Here we see if the non zero point is greater than or less that out polynomial point +/- our margin on the x axis
        left_lane_inds = (
            nonzerox
            > (
                previous_left_fit[0] * (nonzeroy ** 2)
                + previous_left_fit[1] * nonzeroy
                + previous_left_fit[2]
                - margin
            )
        ) & (
            nonzerox
            < (
                previous_left_fit[0] * (nonzeroy ** 2)
                + previous_left_fit[1] * nonzeroy
                + previous_left_fit[2]
                + margin
            )
        )

        right_lane_inds = (
            nonzerox
            > (
                previous_right_fit[0] * (nonzeroy ** 2)
                + previous_right_fit[1] * nonzeroy
                + previous_right_fit[2]
                - margin
            )
        ) & (
            nonzerox
            < (
                previous_right_fit[0] * (nonzeroy ** 2)
                + previous_right_fit[1] * nonzeroy
                + previous_right_fit[2]
                + margin
            )
        )

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        new_left_fit, new_right_fit, new_left_m, new_right_m, left_fitx, right_fitx, ploty, out_img = self.fit_poly(
            binary_warped.shape,
            leftx,
            lefty,
            rightx,
            righty,
            out_img=np.dstack((binary_warped, binary_warped, binary_warped)) * 255,
        )

        if debug_img:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array(
                [np.transpose(np.vstack([left_fitx - margin, ploty]))]
            )
            left_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))]
            )
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array(
                [np.transpose(np.vstack([right_fitx - margin, ploty]))]
            )
            right_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))]
            )
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Draw new polylines
            draw_points_l = (np.asarray([left_fitx, ploty]).T).astype(np.int32)
            draw_points_r = (np.asarray([right_fitx, ploty]).T).astype(np.int32)

            cv2.polylines(result, [draw_points_l], False, (0, 255, 255), thickness=3)
            cv2.polylines(result, [draw_points_r], False, (0, 255, 255), thickness=3)

            return (
                new_left_fit,
                new_right_fit,
                new_left_m,
                new_right_m,
                left_fitx,
                right_fitx,
                ploty,
                result,
            )
        else:
            result = None
            return (
                new_left_fit,
                new_right_fit,
                new_left_m,
                new_right_m,
                left_fitx,
                right_fitx,
                ploty,
                result,
            )

    def find_radius_of_curvature(self, left_fit_m, right_fit_m, ploty):
        """
        :param left_fit_m: The left fit polynomial converted to meters
        :param right_fit_m: The right fit polynomial converted into meters.
        :param ploty: The y-points.
        :return: The left and right radius of curvature and the average radius of curvature.
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = self.params["radius_of_curvature"][
            "ym_per_pix"
        ]  # meters per pixel in y dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = (
            (1 + (2 * left_fit_m[0] * y_eval * ym_per_pix + left_fit_m[1]) ** 2) ** 1.5
        ) / np.absolute(2 * left_fit_m[0])
        right_curverad = (
            (1 + (2 * right_fit_m[0] * y_eval * ym_per_pix + right_fit_m[1]) ** 2)
            ** 1.5
        ) / np.absolute(2 * right_fit_m[0])

        overall_curve_rad = round(((left_curverad + right_curverad) / 2), 3)

        return left_curverad, right_curverad, overall_curve_rad

    def find_position_off_centre(self, image_shape, left_coeff_m, right_coeff_m):
        ym_per_pix = self.params["radius_of_curvature"][
            "ym_per_pix"
        ]  # meters per pixel in y dimension
        xm_per_pix = self.params["radius_of_curvature"][
            "xm_per_pix"
        ]  # meters per pixel in y dimension

        ego_pose = image_shape[1] / 2 * xm_per_pix

        y_to_m = image_shape[0] * ym_per_pix

        l_x_intercept = (
            left_coeff_m[0] * y_to_m ** 2 + left_coeff_m[1] * y_to_m + left_coeff_m[2]
        )
        r_x_intercept = (
            right_coeff_m[0] * y_to_m ** 2
            + right_coeff_m[1] * y_to_m
            + right_coeff_m[2]
        )

        lane_mid_point = (l_x_intercept + r_x_intercept) / 2

        off_centre_value = ego_pose - lane_mid_point

        return round(off_centre_value, 3)


################################ TEST FUNCTION ##############################################


def test_lane_pipeline(
    plot_hist=False, plot_sliding_windows=False, plot_poly_search=False
):
    images = glob.glob("../../test_images/*.jpg")
    utils = GeneralUtils()
    calibration_params = utils.read_yaml("./config/calibration_parameters.yaml")
    image_processing_params = utils.read_yaml(
        "./config/image_processing_parameters.yaml"
    )
    lane_detection_params = utils.read_yaml("./config/lane_detection_parameters.yaml")

    for image in images:
        process_image = ImageProcessingPipeline(
            calibration_params, image_processing_params
        )
        dst = process_image.undistorted_image(image, draw_roi=False)
        warped, minv = process_image.perspective_warp(dst)
        hls_image, l_channel, s_channel = process_image.convert_image_to_hls(warped)
        x_grad = process_image.abs_sobel_thresh(l_channel, orient="x")
        y_grad = process_image.abs_sobel_thresh(l_channel, orient="y")
        mag_grad = process_image.mag_thresh(l_channel)
        dir_grad = process_image.dir_thresh(l_channel)
        combined_grad = process_image.combine_gradients(
            x_grad, y_grad, mag_grad, dir_grad
        )
        extracted_colour = process_image.extract_colour_channel(s_channel)
        combined_grad_colour = process_image.combine_colour_gradient(
            combined_grad, extracted_colour
        )

        # Lane line processing
        process_lanes = LaneDetectionPipeline(lane_detection_params)

        # Histogram
        leftx_base, rightx_base, hist, out_img = process_lanes.histogram_image(
            combined_grad_colour, debug_img=False
        )

        if plot_hist:
            hist_filename = image.split("/")[-1].split(".")[0]
            hist_filename = hist_filename + "_hist.jpg"
            plt.title("Lane Pixel Histogram")
            plt.xlabel("Image x-axis Position")
            plt.ylabel("Count")
            plt.plot(hist)
            plt.savefig(f"../../output_images/lane_detection/histogram/{hist_filename}")
            plt.close()

        # Sliding windows algorithm
        leftx, lefty, rightx, righty, out_img = process_lanes.determine_lane_pixels(
            combined_grad_colour,
            leftx_base,
            rightx_base,
            debug_img=True,
            out_img=out_img,
        )

        # Fit a polynomial
        image_shape = combined_grad_colour.shape
        left_fit, right_fit, left_fit_m, right_fit_m, left_fitx, right_fitx, ploty, out_img = process_lanes.fit_poly(
            image_shape, leftx, lefty, rightx, righty, debug_img=True, out_img=out_img
        )

        if plot_sliding_windows:
            sliding_filename = image.split("/")[-1].split(".")[0]
            sliding_filename = sliding_filename + "_sliding_window.jpg"
            cv2.imwrite(
                f"../../output_images/lane_detection/sliding_windows/{sliding_filename}",
                out_img,
            )

        # Fit polynomial based on previous result
        new_left_fit, new_right_fit, new_left_m, new_right_m, left_fitx, right_fitx, ploty, result = process_lanes.search_around_poly(
            combined_grad_colour, left_fit, right_fit, debug_img=True
        )

        if plot_poly_search:
            search_filename = image.split("/")[-1].split(".")[0]
            search_filename = search_filename + "_previous_search.jpg"
            cv2.imwrite(
                f"../../output_images/lane_detection/previous_polynomial_coefficients/{search_filename}",
                result,
            )

        rad_curve_l, rad_curve_r, curvature = process_lanes.find_radius_of_curvature(
            new_left_m, new_right_m, ploty
        )
        off_centre = process_lanes.find_position_off_centre(
            image_shape, new_left_m, new_right_m
        )
        print(image)
        print(f"Overall Radius of Curvature: {curvature}(m)")
        print(f"Distance Off Center Line: {off_centre}(m)")
        print("#" * 10)
        print("\n")

    cv2.destroyAllWindows()
