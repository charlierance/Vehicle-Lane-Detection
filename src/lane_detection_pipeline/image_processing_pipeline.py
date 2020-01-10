import numpy as np
import glob
import cv2

from src.lane_detection_pipeline.utils import GeneralUtils
from src.lane_detection_pipeline.calibratecamera import CalibrateCamera


class ImageProcessingPipeline:

    def __init__(self, calibration_parameters, image_processing_params):
        self.calibration_parameters = calibration_parameters
        self.image_processing_params = image_processing_params
        self.kernel = self.image_processing_params["gradient_params"]["sobel_kernel"]

    def undistorted_image(self, bgr_image, draw_roi=False):
        """
        :param bgr_image: Takes an input undistorted bgr image.
        :param draw_roi: If true, plot the roi that will be used to transform the image perspective.
        :return: Takes in the raw image instantiated in the camera calibration module and uses the previously computed
                 matrix and distortion coefficients to return an undistorted image.
        """
        # Read params
        camera_matrix = np.asarray(self.calibration_parameters["calibration_values"]["camera_matrix"])
        distortion_coefficient = np.asarray(
            self.calibration_parameters["calibration_values"]["distortion_coefficients"]
        )

        calibrate_camera = CalibrateCamera(bgr_image, self.calibration_parameters)

        dst = calibrate_camera.undistort_image(camera_matrix, distortion_coefficient, save_image=False)

        if draw_roi:

            pts = np.array(
                [[211, 670],        # Bottom left
                [610, 435],        # Top left
                [670, 435],        # Top right
                [1090, 670]        # Bottom right
            ])
            return cv2.polylines(dst, [pts], 1, (0, 0, 255), thickness=3)

        return dst

    @staticmethod
    def perspective_warp(undistorted_image, offset=100):
        """
        :param undistorted_image: Takes in the bgr_undistorted image.
        :param offset: Define an offset by which we will used to produce the warped final image e.g. If offset = 100 and
                       the x-axis max = 200 then we can compute a transformed point 200-offset = 100
        :return: A perspective transformed image from a birds eye view.
        """

        img_size = (undistorted_image.shape[1], undistorted_image.shape[0])

        # Four source coordinates
        src = np.float32(
            [[205, 719],  # Bottom left
             [570, 460],  # Top left
             [745, 460],  # Top right
             [1145, 719]  # Bottom right
             ])

        # Four desired coordinates
        dst = np.float32([
            [offset, img_size[1]],
            [offset, 0],
            [img_size[0] - offset, 0],
            [img_size[0] - offset, img_size[1]]
        ])

        # Compute the perspective transform matrix M
        M = cv2.getPerspectiveTransform(src, dst)

        # Compute the inverse perspective transform for unwarping the image
        Minv = cv2.getPerspectiveTransform(dst, src)

        # Finally warm the image using the transform we have computed and linear interpolation
        warped = cv2.warpPerspective(undistorted_image, M, img_size, flags=cv2.INTER_LINEAR)

        return warped, Minv

    @staticmethod
    def convert_image_to_hls(warped_image):
        """
        :param warped_image: Expects a BGR undistorted image as an input.
        :return: hls_img: Return the HLS converted image
                 l_channel: Return the l_channel of hsl, hint - l_channel is useful for detecting white
                 s_channel: Return the s channel of hsl, hint - s_channel is useful for detecting yellow
        """
        # Convert to HLS color space and separate the V channel
        hls_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
        l_channel = hls_image[:, :, 1]
        s_channel = hls_image[:, :, 2]

        return hls_image, l_channel, s_channel

    def abs_sobel_thresh(self, img, orient='x'):
        """
        :param img: Expects the l_channel from the hls image.
        :param orient: Direction of gradient to search for, either x or y
        :return: A binary image of the matching thresholds found by the sobel matrix dependant on direction:

                sum(region(x or y)Sobel_(x or y)) > 0 = A Gradient

                Note: The the region size it defined by the kernel size
        """

        # Apply sobel matrix to image dependent on if x or y derivative required
        if orient.lower() == "x":
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel)
        elif orient.lower() == "y":
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel)

        # Find absolute value of the derivitives
        abs_sobel = np.absolute(sobel)

        # Convert from 64 bit image produced by sobel to 8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Define thresholds to capture appropriate gradients
        thresh = self.image_processing_params["gradient_params"]["gradient_binary_thresh"]
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def mag_thresh(self, img):
        """
        :param img:  Expects the l_channel from the hls image.
        :return: A binary image based upon the magnitude of the gradient vector:

                sqrt(sobel_x**2 + sobel_y**2)
        """
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel)

        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        # Create a binary image of ones where threshold is met, zeros otherwise
        mag_thresh = self.image_processing_params["gradient_params"]["magnitude_binary_thresh"]
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_thresh(self, img):
        """
        :param img: Expects the l_channel from the hls image.
        :return: A binary image based on the direction of the gradient:

                 arctan(sobel_y / soble_x)
        Note: In this case the take the absolute value of sobel* to limit the max/min value to +_- pi/2
        """
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        thresh = self.image_processing_params["gradient_params"]["direction_binary_thresh"]
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    @staticmethod
    def combine_gradients(x_gradient, y_gradient, magnitude_binary, direction_binary):
        """
        :param x_gradient: Image returned from abs_sobel_thresh() (x-direction)
        :param y_gradient: Image returned from  abs_sobel_thresh() (y-direction)
        :param magnitude_binary: Image returned from mag_thresh()
        :param direction_binary: Image returned from dir_thresh()
        :return:
        """
        combined = np.zeros_like(direction_binary)
        combined[((x_gradient == 1) & (y_gradient == 1)) | ((magnitude_binary == 1) & (direction_binary == 1))] = 1
        return combined

    def extract_colour_channel(self, img):
        """
        :param img: The s_channel from the hls image.
        :return: A binary image based upon the defined colour thresholds.
        """
        s_thresh = self.image_processing_params["gradient_params"]["s_channel_thresh"]

        # Threshold color channel
        s_binary = np.zeros_like(img)
        s_binary[(img >= s_thresh[0]) & (img <= s_thresh[1])] = 1

        return s_binary

    @staticmethod
    def combine_colour_gradient(gradient_binary, colour_binary):
        """
        :param gradient_binary: The binary image returned from all gradient filters.
        :param colour_binary: The binary image of the extracted colours.
        :return: An overlaid version of the combined image.
        """
        combined_binary = np.zeros_like(gradient_binary)
        combined_binary[((gradient_binary == 1) | (colour_binary ==1))] = 1
        return combined_binary

################################ TEST FUNCTION ##############################################

def test_image_pipeline(save_images=True):
    images = glob.glob("../../test_images/*.jpg")
    utils = GeneralUtils()
    calibration_params = utils.read_yaml("./config/calibration_parameters.yaml")
    image_processing_params = utils.read_yaml("./config/image_processing_parameters.yaml")

    for image in images:
        process_image = ImageProcessingPipeline(calibration_params, image_processing_params)
        dst = process_image.undistorted_image(image, draw_roi=False)
        warped, minv = process_image.perspective_warp(dst)
        hls_image, l_channel, s_channel = process_image.convert_image_to_hls(warped)
        x_grad = process_image.abs_sobel_thresh(l_channel, orient='x')
        y_grad = process_image.abs_sobel_thresh(l_channel, orient='y')
        mag_grad = process_image.mag_thresh(l_channel)
        dir_grad = process_image.dir_thresh(l_channel)
        combined_grad = process_image.combine_gradients(x_grad, y_grad, mag_grad, dir_grad)
        extracted_colour = process_image.extract_colour_channel(s_channel)
        combined_grad_colour = process_image.combine_colour_gradient(combined_grad, extracted_colour)

        if save_images:
            filename = image.split("/")[-1].split(".")[0]

            dst_filename = filename + "_undistorted.jpg"
            cv2.imwrite(f'../../output_images/image_processing/undistorted/{dst_filename}', dst)

            roi_filename = filename + "_roi.jpg"
            roi = process_image.undistorted_image(image, draw_roi=True)
            cv2.imwrite(f'../../output_images/image_processing/roi/{roi_filename}', roi)

            warped_filename = filename + "_warped_roi.jpg"
            warped_roi = process_image.perspective_warp(roi)
            cv2.imwrite(f'../../output_images/image_processing/warped_roi/{warped_filename}', warped_roi)

            hls_filename = filename + "_hls.jpg"
            cv2.imwrite(f'../../output_images/image_processing/hls/{hls_filename}', hls_image)

            x_filename = filename + "_x_grad.jpg"
            x_grad = x_grad*255
            x_grad = x_grad.astype('uint8')
            cv2.imwrite(f'../../output_images/image_processing/gradients/x/{x_filename}', x_grad)

            y_filename = filename + "_y_grad.jpg"
            y_grad = y_grad*255
            y_grad = y_grad.astype('uint8')
            cv2.imwrite(f'../../output_images/image_processing/gradients/y/{y_filename}', y_grad)

            mag_filename = filename + "_mag_grad.jpg"
            mag_grad = mag_grad*255
            mag_grad = mag_grad.astype('uint8')
            cv2.imwrite(f'../../output_images/image_processing/gradients/magnitude/{mag_filename}', mag_grad)

            dir_filename = filename + "_direction_grad.jpg"
            dir_grad = dir_grad*255
            dir_grad = dir_grad.astype('uint8')
            cv2.imwrite(f'../../output_images/image_processing/gradients/direction/{dir_filename}', dir_grad)

            comb_filename = filename + "_combined_grad.jpg"
            combined_grad = combined_grad*255
            combined_grad = combined_grad.astype('uint8')
            cv2.imwrite(f'../../output_images/image_processing/gradients/combined/{comb_filename}', combined_grad)

            col_filename = filename + "_colour_extracted.jpg"
            cv2.imwrite(f'../../output_images/image_processing/colour_extracted/{col_filename}', extracted_colour)

            final_filename = filename + "_colour_gradient_binary.jpg"
            cv2.imwrite(
                f'../../output_images/image_processing/colour_gradient_binary/{final_filename}', combined_grad_colour
            )

        cv2.imshow(" ", combined_grad_colour)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


