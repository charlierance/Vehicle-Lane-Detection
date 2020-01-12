"""
Interesting things found:
    - Occluded camera image throws error when finding chessboard corners as expected values not found.
"""

import glob

import cv2

import matplotlib.pyplot as plt
import numpy as np
from src.lane_detection_pipeline.utils import GeneralUtils


class CalibrateCamera:
    """
    This class have be used to calibrate the camera image, for this we perform the following steps:
        1) Determine the camera matrix and distortion coeffcients to undistort the image for both the radial and
           tangential distortion.
        2) Apply the camera calibration to the image, highlighting the change in image points.
        3) Carry out a perspective transform to change the image from forward, to a top down perspective.
    """

    def __init__(self, distorted_image, parameters: dict):
        self.distorted_image = distorted_image
        self.param = parameters

    def calibrate(self, checkerboard_image_glob: list):
        """
        :return: mtx: Required camera matrix
                 dist: The distortion coefficients (K_1, K_2, P_1, P_2, K_3)
        """
        # Initialise empty arrays
        # imgpoints = 2D image points (x, y)
        # objpoints = 3D points in real world (x, y, z) note z=0 due to flat image

        imgpoints = []
        objpoints = []

        # Initialise objpoints array
        ny = self.param["checkerboard_corners"]["ny"]
        nx = self.param["checkerboard_corners"]["nx"]
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for idx, fname in enumerate(checkerboard_image_glob):

            img = cv2.imread(fname)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

                img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                save_name = fname.split("/")[-1].split(".")[0] + "_corners.jpg"
                save_name = f"../../output_images/calibration/found_corners/{save_name}"
                cv2.imwrite(save_name, img)

            else:
                print(f"Could not find checkerboard corners for {fname}")

        imgsize = (self.param["image_shape"]["x"], self.param["image_shape"]["y"])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, imgsize, None, None
        )

        return mtx, dist

    @staticmethod
    def save_params(param_file, mtx, dst):
        utils = GeneralUtils()
        data = utils.read_yaml(param_file)
        data["calibration_values"] = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dst.tolist(),
        }
        utils.write_yaml(param_file, data)

    def undistort_image(
        self, image, camera_matrix, distortion_coefficients, save_image=False
    ):
        """
        :param camera_matrix: The camera matrix that is determined by running the determine_coefficients_and_matrix()
                              method.
        :param distortion_coefficients: The distortion coefficients that are determined by running the
                                        determine_coefficients_and_matrix() method.
        :param save_image: Save the image to disk in a predefined location showing before and after.
        :return: An undistorted bgr color image.
        """
        dst = cv2.undistort(
            image, camera_matrix, distortion_coefficients, None, camera_matrix
        )

        if save_image:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(image)
            ax1.set_title("Original Image", fontsize=30)
            ax2.imshow(dst)
            ax2.set_title("Undistorted Image", fontsize=30)
            plt.savefig(
                "../../output_images/calibration/calibrated_and_undistorted_image.jpg"
            )

        return dst


################################ TEST FUNCTION ##############################################


def run_calibration(distorted_image=None, param_file=None, raw_images=None):

    # Read parameters
    utils = GeneralUtils()
    calibration_params = utils.read_yaml(param_file)

    # Carry out calibration routine
    raw_images = glob.glob(raw_images)
    camera_calibration = CalibrateCamera(distorted_image, calibration_params)
    mtx, dist = camera_calibration.calibrate(raw_images)
    camera_calibration.save_params(param_file, mtx, dist)
    dst = camera_calibration.undistort_image(mtx, dist, save_image=True)

    cv2.imshow("Undistorted Image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration(
        distorted_image="../../camera_cal_images/calibration1.jpg",
        param_file="./config/calibration_parameters.yaml",
        raw_images="../../camera_cal_images/calibration*.jpg",
    )
