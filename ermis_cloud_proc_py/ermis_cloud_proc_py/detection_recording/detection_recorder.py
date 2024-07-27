import os
import numpy as np
import datetime
import yaml

"""
    This module is responsible for recording the detections of the system.
    Records the detected 3d bounding boxes in npy files.
    Each frame is saved in a separate file in a given directory.
    format: detected_bbox3d_{frame_number}.npy

    format for the npy file: TODO this has to be updated
    (
        detections: (
            id: int, # id of the detection, optional, for now a placeholder, to be compatible with tracking
            label: str, # label of the detection, optional, for now a placeholder
            points: np.ndarray, # 3d points of the bounding box, shape: (8, 3)
            centroid: np.ndarray, # centroid of the bounding box, shape: (3,)
            transform: np.ndarray, # transformation matrix of the bounding box, shape: (4, 4), optional, for now a placeholder
        ),
        seconds: int, # seconds of the timestamp of the detection
        nanoseconds: int, # nanoseconds of the timestamp of the detection
    )
"""

class DetectionRecorder:
    def __init__(self, save_dir, use_zed_wrapper_properties=True): # TODO read the optional argument from the config file
        self.save_dir = save_dir
        self.frame_number = 0

        # specific

        zed_wrapper_properties_suffix = ''

        USE_ZED_CONFIG_SUFFIX = False # TODO read the optional argument from the config file

        if USE_ZED_CONFIG_SUFFIX:
            with open('/home/colino/hawk/zed_wrapper_config/common.yaml', 'r') as file: # TODO read the path from the config file
                config = yaml.load(file, Loader=yaml.FullLoader)
                config = config['/**']['ros__parameters']
                pub_downscale_factor = config['general']['pub_downscale_factor']
                pub_frame_rate = config['general']['pub_frame_rate']
                depth_mode = config['depth']['depth_mode'].replace('_', '')
                point_cloud_freq = config['depth']['point_cloud_freq']
                depth_confidence = config['depth']['depth_confidence']
                depth_texture_conf = config['depth']['depth_texture_conf']

                zed_wrapper_properties_suffix = f'-pdf{pub_downscale_factor}_pfr{pub_frame_rate}_dm{depth_mode}_pcf{point_cloud_freq}_dc{depth_confidence}_dtf{depth_texture_conf}'
            
                self.save_dir = f'{self.save_dir}{zed_wrapper_properties_suffix}'

                print(f'\nsave_dir after zed_wrapper_properties_suffix: {self.save_dir}')

        # create the save directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir )
        else:
            # add suffix based on current date and time if the directory already exists and issue a warning
            now = datetime.datetime.now()
            suffix = now.strftime("%Y%m%d_%H%M%S")
            old_save_dir = self.save_dir
            self.save_dir = f'{self.save_dir }-{suffix}'
            os.makedirs(self.save_dir)
            print(f'\nWarning: The directory {old_save_dir} already exists. Saving detections to {self.save_dir} instead.')

    def record(self, bboxes, centroids, seconds, nanoseconds):
        # Define the dtype for the structured array
        dtype = np.dtype([
            ('id', 'i4'),  # int32 for id
            ('label', 'U50'),  # Unicode string with max length 50 for label
            ('points', 'f4', (8, 3)),  # float32 array with shape (8, 3) for points
            ('centroid', 'f4', (3,)),  # float32 array with shape (3,) for centroid
            ('transform', 'f4', (4, 4))  # float32 array with shape (4, 4) for transform
        ])

        # Create an empty structured array
        detections = np.empty(len(bboxes), dtype=dtype)

        # Fill the structured array
        for i in range(len(bboxes)):
            detections[i] = (
                -1,  # id placeholder, set to -1 or any default value
                '',  # label placeholder, set to empty string or any default value
                np.asarray(bboxes[i].get_box_points(), dtype='f4'),  # points
                np.asarray(centroids[i], dtype='f4'),  # centroid
                np.eye(4, dtype='f4')  # transform
            )

        # Define the dtype for the final structured array
        dtype_2 = np.dtype([
            ('detections', dtype, (len(bboxes),)),  # Array of detections
            ('seconds', 'i4'),
            ('nanoseconds', 'i4')
        ])

        # Create the structured array
        output_data = np.array(
            (
                detections,
                seconds,
                nanoseconds
            ),
            dtype=dtype_2
        )

        # Create the file name
        str_frame_number = f'{self.frame_number:05d}'
        npy_file = os.path.join(self.save_dir, f'detected_bbox3d_{str_frame_number}.npy')

        # Save the structured array and timestamps
        np.save(npy_file, output_data)

        self.frame_number += 1