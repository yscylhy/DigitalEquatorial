import numpy as np
import imageio
import matplotlib.pyplot as plt
import io_utils as io
import process_utils as utils

if __name__ == "__main__":
    reader = io.DNG_Reader()
    line_finder = utils.LineFinder()
    polaris_finder = utils.PolarisFinder()

    img_stack, avg_img = reader.imread(folder_path='./DCIM/20200529_niulang', str_format='DJI_0{}.DNG',
                                       start_idx=515, end_idx=533, step=1, log_compress=False, im_format='DNG')
    plt.figure()
    plt.imshow(np.log(avg_img[:,:,1]))
    plt.show()
    # print(img_stack.shape)
    # roi = img_stack[:,:,:,1]
    # roi = img_stack[:, 200:1500, 2000:3100, 1]
    roi = img_stack[:, :1500, 1200:2200, 1]

    Q1, Q2 = line_finder.detect_movement(roi, line_window=31)
    polaris_finder.cross_polaris(roi, Q2, frame_step=8, ransac_thresh=50, peak_distance=31, peak_number=100)



