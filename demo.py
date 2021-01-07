import numpy as np
import matplotlib.pyplot as plt
import io_utils
import os
import ransac_engine
import process_utils
import imageio


if __name__ == "__main__":
    data_path = './data/OSMO'
    frame_stack, frame_mean = \
        io_utils.DNGReader.imread(data_path, "DJI_0{}.JPG", start_idx=739, end_idx=747, step=1,
                                  log_compress=False, im_format='JPG')
    frame_len = len(frame_stack)

    print("Start processing {} images.".format(frame_len))

    canvas = frame_stack[0]
    for i in range(1, frame_len):
        arrow_map, h_pos_map, w_pos_map = \
            process_utils.patch_register(frame_stack[i], canvas, patch_size=300, margin=50, step=300)

        row_solver = ransac_engine.RANSAC(arrow_map[:, :, 0].flatten(), h_pos_map.flatten(), w_pos_map.flatten(),
                                          iter_num=2000, thresh=1/8, polynomial='quadratic')
        row_fit_param, row_fit_idx, row_residual = row_solver.fit(l2_prune=True)

        col_solver = ransac_engine.RANSAC(arrow_map[:, :, 1].flatten(), h_pos_map.flatten(), w_pos_map.flatten(),
                                          iter_num=2000, thresh=1/8, polynomial='quadratic')
        col_fit_param, col_fit_idx, col_residual = col_solver.fit(l2_prune=True)
        print("{}th done: row res.: {:.2f}. col res.: {:.2f}.".format(i, row_residual, col_residual))

        overlay = process_utils.Overlayer(canvas, frame_stack[i], row_fit_param, col_fit_param, i/(i+1))
        canvas = overlay.overlay()

    log_canvas = np.log(canvas)
    normal_log_canvas = process_utils.low_pass_filter(log_canvas, 0, 3)
    imageio.imwrite('milky_way.png', process_utils.normalize(normal_log_canvas).astype(np.uint8))

