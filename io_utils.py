import numpy as np
import rawpy
import imageio
import os


# -- Osmo action use DNG as the raw format.
class DNG_Reader:
    @staticmethod
    def imread(folder_path, str_format, start_idx, end_idx, step, log_compress=False, im_format='DNG'):
        img_stack = []
        for img_id in range(start_idx, end_idx+1, step):
            file_name = os.path.join(folder_path, str_format.format(img_id))
            if im_format == 'DNG':
                with rawpy.imread(file_name) as raw:
                    img = raw.postprocess(gamma=(1, 1), no_auto_bright=True, no_auto_scale=True, output_bps=16)
                    img_stack.append(img)
            elif im_format == 'JPG':
                img = imageio.imread(file_name)
                img_stack.append(img)

        img_stack = np.array(img_stack).astype(np.float)
        mean_img = np.mean(img_stack, axis=0)
        if log_compress:
            img_stack = np.log(img_stack)
            mean_img = np.log(mean_img)
        return img_stack, mean_img


