import numpy as np
import rawpy
import imageio
import os
import ncempy.io as nio
import matplotlib.pyplot as plt


# -- Osmo action use DNG as the raw format.
class DNGReader:
    @staticmethod
    def imread(folder_path, str_format, start_idx, end_idx, step, log_compress=False, im_format='DNG'):
        img_stack = []
        for img_id in range(start_idx, end_idx+1, step):
            file_name = os.path.join(folder_path, str_format.format(img_id))
            if im_format == 'DNG':
                with rawpy.imread(file_name) as raw:
                    img = raw.postprocess(gamma=(1, 1), no_auto_bright=True, no_auto_scale=True, output_bps=16)
                    img_stack.append(img)
            elif im_format == 'JPG' or im_format == "PNG":
                img = imageio.imread(file_name)
                img_stack.append(img)

        img_stack = np.array(img_stack).astype(np.float)
        mean_img = np.mean(img_stack, axis=0)
        if log_compress:
            img_stack = np.log(img_stack)
            mean_img = np.log(mean_img)
        return img_stack, mean_img


class SERReader:
    @staticmethod
    def imread(file_path, im_format='SER'):
        if im_format == 'AVI':
            cap = cv2.VideoCapture(file_path)
            frame_stack = []
            frame_len = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_stack.append(frame)
                frame_len += 1
            cap.release()
            return frame_stack, frame_len
        elif im_format == 'SER':
            with nio.ser.fileSER(file_path) as ser1:
                data, metadata = ser1.getDataset(0)
            return None, None


class FolderReader:
    @staticmethod
    def imread(path):
        img_list = os.listdir(path)
        img_list.sort(key=lambda x: int(x.split('_')[0]))
        frame_stack = []
        for name in img_list:
            img = cv2.imread(os.path.join(path, name))
            frame_stack.append(img)

        return frame_stack
