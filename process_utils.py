import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, fftn, ifftn
from skimage.feature import peak_local_max
from scipy import ndimage
import heapq


def low_pass_filter(img, h_thresh, w_tresh):
    def _low_pass_filter(img):
        h, w = img.shape
        img_fft = np.fft.fft2(img)
        for di in range(-h_thresh, h_thresh+1):
            for dj in range(-w_tresh, w_tresh+1):
                i = (di+h)%h
                j = (dj+w)%w
                img_fft[i, j] = 0
        img = np.real(np.fft.ifft2(img_fft))
        return img

    if len(img.shape)>2:
        _, _, c = img.shape
        for c_idx in range(c):
            img[:,:,c_idx] = _low_pass_filter(img[:,:,c_idx])
    else:
        img = _low_pass_filter(img)

    return img


class Overlayer:
    def __init__(self, src_img, tar_img, row_fit_param, col_fit_param, src_weight):
        self.src_img = src_img
        self.tar_img = tar_img
        self.row_fit_param = row_fit_param
        self.col_fit_param = col_fit_param
        self.src_weight = src_weight
        if len(src_img)>2:
            self.h, self.w, self.c = src_img.shape
        else:
            self.h, self.w = src_img.shape
            self.c = None

    def overlay(self):
        t_src_img = self.dense_translate()
        overlayed_img = t_src_img*self.src_weight + self.tar_img*(1-self.src_weight)
        return overlayed_img

    def dense_translate(self):
        row_distortion_field = self.get_distortion(self.row_fit_param)
        col_distortion_field = self.get_distortion(self.col_fit_param)
        x = np.arange(0, self.w, 1)
        y = np.arange(0, self.h, 1)
        u, v = np.meshgrid(x, y)
        org_idx = np.vstack([v.flatten(), u.flatten()]).transpose()
        new_idx = np.array([org_idx[:, 0] + row_distortion_field, org_idx[:, 1] + col_distortion_field]).astype(np.int)
        new_idx[0, :] = np.maximum(new_idx[0, :], 0)
        new_idx[0, :] = np.minimum(new_idx[0, :], self.h-1)
        new_idx[1, :] = np.maximum(new_idx[1, :], 0)
        new_idx[1, :] = np.minimum(new_idx[1, :], self.w-1)

        if self.c is not None:
            src_canvas = np.zeros([self.h, self.w, self.c])
            for c_idx in range(self.c):
                temp = self.src_img[:, :, c_idx]
                src_canvas[:, :, c_idx] = temp[new_idx[0, :], new_idx[1, :]].reshape([self.h, self.w])
        else:
            src_canvas = self.src_img[new_idx[0, :], new_idx[1, :]]
            src_canvas.reshape([self.h, self.w])

        return src_canvas

    def get_distortion(self, fit_param):
        if len(self.src_img) > 2:
            h, w, c = self.src_img.shape
        else:
            h, w = self.src_img.shape
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)
        u, v = np.meshgrid(x, y)
        u = u.flatten()
        v = v.flatten()
        org_idx = np.vstack([v*v, u*u, v*u, v, u, np.ones(u.shape[0])])
        delta_idx = np.dot(fit_param, org_idx)
        return delta_idx


def plot_arrow_map(data, colors=None):
    h, w, _ = data.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    fig, ax = plt.subplots()
    if colors is not None:
        colors = colors.reshape(data.shape[0], data.shape[1])
        colors = 1-colors.astype(np.float)
        q = ax.quiver(x, y, data[:, :, 1], -data[:, :, 0], colors)
    else:
        q = ax.quiver(x, y, data[:, :, 1], -data[:, :, 0])
    ax.invert_yaxis()
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label='Quiver key, length = 10', labelpos='E')

    plt.show()


def translate(img, t):
    delta_row, delta_col = t
    translated_img = np.zeros(img.shape)
    if len(img.shape) == 3:
        h, w, _ = img.shape
        is_color = True
    else:
        h, w = img.shape
        is_color = False

    overlap_h = h - abs(delta_row)
    overlap_w = w - abs(delta_col)
    t_start_row = 0 if delta_row < 0 else delta_row
    t_start_col = 0 if delta_col < 0 else delta_col
    i_start_row = 0 if delta_row > 0 else -delta_row
    i_start_col = 0 if delta_col > 0 else -delta_col
    if is_color:
        translated_img[t_start_row:t_start_row+overlap_h, t_start_col:t_start_col+overlap_w, :] = \
            img[i_start_row:i_start_row+overlap_h, i_start_col:i_start_col+overlap_w, :]
    else:
        translated_img[t_start_row:t_start_row + overlap_h, t_start_col:t_start_col + overlap_w, :] = \
            img[i_start_row:i_start_row + overlap_h, i_start_col:i_start_col + overlap_w, :]

    return translated_img


def normalize(np_data):
    max_val = np.max(np_data.flatten())
    min_val = np.min(np_data.flatten())
    return (np_data - min_val)/(max_val-min_val)*255


def fft_register(img1, img2):
    if len(img1.shape) == 3:
        img1 = img1[:, :, 1]
        img2 = img2[:, :, 1]
    h, w = img1.shape
    idx = np.argmax(np.fft.ifft2(np.fft.fft2(img1)*np.conjugate(np.fft.fft2(img2))))
    row_idx, col_idx = np.unravel_index(idx, img1.shape)
    if row_idx > h//2:
        row_idx -= h
    if col_idx > w//2:
        col_idx -= w
    return [row_idx, col_idx]


def feature_register(img1, img2):
    if len(img1.shape) == 3:
        img1 = img1[:, :, 1].astype('uint8')
        img2 = img2[:, :, 1].astype('uint8')

    # -- Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # -- Find keypoints and descriptors.
    kp1, d1 = orb_detector.detectAndCompute(img1, mask=None)
    kp2, d2 = orb_detector.detectAndCompute(img2, mask=None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches.sort(key=lambda x: x.distance)

    # -- Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 10)]
    no_of_matches = len(matches)

    # -- Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

        # -- Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return homography


def feature_translate(img, t):
    if len(img.shape) == 3:
        h, w, _ = img.shape
        is_color = True
    else:
        h, w = img.shape
        is_color = False
    transformed_img = cv2.warpPerspective(img, t, (w, h))
    return transformed_img


def margin_register(src_patch, tar_patch):
    src_h, src_w = src_patch.shape
    tar_h, tar_w = tar_patch.shape
    assert tar_h >= src_h and tar_w >= src_w, "target patch should be larger than the source patch!"
    src_patch = src_patch - np.mean(src_patch)
    tar_patch = tar_patch - np.mean(tar_patch)

    src_canvas = np.zeros([tar_h, tar_w])
    src_canvas[:src_h, :src_w] = src_patch

    idx = np.argmax(np.fft.ifft2(np.fft.fft2(tar_patch) * np.conjugate(np.fft.fft2(src_canvas))))
    row_idx, col_idx = np.unravel_index(idx, tar_patch.shape)
    if row_idx > tar_h // 2:
        row_idx -= tar_h
    if col_idx > tar_w // 2:
        col_idx -= tar_w

    return [row_idx, col_idx]


def patch_register(src_img, tar_img, patch_size=100, margin=100, step=50):
    if len(src_img.shape) == 3:
        src_img = src_img[:, :, 1].astype('uint8')
        tar_img = tar_img[:, :, 1].astype('uint8')

    h, w = src_img.shape
    arrow_map = []
    h_pos_map = []
    w_pos_map = []
    for s_h in range(margin, h-margin-patch_size, step):
        arrow_map.append([])
        h_pos_map.append([])
        w_pos_map.append([])

        for s_w in range(margin, w-margin-patch_size, step):
            src_patch = src_img[s_h:s_h+patch_size, s_w:s_w+patch_size]
            tar_patch = tar_img[s_h-margin:s_h+patch_size+margin, s_w-margin:s_w+patch_size+margin]
            shift_vector = margin_register(src_patch, tar_patch)
            arrow_map[-1].append([x-margin for x in shift_vector])
            h_pos_map[-1].append(s_h + patch_size//2)
            w_pos_map[-1].append(s_w + patch_size//2)
    arrow_map = np.array(arrow_map)
    h_pos_map = np.array(h_pos_map)
    w_pos_map = np.array(w_pos_map)
    return arrow_map, h_pos_map, w_pos_map


# -- find the linear movement of the star (0.25 degree per minute. 15 degree per hour.)
class LineFinder:
    @classmethod
    def detect_movement(cls, img_stack, line_window=31):
        std_map = np.std(img_stack, axis=0)
        mean_map = np.mean(img_stack, axis=0)
        Q1, Q2 = cls._get_eigen_value(std_map, window_size=line_window)

        plt.figure()
        plt.imshow(np.log(std_map))
        plt.figure()
        plt.imshow(np.log(mean_map))
        plt.figure()
        plt.imshow(Q1)
        plt.figure()
        plt.imshow(Q2)
        plt.show()
        return Q1, Q2

    @classmethod
    def _get_eigen_value(cls, std_map, window_size):
        H, W = std_map.shape
        # -- get the optical transfer function of gradient on x and y directions. (similar to psf2otf in Matlab)
        dx, dy, spatial_window = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
        dx[H // 2, W // 2 - 1:W // 2 + 1] = [-1, 1]
        dy[H // 2 - 1:H // 2 + 1, W // 2] = [-1, 1]
        h_start, h_end = H // 2 - window_size // 2, H // 2 + window_size // 2 + 1
        w_start, w_end = W // 2 - window_size // 2, W // 2 + window_size // 2 + 1
        spatial_window[h_start:h_end, w_start:w_end] = 1 / (window_size ** 2)

        fourier_window = fft2(spatial_window)
        # -- Centered gradients
        dx, dy = np.zeros((H, W)), np.zeros((H, W))
        dx[:, 1:W - 1] = (std_map[:, 2:] - std_map[:, :W - 2]) / 2
        dy[1:H - 1, :] = (std_map[2:, :] - std_map[:H - 2, :]) / 2
        # dx_mean = np.real(fftshift(ifft2(fft2(dx) * fourier_window)))
        # dy_mean = np.real(fftshift(ifft2(fft2(dy) * fourier_window)))
        # dx = dx-dx_mean
        # dy = dy-dy_mean
        E_dx = np.real(fftshift(ifft2(fft2(dx ** 2) * fourier_window) * window_size ** 2))
        E_dy = np.real(fftshift(ifft2(fft2(dy ** 2) * fourier_window) * window_size ** 2))
        E_dxy = np.real(fftshift(ifft2(fft2(dx * dy) * fourier_window) * window_size ** 2))
        delta = np.abs((E_dx + E_dy) ** 2 - 4 * (E_dx * E_dy - E_dxy ** 2)) ** 0.5
        b = E_dx + E_dy
        lambda1 = np.abs((b - delta) / 2) ** 0.5
        lambda2 = np.abs((b + delta) / 2) ** 0.5

        # --- The meaning of Q1 is to be explored...
        Q1 = (E_dxy * lambda1) / (1 - E_dx * lambda1)
        Q2 = (abs(lambda2 - lambda1) / (lambda1 + lambda2))
        return Q1, Q2


class PolarisFinder:
    @classmethod
    def cross_polaris(cls, img_stack, star_prob, frame_step, ransac_thresh=100, peak_distance=11, peak_number=100):
        assert len(img_stack) >= 3, "More than 3 consecutive images required!"
        pre_peaks, cur_peaks, next_peaks = None, None, None
        for idx in range(frame_step, len(img_stack)-frame_step, frame_step):
            next_peaks = cls._get_spatial_peaks(img_stack[idx+frame_step], star_prob, peak_distance, peak_number)
            if cur_peaks is None:
                pre_peaks = cls._get_spatial_peaks(img_stack[idx - frame_step], star_prob, peak_distance, peak_number)
                cur_peaks = cls._get_spatial_peaks(img_stack[idx], star_prob, peak_distance, peak_number)

            pre_peaks, after_peaks = cls._get_peak_pair(pre_peaks, cur_peaks, next_peaks)
            peak_vis = cls._vis_points_pair(pre_peaks, after_peaks, img_stack.shape[1], img_stack.shape[2])
            polaris_pos, fit_number = cls._ransac_peak(pre_peaks, after_peaks,
                                                       thresh_hold=ransac_thresh, iter_number=500)
            print("{} {}".format(len(pre_peaks), fit_number))
            print(polaris_pos)
            plt.figure()
            plt.imshow(peak_vis)
            plt.show()
            pre_peaks, cur_peaks = cur_peaks, next_peaks

        return

    @classmethod
    def _get_spatial_peaks(cls, img, star_prob, peak_distance, peak_num):
        peaks = peak_local_max(img, min_distance=peak_distance)
        heap = []
        for peak in peaks:
            heapq.heappush(heap, [star_prob[peak[0], peak[1]], peak])
            if len(heap) > peak_num:
                heapq.heappop(heap)
        return np.array([x[1] for x in heap])

    @classmethod
    def _get_peak_pair(cls, peak0, peak1, peak2):
        cur_peak_id1, lut1 = cls._get_reliable_peak_id(peak1, peak0)
        cur_peak_id2, lut2 = cls._get_reliable_peak_id(peak1, peak2)
        commend_peak = set(cur_peak_id1) & set(cur_peak_id2)
        pre_peak_pos, after_peak_pos = [], []
        for peak_id in commend_peak:
            pos0, pos1 = peak0[lut1[peak_id]], peak2[lut2[peak_id]]
            if pos0[0] == pos1[0] or pos0[1] == pos1[1]:
                continue
            pre_peak_pos.append(pos0)
            after_peak_pos.append(pos1)
        return pre_peak_pos, after_peak_pos

    @classmethod
    def _get_reliable_peak_id(cls, peak1, peak2):
        peak1_stack = np.tile(peak1, [peak2.shape[0], 1, 1])
        peak2_stack = np.tile(peak2, [peak1.shape[0], 1, 1])
        peak2_stack = np.swapaxes(peak2_stack, 0, 1)
        pos_distance = (peak1_stack[:, :, 0] - peak2_stack[:, :, 0]) ** 2 \
                       + (peak1_stack[:, :, 1] - peak2_stack[:, :, 1]) ** 2
        lut12 = np.argmin(pos_distance, axis=0)
        lut21 = np.argmin(pos_distance, axis=1)
        reliable_list = [x for x in range(len(peak1)) if (lut21[lut12[x]] == x)]
        return reliable_list, lut12

    @classmethod
    def _ransac_peak(cls, pos1, pos2, thresh_hold, iter_number=100):
        best_fit_number = 0
        best_polaris = None
        line_eqn = tuple(cls._get_line_from_points(x, y) for x, y in zip(pos1, pos2))

        for i in range(iter_number):
            np.random.seed(i)   # -- so the results are repeatable.
            id1, id2 = np.random.choice(len(line_eqn), size=2, replace=False)
            line1, line2 = line_eqn[id1], line_eqn[id2]
            polaris_x, polaris_y, fit_numer = cls._find_point_from_lines(line1, line2, line_eqn, thresh_hold)
            if fit_numer > best_fit_number:
                best_fit_number = fit_numer
                best_polaris = [polaris_x, polaris_y]
        return best_polaris, best_fit_number

    @classmethod
    def _get_line_from_points(cls, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        a = x2 - x1
        b = y2 - y1
        c = -((x1 + x2) / 2 * (x2 - x1) + (y1 + y2) / 2 * (y2 - y1))
        return a, b, c

    @classmethod
    def _find_point_from_lines(cls, line1, line2, line_eqn, thresh_hold):
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        y = (a1 / a2 - 1) * c1 / (b1 - a1 * b2 / a2)
        x = (-b1 * y - c1) / a1
        fit_number = cls._fit_check(x, y, line_eqn, thresh_hold)
        return x, y, fit_number

    @classmethod
    def _fit_check(cls, x, y, line_eqn, thresh_hold):
        fit_number = 0
        for a, b, c in line_eqn:
            dist = abs(a*x + b*y + c) / ((a**2 + b**2)**0.5)
            if dist < thresh_hold:
                fit_number += 1
        return fit_number

    @classmethod
    def _vis_points_pair(cls, points1, points2, h, w):
        canvas = np.zeros([h, w, 3], np.uint8)
        for y, x in points1:
            canvas[y, x, 0] = True
        for y, x in points2:
            canvas[y, x, 1] = True
        canvas[:, :, 0] = ndimage.morphology.binary_dilation(canvas[:, :, 0], iterations=3)
        canvas[:, :, 1] = ndimage.morphology.binary_dilation(canvas[:, :, 1], iterations=3)
        canvas = canvas*255
        return canvas


class Purifier:
    @classmethod
    def enhancement(cls, img_stack):
        std_map = np.std(img_stack, axis=0)
        min_map = np.min(img_stack, axis=0)
        max_map = np.max(img_stack, axis=0)
        img1 = img_stack[0]
        plt.figure()
        plt.imshow(np.log((img1 - min_map)+256))
        plt.show()


class PatchRegistration:
    @classmethod
    def patch_register(cls, img1, img2, patch_size, step_size=None):
        if step_size is None:
            step_size = patch_size
        H, W = img1.shape
        shift_map = []
        for i in range(0, H-patch_size, step_size):
            for j in range(0, W-patch_size, step_size):
                patch_1 = img1[i:i+patch_size, j:j+patch_size]
                patch_2 = img2[j:j+patch_size, j:j+patch_size]
                cross_corr = np.fft.ifft2(np.fft.fft2(patch_1)*np.conj(np.fft.fft2(patch_2)))
                max_idx = np.argmax(cross_corr)
                idx_i, idx_j = np.unravel_index(max_idx, [patch_size, patch_size])
                shift_map.append([idx_i, idx_j])

        return shift_map




















