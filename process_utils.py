import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, fftn, ifftn
from skimage.feature import peak_local_max
from scipy import ndimage
import heapq

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


















