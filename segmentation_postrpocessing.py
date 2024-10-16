import numpy as np
import pandas as pd
import cv2
import subprocess
import os
from scipy.ndimage import label as nd_label, binary_fill_holes
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2
from IPython.display import clear_output

# Constants
ID = 12
SE11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
SE12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
EXCEL_PATH = r'...\Excel_files'
EXCEL_FILE = os.path.join(EXCEL_PATH, 'Cells_mammalian_book.xlsx')

# Initialize DataFrame
df = pd.DataFrame(columns=["slice id", "lbp_hist_median", "fourier_transform_max", "fourier_transform_min",
                           "fourier_transform_range", "fourier_transform_variance", "fourier_transform_skewness",
                           "fourier_transform_kurtosis", "fourier_transform_entropy"])
writer = pd.ExcelWriter(EXCEL_FILE, engine='xlsxwriter')

# Placeholder for segmented slices
Z = [(labels == ID).astype('uint8')]
NN = S.shape[0] - 1
ZZ = np.zeros((NN, S.shape[1], S.shape[2], 3), dtype='uint8')

dict_list = []

for idx in range(1, NN):
    # Perform dilation and masking
    msk0 = Z0 = cv2.dilate(Z[idx - 1].astype('uint8'), SE11, iterations=1)
    msk = cv2.dilate(msk0, KERNEL, iterations=1)
    labels2, nlabels2 = nd_label(msk * (S[idx] > 0.6))

    # Find the largest connected component
    maxA = 0
    maxI = 0
    for idx2 in range(1, nlabels2):
        area = ((labels2 == idx2) * Z[idx - 1]).sum()
        if area > maxA:
            maxA = area
            maxI = idx2

    Z0 = (labels2 == maxI).astype('uint8')
    Z0 = cv2.dilate(Z0, SE11, iterations=1)
    Z0 = binary_fill_holes(Z0).astype(int)

    # Ellipse fitting
    binary = np.asarray(Z0, dtype="float32").astype('uint8')
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if contours:
        ellipse = cv2.fitEllipse(contours[0])
        ZZ[idx] = cv2.ellipse(ZZ[idx], ellipse, (255, 0, 0), 1)

    # Morphological operations
    Z1 = cv2.dilate((Z0 > 0).astype('uint8'), SE11, iterations=1)
    Z1 = cv2.erode(Z1, SE11, iterations=2)
    Z1 = cv2.dilate(Z1, SE12, iterations=1)

    # Ellipse and line drawing
    binary = np.asarray(Z1, dtype="float32").astype('uint8')
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if contours:
        ellipse = cv2.fitEllipse(contours[0])
        center = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]

        # Calculate the endpoints of the major axis
        major_axis_length = max(axes)
        half_length = major_axis_length / 2
        angle_radians = np.deg2rad(angle)
        end1 = (int(center[0] + half_length * np.cos(angle_radians)),
                int(center[1] + half_length * np.sin(angle_radians)))
        end2 = (int(center[0] - half_length * np.cos(angle_radians)),
                int(center[1] - half_length * np.sin(angle_radians)))
        
        ZZ[idx] = cv2.ellipse(ZZ[idx], ellipse, (0, 255, 0), 1)
        ZZ[idx] = cv2.line(ZZ[idx], end1, end2, (0, 165, 255), 2)

    # Feature extraction
    masked_region = Z0 * dataset[idx]

    # Extract LBP features
    lbp = local_binary_pattern(masked_region, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)

    # Perform Fourier transformation
    fourier_transform = np.abs(fft2(masked_region))

    # Calculate single-value metrics
    lbp_hist_median = np.median(lbp_hist)
    fourier_transform_max = np.max(fourier_transform)
    fourier_transform_min = np.min(fourier_transform)
    fourier_transform_range = np.ptp(fourier_transform)
    fourier_transform_variance = np.var(fourier_transform)
    fourier_transform_skewness = np.mean((fourier_transform - np.mean(fourier_transform))**3) / np.std(fourier_transform)**3
    fourier_transform_kurtosis = np.mean((fourier_transform - np.mean(fourier_transform))**4) / np.var(fourier_transform)**2
    fourier_transform_entropy = -np.sum(fourier_transform * np.log(fourier_transform))

    # Append results to dictionary
    row_dict = {
        "slice id": idx,
        "lbp_hist_median": np.round(lbp_hist_median, 1),
        "fourier_transform_max": np.round(fourier_transform_max, 1),
        "fourier_transform_min": np.round(fourier_transform_min, 1),
        "fourier_transform_range": np.round(fourier_transform_range, 1),
        "fourier_transform_variance": np.round(fourier_transform_variance, 1),
        "fourier_transform_skewness": np.round(fourier_transform_skewness, 1),
        "fourier_transform_kurtosis": np.round(fourier_transform_kurtosis, 1),
        "fourier_transform_entropy": np.round(fourier_transform_entropy, 1)
    }
    dict_list.append(row_dict)

    # Update DataFrame and write to Excel
    df = pd.DataFrame.from_dict(dict_list)
    sheet_name = f'ROI_{ID}'
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Update slice list
    Z.append(Z0)

    # Clear output and print progress
    clear_output(wait=True)
    print('slice: ', idx, '/', NN - 1)

# Finalize and open Excel file
writer.close()
subprocess.Popen(f'explorer {EXCEL_PATH}')
