import cv2
import numpy as np
from scipy.optimize import differential_evolution
from skimage.metrics import mean_squared_error
import sys

def read_image(path):
    """Reads an image from a file and converts it to RGB format with float32 precision."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return img

def objective_function(params, source_img, target_img):
    """
    Objective function for optimization.
    Applies an affine transformation (scaling and translation only) to the source image
    and computes the MSE with the target image.
    Assumes padding color is always white (255, 255, 255).
    """
    # Extract affine parameters from params
    s, t_x, t_y = params
    h_target, w_target = target_img.shape[:2]
    t_x *= w_target
    t_y *= h_target

    # Set padding color to white
    padding_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White in [0,1] range

    # Build the affine transformation matrix (no shear or rotation)
    M = np.array([[s, 0, t_x],
                  [0, s, t_y]], dtype=np.float32)

    # Apply the affine transformation to the source image
    transformed_img = cv2.warpAffine(source_img, M, (w_target, h_target),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=padding_color.tolist())

    # Compute the Mean Squared Error between the transformed source image and the target image
    mse = mean_squared_error(target_img, transformed_img)

    return mse

def align_images(source_img, target_img):
    """
    Uses differential evolution to find the affine transformation (scaling and translation)
    that minimizes the MSE between the transformed source image and the target image.
    Assumes padding color is always white (255, 255, 255).
    Returns the transformed image and the optimization result.
    """

    # initially rescale source image for optimization
    ratio = source_img.shape[0] / target_img.shape[1]
    source_img = cv2.resize(source_img, (round(source_img.shape[1] / ratio), round(source_img.shape[0] / ratio)))

    # Get dimensions of the images
    h_source, w_source = source_img.shape[:2]
    h_target, w_target = target_img.shape[:2]

    # Set bounds for the affine transformation parameters
    bounds = [
        (0.5, 2.0),    # s (scaling in x and y)
        (-0.5, 0.5),   # t_x (translation in x)
        (-0.5, 0.5),   # t_y (translation in y)
    ]

    print([1.0, (w_target - w_source) / 2, 0])
    # Perform the optimization
    result = differential_evolution(
        objective_function,
        bounds,
        x0=[1.0, 0.25, 0],
        args=(source_img, target_img),
        strategy='best1bin',
        maxiter=100,
        popsize=4,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=True
    )

    # Extract the optimal parameters
    s, t_x, t_y = result.x
    t_x *= w_target
    t_y *= h_target

    # Build the affine transformation matrix with the optimal parameters
    M = np.array([[s, 0, t_x],
                  [0, s, t_y]], dtype=np.float32)

    # Set padding color to white
    padding_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White in [0,1] range

    # Apply the optimal affine transformation
    transformed_img = cv2.warpAffine(source_img, M, (w_target, h_target),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=padding_color.tolist())

    return transformed_img, result

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py source_image target_image output_image")
        sys.exit(1)
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    output_path = sys.argv[3]

    # Read the images
    source_img = read_image(source_path)
    target_img = read_image(target_path)

    # Perform the alignment
    transformed_img, optimization_result = align_images(source_img, target_img)

    print("Optimization result:")
    print(optimization_result)

    # Save the transformed image
    transformed_img_uint8 = (transformed_img * 255.0).astype(np.uint8)
    transformed_img_bgr = cv2.cvtColor(transformed_img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, transformed_img_bgr)
    print(f"Transformed image saved as {output_path}")

