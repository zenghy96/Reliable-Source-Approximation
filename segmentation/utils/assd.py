import numpy as np
from scipy import ndimage
import math


def compute_average_surface_distance(surface_distances):
    """Returns the average surface distance.

    Computes the average surface distances by correctly taking the area of each
    surface element into account. Call compute_surface_distances(...) before, to
    obtain the `surface_distances` dict.

    Args:
        surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
        "surfel_areas_gt", "surfel_areas_pred" created by
        compute_surface_distances()

    Returns:
        A tuple with two float values:
        - the average distance (in mm) from the ground truth surface to the
            predicted surface
        - the average distance from the predicted surface to the ground truth
            surface.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = (
        np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt))
    average_distance_pred_to_gt = (
        np.sum(distances_pred_to_gt * surfel_areas_pred) /
        np.sum(surfel_areas_pred))
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_surface_distances(mask_gt,
                              mask_pred,
                              spacing_mm):
    """Computes closest distances from all surface points to the other surface.

    This function can be applied to 2D or 3D tensors. For 2D, both masks must be
    2D and `spacing_mm` must be a 2-element list. For 3D, both masks must be 3D
    and `spacing_mm` must be a 3-element list. The description is done for the 2D
    case, and the formulation for the 3D case is present is parenthesis,
    introduced by "resp.".

    Finds all contour elements (resp surface elements "surfels" in 3D) in the
    ground truth mask `mask_gt` and the predicted mask `mask_pred`, computes their
    length in mm (resp. area in mm^2) and the distance to the closest point on the
    other contour (resp. surface). It returns two sorted lists of distances
    together with the corresponding contour lengths (resp. surfel areas). If one
    of the masks is empty, the corresponding lists are empty and all distances in
    the other list are `inf`.

    Args:
        mask_gt: 2-dim (resp. 3-dim) bool Numpy array. The ground truth mask.
        mask_pred: 2-dim (resp. 3-dim) bool Numpy array. The predicted mask.
        spacing_mm: 2-element (resp. 3-element) list-like structure. Voxel spacing
        in x0 anx x1 (resp. x0, x1 and x2) directions.

    Returns:
        A dict with:
        "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
            from all ground truth surface elements to the predicted surface,
            sorted from smallest to largest.
        "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
            from all predicted surface elements to the ground truth surface,
            sorted from smallest to largest.
        "surfel_areas_gt": 1-dim numpy array of type float. The length of the
        of the ground truth contours in mm (resp. the surface elements area in
        mm^2) in the same order as distances_gt_to_pred.
        "surfel_areas_pred": 1-dim numpy array of type float. The length of the
        of the predicted contours in mm (resp. the surface elements area in
        mm^2) in the same order as distances_gt_to_pred.

    Raises:
        ValueError: If the masks and the `spacing_mm` arguments are of incompatible
        shape or type. Or if the masks are not 2D or 3D.
    """
    # The terms used in this function are for the 3D case. In particular, surface
    # in 2D stands for contours in 3D. The surface elements in 3D correspond to
    # the line elements in 2D.

    _assert_is_bool_numpy_array("mask_gt", mask_gt)
    _assert_is_bool_numpy_array("mask_pred", mask_pred)
    ENCODE_NEIGHBOURHOOD_2D_KERNEL = np.array([[8, 4], [2, 1]])

    if not len(mask_gt.shape) == len(mask_pred.shape) == len(spacing_mm):
        raise ValueError("The arguments must be of compatible shape. Got mask_gt "
                        "with {} dimensions ({}) and mask_pred with {} dimensions "
                        "({}), while the spacing_mm was {} elements.".format(
                            len(mask_gt.shape),
                            mask_gt.shape, len(mask_pred.shape), mask_pred.shape,
                         len(spacing_mm)))

    num_dims = len(spacing_mm)
    if num_dims == 2:
        # compute the area for all 16 possible surface elements
        # (given a 2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            create_table_neighbour_code_to_contour_length(spacing_mm))
        kernel = ENCODE_NEIGHBOURHOOD_2D_KERNEL
        full_true_neighbours = 0b1111

    # compute the bounding box of the masks to trim the volume to the smallest
    # possible processing subvolume
    bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    # Both the min/max bbox are None at the same time, so we only check one.
    if bbox_min is None:
        return {
            "distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([]),
        }

    # crop the processing subvolume.
    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    # compute the neighbour code (local binary pattern) for each voxel
    # the resulting arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) &
                    (neighbour_code_map_gt != full_true_neighbours))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != full_true_neighbours))

    # compute the distance transform (closest distance of each voxel to the
    # surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[
        neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        distances_gt_to_pred, surfel_areas_gt = _sort_distances_surfels(
            distances_gt_to_pred, surfel_areas_gt)

    if distances_pred_to_gt.shape != (0,):
        distances_pred_to_gt, surfel_areas_pred = _sort_distances_surfels(
            distances_pred_to_gt, surfel_areas_pred)

    return {
        "distances_gt_to_pred": distances_gt_to_pred,
        "distances_pred_to_gt": distances_pred_to_gt,
        "surfel_areas_gt": surfel_areas_gt,
        "surfel_areas_pred": surfel_areas_pred,
    }


def _compute_bounding_box(mask):
    """Computes the bounding box of the masks.

    This function generalizes to arbitrary number of dimensions great or equal
    to 1.

    Args:
        mask: The 2D or 3D numpy mask, where '0' means background and non-zero means
        foreground.

    Returns:
        A tuple:
        - The coordinates of the first point of the bounding box (smallest on all
        axes), or `None` if the mask contains only zeros.
        - The coordinates of the second point of the bounding box (greatest on all
        axes), or `None` if the mask contains only zeros.
    """
    num_dims = len(mask.shape)
    bbox_min = np.zeros(num_dims, np.int64)
    bbox_max = np.zeros(num_dims, np.int64)

    # max projection to the x0-axis
    proj_0 = np.amax(mask, axis=tuple(range(num_dims))[1:])
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
        return None, None

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the i-th-axis for i in {1, ..., num_dims - 1}
    for axis in range(1, num_dims):
        max_over_axes = list(range(num_dims))  # Python 3 compatible
        max_over_axes.pop(axis)  # Remove the i-th dimension from the max
        max_over_axes = tuple(max_over_axes)  # numpy expects a tuple of ints
        proj = np.amax(mask, axis=max_over_axes)
        idx_nonzero = np.nonzero(proj)[0]
        bbox_min[axis] = np.min(idx_nonzero)
        bbox_max[axis] = np.max(idx_nonzero)

    return bbox_min, bbox_max


def _sort_distances_surfels(distances, surfel_areas):
    """Sorts the two list with respect to the tuple of (distance, surfel_area).

    Args:
        distances: The distances from A to B (e.g. `distances_gt_to_pred`).
        surfel_areas: The surfel areas for A (e.g. `surfel_areas_gt`).

    Returns:
        A tuple of the sorted (distances, surfel_areas).
    """
    sorted_surfels = np.array(sorted(zip(distances, surfel_areas)))
    return sorted_surfels[:, 0], sorted_surfels[:, 1]


def _crop_to_bounding_box(mask, bbox_min, bbox_max):
    """Crops a 2D or 3D mask to the bounding box specified by `bbox_{min,max}`."""
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right (and the back on 3D) sides. This is required to obtain the
    # "full" convolution result with the 2x2 (or 2x2x2 in 3D) kernel.
    # TODO:  This is correct only if the object is interior to the
    # bounding box.
    cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

    num_dims = len(mask.shape)
    # pyformat: disable
    if num_dims == 2:
        cropmask[0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                    bbox_min[1]:bbox_max[1] + 1]
    elif num_dims == 3:
        cropmask[0:-1, 0:-1, 0:-1] = mask[bbox_min[0]:bbox_max[0] + 1,
                                        bbox_min[1]:bbox_max[1] + 1,
                                        bbox_min[2]:bbox_max[2] + 1]
    # pyformat: enable
    else:
        assert False

    return cropmask


def create_table_neighbour_code_to_contour_length(spacing_mm):
    """Returns an array mapping neighbourhood code to the contour length.

    For the list of possible cases and their figures, see page 38 from:
    https://nccastaff.bournemouth.ac.uk/jmacey/MastersProjects/MSc14/06/thesis.pdf

    In 2D, each point has 4 neighbors. Thus, are 16 configurations. A
    configuration is encoded with '1' meaning "inside the object" and '0' "outside
    the object". The points are ordered: top left, top right, bottom left, bottom
    right.

    The x0 axis is assumed vertical downward, and the x1 axis is horizontal to the
    right:
    (0, 0) --> (0, 1)
        |
    (1, 0)

    Args:
        spacing_mm: 2-element list-like structure. Voxel spacing in x0 and x1
        directions.
    """
    neighbour_code_to_contour_length = np.zeros([16])

    vertical = spacing_mm[0]
    horizontal = spacing_mm[1]
    diag = 0.5 * math.sqrt(spacing_mm[0]**2 + spacing_mm[1]**2)
    # pyformat: disable
    neighbour_code_to_contour_length[int("00"
                                        "01", 2)] = diag

    neighbour_code_to_contour_length[int("00"
                                        "10", 2)] = diag

    neighbour_code_to_contour_length[int("00"
                                        "11", 2)] = horizontal

    neighbour_code_to_contour_length[int("01"
                                        "00", 2)] = diag

    neighbour_code_to_contour_length[int("01"
                                        "01", 2)] = vertical

    neighbour_code_to_contour_length[int("01"
                                        "10", 2)] = 2*diag

    neighbour_code_to_contour_length[int("01"
                                        "11", 2)] = diag

    neighbour_code_to_contour_length[int("10"
                                        "00", 2)] = diag

    neighbour_code_to_contour_length[int("10"
                                        "01", 2)] = 2*diag

    neighbour_code_to_contour_length[int("10"
                                        "10", 2)] = vertical

    neighbour_code_to_contour_length[int("10"
                                        "11", 2)] = diag

    neighbour_code_to_contour_length[int("11"
                                        "00", 2)] = horizontal

    neighbour_code_to_contour_length[int("11"
                                        "01", 2)] = diag

    neighbour_code_to_contour_length[int("11"
                                        "10", 2)] = diag
    # pyformat: enable

    return neighbour_code_to_contour_length


def _assert_is_bool_numpy_array(name, array):
    _assert_is_numpy_array(name, array)
    if array.dtype != bool:
        raise ValueError("The argument {!r} should be a numpy array of type bool, "
                        "not {}".format(name, array.dtype))
  
def _assert_is_numpy_array(name, array):
    """Raises an exception if `array` is not a numpy array."""
    if not isinstance(array, np.ndarray):
        raise ValueError("The argument {!r} should be a numpy array, not a "
                        "{}".format(name, type(array)))


def cal_assd(pred, gt, spacing=(0.6, 0,6)):
    if pred.max() == 0:
        assd = 10
    else:
        surface_distances = compute_surface_distances(gt>0, pred>0, spacing_mm=spacing)
        average_distance_gt_to_pred, average_distance_pred_to_gt = compute_average_surface_distance(surface_distances)
        assd = (average_distance_gt_to_pred + average_distance_pred_to_gt) / 2
    return assd
