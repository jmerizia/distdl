import numpy as np

from distdl.utilities.slicing import compute_start_index
from distdl.utilities.slicing import compute_subshape


class HaloMixin:

    def _compute_exchange_info(self,
                               halo_shape,
                               neighbor_halo_shapes):

        recv_buffer_shape = halo_shape.copy()
        send_buffer_shape = np.zeros_like(halo_shape, dtype=int)

        dims = len(halo_shape)
        for dim in range(dims):
            left_halo_shape, right_halo_shape = neighbor_halo_shapes[dim]

            if left_halo_shape is not None:
                send_buffer_shape[dim, 0] = left_halo_shape[dim, 1]

            if right_halo_shape is not None:
                send_buffer_shape[dim, 1] = right_halo_shape[dim, 0]

        return recv_buffer_shape, send_buffer_shape

    def _left_pad_to_length(self, array, length, value):
        lpad = length - len(array)
        return np.pad(array,
                      pad_width=(lpad, 0),
                      mode='constant',
                      constant_values=value)

    def _compute_local_start_stop_indices(self, subtensor_shapes_unbalanced, x_local_shape):
        # this could use a refactor at some point
        x_local_start_index = np.zeros_like(x_local_shape, dtype=np.int)
        dims = len(x_local_shape)
        for dim in range(2, dims):
            for i in range(self.P_x.index[dim]):
                idx = tuple(i if j == dim else 0 for j in range(dims))
                x_local_start_index[dim] += subtensor_shapes_unbalanced[idx][dim]
        x_local_stop_index = x_local_start_index + x_local_shape - 1
        return x_local_start_index, x_local_stop_index

    def _compute_needed_ranges(self, tensor_shape, halo_shape):

        ranges = np.zeros_like(halo_shape)

        # If we have a negative halo on the left, we want to not pass that
        # data to the torch layer
        ranges[:, 0] = -1*np.minimum(0, halo_shape[:, 0])

        # The stop of the slice will be the data + the length of the two halos
        # and the last maximum is so that we dont shorten the stop (keeps the
        # parallel and sequential behavior exactly the same, but I dont think
        # it is strictly necessary)
        ranges[:, 1] = tensor_shape[:] + np.maximum(0, halo_shape[:, 0]) + np.maximum(0, halo_shape[:, 1])

        return ranges

    def _compute_out_shape(self, in_shape, kernel_size, stride, padding, dilation):
        # formula from pytorch docs for maxpool
        return np.floor((in_shape
                         + 2*padding
                         - dilation*(kernel_size-1) - 1)/stride + 1).astype(in_shape.dtype)

    def _compute_halo_shape(self,
                            partition_shape,
                            partition_index,
                            x_global_shape,
                            x_local_start_index,
                            x_local_stop_index,
                            kernel_size,
                            stride,
                            padding,
                            dilation):

        x_global_shape = np.asarray(x_global_shape, dtype=np.int)

        y_global_shape = self._compute_out_shape(x_global_shape, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, dilation=dilation)

        # Since the output should be balanced, we may use standard slicing functions.
        y_local_shape = compute_subshape(partition_shape, partition_index, y_global_shape)
        y_local_start_index = compute_start_index(partition_shape, partition_index, y_global_shape)

        y_local_left_global_index = y_local_start_index
        x_local_left_global_index_needed = self._compute_min_input_range(y_local_left_global_index,
                                                                         kernel_size, stride, padding, dilation)

        y_local_right_global_index = y_local_start_index + y_local_shape - 1
        x_local_right_global_index_needed = self._compute_max_input_range(y_local_right_global_index,
                                                                          kernel_size, stride, padding, dilation)

        # Compute the actual ghost values
        x_local_left_halo_shape = x_local_start_index - x_local_left_global_index_needed
        x_local_right_halo_shape = x_local_right_global_index_needed - x_local_stop_index

        return np.hstack([x_local_left_halo_shape, x_local_right_halo_shape]).reshape(2, -1).T
