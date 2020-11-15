import numpy as np

from distdl.utilities.slicing import compute_start_index
from distdl.utilities.slicing import compute_subshape


class HaloMixin:

    def _compute_exchange_info(self,
                               x_global_shape,
                               x_local_shape,
                               neighbor_x_local_shapes,
                               x_local_start_index,
                               x_local_stop_index,
                               neighbor_x_local_start_indices,
                               neighbor_x_local_stop_indices,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               partition_active,
                               partition_shape,
                               partition_index):

        if not partition_active:
            return None, None, None, None

        dims = len(partition_shape)

        halo_shape = self._compute_halo_shape(partition_shape,
                                              partition_index,
                                              x_global_shape,
                                              x_local_shape,
                                              x_local_start_index,
                                              x_local_stop_index,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation)

        # print(self.P_x.rank, 'halo shape', halo_shape)
        # assert False

        recv_buffer_shape = halo_shape.copy()

        send_buffer_shape = np.zeros_like(halo_shape, dtype=int)

        # print(self.P_x.rank)
        for dim in range(dims):
            left_x_local_shape, right_x_local_shape = neighbor_x_local_shapes[dim]
            left_x_local_start_index, right_x_local_start_index = neighbor_x_local_start_indices[dim]
            left_x_local_stop_index, right_x_local_stop_index = neighbor_x_local_stop_indices[dim]

            # If I have a left neighbor, my left send buffer size is my left
            # neighbor's right halo size
            lindex = [x - 1 if i == dim else x for i, x in enumerate(partition_index)]
            if lindex[dim] > -1:
                lpartition_halo = self._compute_halo_shape(partition_shape,
                                                           lindex,
                                                           x_global_shape,
                                                           left_x_local_shape,
                                                           left_x_local_start_index,
                                                           left_x_local_stop_index,
                                                           kernel_size,
                                                           stride,
                                                           padding,
                                                           dilation)
                # print('left, dim', dim, 'shape', left_x_local_shape, 'halo' ,lpartition_halo)
                send_buffer_shape[dim, 0] = lpartition_halo[dim, 1]

            # If I have a right neighbor, my right send buffer size is my right
            # neighbor's left halo size
            rindex = [x + 1 if i == dim else x for i, x in enumerate(partition_index)]
            if rindex[dim] < partition_shape[dim]:
                rpartition_halo = self._compute_halo_shape(partition_shape,
                                                           rindex,
                                                           x_global_shape,
                                                           right_x_local_shape,
                                                           right_x_local_start_index,
                                                           right_x_local_stop_index,
                                                           kernel_size,
                                                           stride,
                                                           padding,
                                                           dilation)
                # print('right, dim', dim, 'shape', right_x_local_shape, 'halo', rpartition_halo)
                send_buffer_shape[dim, 1] = rpartition_halo[dim, 0]

        halo_shape_with_negatives = self._compute_halo_shape(partition_shape,
                                                            partition_index,
                                                            x_global_shape,
                                                            x_local_shape,
                                                            x_local_start_index,
                                                            x_local_start_index,
                                                            kernel_size,
                                                            stride,
                                                            padding,
                                                            dilation,
                                                            require_nonnegative=False)
        needed_ranges = self._compute_needed_ranges(x_local_shape, halo_shape_with_negatives)

        halo_shape = halo_shape.astype(int)
        needed_ranges = needed_ranges.astype(int)
        # print('send', send_buffer_shape)
        # print('recv', recv_buffer_shape)
        # assert False

        return halo_shape, recv_buffer_shape, send_buffer_shape, needed_ranges

    def _left_pad_to_length(self, array, length, value):
        lpad = length - len(array)
        return np.pad(array,
                      pad_width=(lpad, 0),
                      mode='constant',
                      constant_values=value)

    def _add_padding_to_shape(self, shape, padding):
        # return np.array(shape) + np.sum(padding, axis=1, keepdims=False)
        return np.array(shape) + padding

    def _compute_local_start_stop_indices(self, subtensor_shapes_unbalanced, x_local_shape):
        # this could use a refactor at some point
        x_local_start_index = np.zeros_like(x_local_shape, dtype=np.int)
        dims = len(x_local_shape)
        for dim in range(2, dims):
            for i in range(self.P_x.index[dim]):
                idx = tuple(i if j == dim else 0 for j in range(self.P_x.shape[dim]))
                x_local_start_index[dim] += subtensor_shapes_unbalanced[idx][dim]
        x_local_stop_index = x_local_start_index + x_local_shape - 1
        return x_local_start_index, x_local_stop_index

    def _broadcast_2d(self, array):
        length = array.shape[0]
        array = array.reshape((length, 1))
        return array + np.zeros((array.shape[0], 2))

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
                            shape,
                            index,
                            x_global_shape,
                            x_local_shape,
                            x_local_start_index,
                            x_local_stop_index,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            require_nonnegative=True):

        x_global_shape = np.asarray(x_global_shape)

        # Since we assume the padding is already added to the input,
        # we need not add it here.
        y_global_shape = self._compute_out_shape(x_global_shape, kernel_size=kernel_size,
                                                 stride=stride, padding=0, dilation=dilation)

        # Since we expect the output to be balanced,
        # we may use standard slicing functions.
        y_local_shape = compute_subshape(shape, index, y_global_shape)
        y_local_start_index = compute_start_index(shape, index, y_global_shape)

        y_local_left_global_index = y_local_start_index
        x_local_left_global_index_needed = stride * y_local_left_global_index

        y_local_right_global_index = y_local_start_index + y_local_shape - 1
        x_local_right_global_index_needed = stride * y_local_right_global_index + dilation * (kernel_size - 1)

        # Compute the actual ghost values
        x_local_left_halo_shape = x_local_start_index - x_local_left_global_index_needed
        x_local_right_halo_shape = x_local_right_global_index_needed - x_local_stop_index

        # Make sure the halos are always positive, so we get valid buffer shape
        if require_nonnegative:
            x_local_left_halo_shape = np.maximum(x_local_left_halo_shape, 0)
            x_local_right_halo_shape = np.maximum(x_local_right_halo_shape, 0)

        return np.hstack([x_local_left_halo_shape, x_local_right_halo_shape]).reshape(2, -1).T
