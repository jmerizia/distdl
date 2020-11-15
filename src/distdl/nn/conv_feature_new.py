import numpy as np
import torch
import torch.nn.functional as F

from distdl.backends.mpi.tensor_decomposition import compute_subtensor_shapes_unbalanced
from distdl.nn.broadcast import Broadcast
from distdl.nn.halo_exchange import HaloExchange
from distdl.nn.mixins.conv_mixin import ConvMixin
from distdl.nn.mixins.halo_mixin_new import HaloMixin
from distdl.nn.module import Module
from distdl.nn.pad import DistributedPad
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.torch import TensorStructure
from distdl.utilities.torch import to_torch_pad
from distdl.utilities.torch import zero_volume_tensor


class DistributedFeatureConvBase(Module, HaloMixin, ConvMixin):
    r"""A feature-space partitioned distributed convolutional layer.

    This class provides the user interface to a distributed convolutional
    layer, where the input (and output) tensors are partitioned in
    feature-space only.

    The base unit of work is given by the input/output tensor partition.  This
    class requires the following of the tensor partitions:

    1. :math:`P_x` over input tensor :math:`x` has shape :math:`1 \times
       P_{\text{c_in}} \times 1 \times \dots \times 1`.

    The output partition, :math:`P_y`, is assumed to be the same as the
    input partition.

    The first dimension of the input/output partitions is the batch
    dimension,the second is the channel dimension, and remaining dimensions
    are feature dimensions.

    The learnable weight and bias terms does not have their own partition.
    They is stored at the 0th rank (index :math:`(0, 0,\dots, 0)` of
    :math:`P_x`.  Each worker in :math:`P_x` does have their own local
    convolutional layer but only one worker has learnable coefficients.

    All inputs to this base class are passed through to the underlying PyTorch
    convolutional layer.

    Parameters
    ----------
    P_x :
        Partition of input tensor.
    in_channels :
        (int)
        Number of channels in the input image
    out_channels :
        (int)
        Number of channels produced by the convolution
    kernel_size :
        (int or tuple)
        Size of the convolving kernel.
    stride :
        (int or tuple, optional)
        Stride of the convolution. Default: 1
    padding :
        (int or tuple, optional)
        Zero-padding added to both sides of the input. Default: 0
    padding_mode :
        (int or tuple, optional)
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        .. warning::
            Currently, only padding_mode = 'zeros' is supported.
    dilation :
        (int or tuple, optional)
        Spacing between kernel elements. Default: 1
    groups :
        (int, optional)
        Number of blocked connections from input channels to output channels. Default: 1
    bias :
        (bool, optional)
        If True, adds a learnable bias to the output. Default: True
    buffer_manager :
        (BufferManager, optional)
        Optional DistDL buffer manager. Default: None
    """

    # Convolution class for base unit of work.
    TorchConvType = None

    def __init__(self,
                 P_x,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 groups=1,
                 bias=True,
                 buffer_manager=None):

        super(DistributedFeatureConvBase, self).__init__()

        # P_x is 1 x 1 x P_d-1 x ... x P_0
        self.P_x = P_x

        if padding_mode != 'zeros':
            raise ValueError('Only padding_mode = \'zeros\' is supported.')

        # Back-end specific buffer manager for economic buffer allocation
        if buffer_manager is None:
            buffer_manager = self._distdl_backend.BufferManager()
        elif type(buffer_manager) is not self._distdl_backend.BufferManager:
            raise ValueError("Buffer manager type does not match backend.")
        self.buffer_manager = buffer_manager

        if not self.P_x.active:
            return

        dims = len(P_x.shape)

        # Do this before checking serial so that the layer works properly
        # in the serial case
        self.conv_layer = self.TorchConvType(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=0,
                                             dilation=dilation,
                                             groups=groups,
                                             bias=bias)

        # Expand the given parameters to the proper shapes
        kernel_size = np.atleast_1d(kernel_size)
        kernel_size = self._left_pad_to_length(kernel_size, dims, 1)
        stride = np.atleast_1d(stride)
        stride = self._left_pad_to_length(stride, dims, 1)
        # For now, only support symmetric padding
        padding = np.atleast_1d(padding)
        padding = self._left_pad_to_length(padding, dims, 0)
        dilation = np.atleast_1d(dilation)
        dilation = self._left_pad_to_length(dilation, dims, 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.serial = False
        if self.P_x.size == 1:
            self.serial = True
            return

        # Weights and biases partition
        P_wb = self.P_x.create_partition_inclusive([0])
        self.P_wb_cart = P_wb.create_cartesian_topology_partition([1])

        # Release temporary resources
        P_wb.deactivate()

        # We want only the root rank of the broadcast to have a weight and a
        # bias parameter. Every other rank gets a zero-volume tensor.
        if self.P_wb_cart.active:
            self.weight = torch.nn.Parameter(self.conv_layer.weight.detach())

            if self.conv_layer.bias is not None:
                self.bias = torch.nn.Parameter(self.conv_layer.bias.detach())
        else:
            self.weight = zero_volume_tensor()

            if self.conv_layer.bias is not None:
                self.bias = zero_volume_tensor()

        self.weight.requires_grad = self.conv_layer.weight.requires_grad

        if self.conv_layer.bias is not None:
            self.bias.requires_grad = self.conv_layer.bias.requires_grad

        # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677/2
        new_weight = self.conv_layer.weight.detach() * 0
        new_weight.requires_grad = self.conv_layer.weight.requires_grad
        del self.conv_layer.weight
        self.conv_layer.weight = new_weight

        if self.conv_layer.bias is not None:
            new_bias = self.conv_layer.bias.detach() * 0
            new_bias.requires_grad = self.conv_layer.bias.requires_grad
            del self.conv_layer.bias
            self.conv_layer.bias = new_bias

        self.w_broadcast = Broadcast(self.P_wb_cart, self.P_x,
                                     preserve_batch=False)

        if self.conv_layer.bias is not None:
            self.b_broadcast = Broadcast(self.P_wb_cart, self.P_x,
                                         preserve_batch=False)

        # We need to pad the input ourselves, since ConvNd can only pad symmetrically.
        # Since global dimensions are needed, construction defered to pre-forward hook.
        self.pad_layer = None

        # We need the halo shape, and other info, to fully populate the pad,
        # halo exchange, and unpad layers.  For pad and unpad, we defer their
        # construction to the pre-forward hook.
        self.torch_halo_pad = None

        # We need to be able to remove some data from the input to the conv
        # layer.
        self.needed_slices = None

        # For the halo layer we also defer construction, so that we can have
        # the halo shape for the input.  The halo will allocate its own
        # buffers, but it needs this information at construction to be able
        # to do this in the pre-forward hook.
        self.halo_layer = None

        # Variables for tracking input changes and buffer construction
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_module_setup(self, input):
        r"""Distributed (feature) convolution module setup function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

        if not self.P_x.active:
            return

        if self.serial:
            return

        dims = len(self.padding)
        pad_left_right = self.padding.reshape((dims, 1)) + np.zeros((dims, 2), dtype=np.int)
        self.pad_layer = DistributedPad(self.P_x, pad_left_right, mode='constant', value=0)

        # Compute the global and local input shapes after padding
        x_global_shape = self._distdl_backend.assemble_global_tensor_structure(input[0], self.P_x).shape
        x_local_shape = np.array(input[0].shape)
        x_global_shape_after_pad = x_global_shape + 2*self.padding
        x_local_shape_after_pad = x_local_shape + np.sum(self.pad_layer.local_pad, axis=1, keepdims=False)

        # Compute left and right start and stop indices for the local input shape
        subtensor_shapes_unbalanced = \
            compute_subtensor_shapes_unbalanced(TensorStructure(shape=x_local_shape_after_pad), self.P_x)
        x_local_start_index, x_local_stop_index = \
            self._compute_local_start_stop_indices(subtensor_shapes_unbalanced, x_local_shape_after_pad)

        # Compute the local halo region
        # Note: Since we assume the padding is already added to the input, we need not add it here.
        halo_shape_with_negative = self._compute_halo_shape(partition_shape=self.P_x.shape,
                                                            partition_index=self.P_x.index,
                                                            x_global_shape=x_global_shape_after_pad,
                                                            x_local_start_index=x_local_start_index,
                                                            x_local_stop_index=x_local_stop_index,
                                                            kernel_size=self.kernel_size,
                                                            stride=self.stride,
                                                            padding=np.zeros_like(self.padding),
                                                            dilation=self.dilation)
        halo_shape = np.maximum(halo_shape_with_negative, 0)

        # The input to PyTorch's functional pad layer is different from NumPy's, so transform it.
        self.torch_halo_pad = to_torch_pad(halo_shape)

        # Determine the halo shapes of the neighboring ranks
        neighbor_halo_shapes = self.P_x.broadcast_to_neighbors(halo_shape)

        # We can now compute the info required for the halo layer.
        recv_buffer_shape, send_buffer_shape = self._compute_exchange_info(halo_shape, neighbor_halo_shapes)

        # We can also set up part of the halo layer.
        self.halo_layer = HaloExchange(self.P_x,
                                       halo_shape,
                                       recv_buffer_shape,
                                       send_buffer_shape,
                                       buffer_manager=self.buffer_manager)

        # We have to select out the "unused" entries.  Sometimes there can
        # be "negative" halos.
        needed_ranges = self._compute_needed_ranges(x_local_shape_after_pad, halo_shape_with_negative)
        self.needed_slices = assemble_slices(needed_ranges[:, 0],
                                             needed_ranges[:, 1])

    def _distdl_module_teardown(self, input):
        r"""Distributed (channel) convolution module teardown function.

        This function is called every time something changes in the input
        tensor structure.  It should not be called manually.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        # Reset all sub_layers
        self.pad_layer = None
        self.torch_halo_pad = None
        self.needed_slices = None
        self.halo_layer = None

        # Reset any info about the input
        self._distdl_is_setup = False
        self._input_tensor_structure = TensorStructure()

    def _distdl_input_changed(self, input):
        r"""Determine if the structure of inputs has changed.

        Parameters
        ----------
        input :
            Tuple of forward inputs.  See
            `torch.nn.Module.register_forward_pre_hook` for more details.

        """

        new_tensor_structure = TensorStructure(input[0])

        return self._input_tensor_structure != new_tensor_structure

    def forward(self, input):
        r"""Forward function interface.

        Parameters
        ----------
        input :
            Input tensor to be broadcast.

        """

        if not self.P_x.active:
            return input.clone()

        if self.serial:
            return self.conv_layer(input)

        w = self.w_broadcast(self.weight)
        self.conv_layer.weight = w

        if self.conv_layer.bias is not None:
            b = self.b_broadcast(self.bias)
            self.conv_layer.bias = b

        input_padded = self.pad_layer(input)
        input_padded_with_halo = F.pad(input_padded, pad=self.torch_halo_pad, mode='constant', value=0)
        input_exchanged = self.halo_layer(input_padded_with_halo)
        input_needed = input_exchanged[self.needed_slices]
        conv_output = self.conv_layer(input_needed)
        return conv_output


class DistributedFeatureConv1d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 1d convolutional layer.

    """

    TorchConvType = torch.nn.Conv1d


class DistributedFeatureConv2d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 2d convolutional layer.

    """

    TorchConvType = torch.nn.Conv2d


class DistributedFeatureConv3d(DistributedFeatureConvBase):
    r"""A feature-partitioned distributed 3d convolutional layer.

    """

    TorchConvType = torch.nn.Conv3d
