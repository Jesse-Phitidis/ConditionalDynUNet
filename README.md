# ConditionalDynUNet
DynUNet from MONAI, with extra input channels in the decoder for conditioning.

## Purpose
It may be desirable to add additional inputs to feature maps in the decoder to condition the model. For example, a known probability map of target lesion location.

## Useage
The class to use is `ConditionalDynUNet` and it contains one extra initialisation argument `conditional_channels`. This should be a list of integers of the same length as `kernel_size` and each entry represents the number of additional input channels to that layer. The first entry is for the bottleneck and the last entry is for the the final layer before the out block. All entries must be the same integer, or zero, since varying number of conditional channels for different layers is not supported. For example, to add a single conditioning channel to the bottleneck and the last layer of a `ConditionalDynUNet` with `kernel_size=[3,3,3,3,3,3]` we would set `conditional_channels=[1,0,0,0,0,1]`.

If `conditional_channels` is not all zeros, then the `forward` method expects an additional input containing the conditioning information. This should be a tensor of shape `(B,C,...)` where `C` is the number of conditional channels and `...` is the same number of spatial dimensions as the input, e.g. 3 for a 3D volume. The spatial shape is interpolated to the correct size for the layer.

If `emb_layers` is non-zero, then this number of additional convolutional layers (conv - norm - act) are used to embed the conditional channels into `emb_dim` channels (via `hidden_dim`).
