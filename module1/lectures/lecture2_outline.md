Concepts refresher
-------------------

1. Recap of likelihood, MLE, NLL, expected NLL as average NLL
2. Cross entropy, cross entropy loss and equivalence of NLL and cross entropy loss
3. KL divergence and relation between cross entropy and KL divergence
4. Convolution operation, kernels, how convolution works, stride, padding
5. How to calculate output dimensions of a convolution operation
6. Valid vs same convolution
7. How convolution learns feature maps through filters. How kernel is 2d, filter is 3d, parameter count for a single filter.
    Total parameter count for a convolution using number of output channels.
8. Dilation and how it increases the kerner size without increasing the parameter count. How it increases the receptive field and is used in segmentation tasks
9. Downsampling and its importance. Techniques for downsampling (strided convolution, pooling)
10. Pooling operations, types of pooling. Output dimensions after a pooling operation.
11. 1x1 convolution and how it helps with dimensionality reduction and learning new channel combinations
12. Depthwise separable convolutions and how it reduces computation and parameter count in edge devices.
13. Upsampling operation using transposed convolution
14. Upsampling using bilinear interpolation followed by convolution.


Encoder decoder architecture
-----------------------------
1. Basic idea. What encoder and decoder are, how encoder compresses input to latent represenation and how decoder uses that latent representation to construct the output
2. Use cases of encoder decoder architecture. How image reconstruction using standard autoencoder can be used for defect detection
3. How convolution encoder works by downsampling
4. Why downsampling is needed
5. A simple convolution encoder implementation using latent vector
6. Types of latent representation
7. Latent vector
8. Spatial latent representation and spatial encoder implementation
9. When spatial latent representation is needed
10. Bottleneck size and its consequences.
11. Upsampling with bilinear interpolation and transposed convolution
12. Need for upsampling.
13. A convolution decoder implementation using transposed convolution
14. Complete image encoder decoder implementation
15. Skip connections and how they help with vanishing gradients and learning in deeper layers
16. Where encoder decoder architecture is used


Semantic segmentation (UNet.ioynb)
----------------------
1. What semantic segmentation is. Where is it used.
2. UNet architecture introduction and how it uses encoder-decoder.
3. UNet architecture details, the encoder block, use of convolutional and max-pooling layers.
5. How convolutional and max-pooling layers increase the receptive field.
6. The bottleneck layer
7. The decoder block and use of transposed convolutions for upsampling
9. Use of skip connections from encoder block to decoder block for passing fine-grained features along with abstract high level contextual features.
10. The last convolutional layer outputs number of channels equal to the number of prediction classes.


Autoencoder basics
---------------------