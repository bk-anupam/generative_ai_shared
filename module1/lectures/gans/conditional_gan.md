# Conditional GANs: generating what we ask for

## Learning objectives

After this lecture, you should be able to explain why both GAN networks receive a condition, distinguish class labels from adversarial targets, follow a conditional GAN training step, and generate MNIST digits of a chosen class.

Companion notebook: [conditional_gan_mnist.ipynb](conditional_gan_mnist.ipynb). Compare its architecture with [dcgan_mnist.py](dcgan_mnist.py).

## 1. Motivation: from random digits to requested digits

A vanilla GAN maps a random vector $z$ to an image: $\hat{x}=G(z)$. On MNIST, the result could be any digit. There is no explicit input for requesting a seven.

A **conditional GAN (cGAN)** adds a condition $y$:

$$
\hat{x}=G(z,y).
$$

For MNIST, $y$ is a digit class from 0 to 9. The condition specifies digit identity; noise provides variation in handwriting, thickness, and slant. With different noise vectors and the same label, we want different examples of that digit:

```text
G(z1, 7) -> thin, upright seven
G(z2, 7) -> thick, slanted seven
G(z3, 7) -> seven with a crossbar
```

These are intended behaviors, not guarantees. A poorly trained generator may ignore the condition or the noise.

An unconditional GAN models $p(x)$, the overall distribution of images. A conditional GAN models $p(x\mid y)$, the distribution of images given a condition.

## 2. Both networks receive the condition

| Network | Ordinary GAN | Conditional GAN |
|---|---|---|
| Generator | $G(z)$ | $G(z,y)$ |
| Discriminator | $D(x)$ | $D(x,y)$ |

Suppose the requested class is 7 but the generator produces a realistic 3. An ordinary discriminator sees a plausible digit and has no way to know what was requested. A conditional discriminator sees the pair `(image of 3, label 7)` and can learn that it does not fit the real distribution for label 7.

The discriminator typically outputs **one real/fake score given the condition**, not ten digit-class scores. Conditioning does not automatically turn it into a digit classifier.

The basic training scheme uses real images paired with their actual labels and generated images paired with their requested labels. Explicitly training on real images paired with wrong labels is an optional extension; it is not required by the basic cGAN objective and is not used in this notebook.

## 3. Representing the condition

### Why give the label as an input?

In a digit classifier, the class label is the answer the network must predict. In a conditional GAN, it is an input condition that specifies which digit to generate or which class to judge an image against.

| Network | Inputs | Output | Training feedback |
|---|---|---|---|
| Digit classifier | Image | Predicted digit class | Compare prediction with the true class label |
| Conditional generator | Noise + requested class | Generated image | Discriminator's score for the generated image–label pair |
| Conditional discriminator | Image + class condition | Real/fake score | Compare score with the real/fake target |

Giving a classifier the true class label would reveal its answer. Giving the conditional discriminator a class label does not reveal whether an image is real or generated: both real and generated images can arrive with label 7. Its training target is still real/fake, not the digit identity.

**For the generator**, noise alone does not specify whether we want a 3 or a 7. Concatenating the label with noise supplies that request. The label does not contain a picture or specify the correct pixels; the generator learns how to use it through feedback from the discriminator.

**For the discriminator**, the condition lets it judge whether an image fits the real examples associated with the supplied class. Without it, a realistic 3 could be accepted even when the generator was asked for a 7. During the generator update, feedback through $D(G(z,7),7)$ encourages the generator to produce realistic sevens.

In short, **a target tells a network what answer it should produce; a condition tells it which task to perform.**

### One-hot encoding

For ten classes, class 3 becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

In a fully connected generator, concatenate this ten-element vector with the noise. A 100-element noise vector becomes a 110-element input. In a fully connected discriminator, concatenate it with the flattened image: a native 28 × 28 MNIST image gives 784 + 10 = 794 input values.

Concatenation lets the first generator layer learn contributions from both inputs:

$$
h=\phi\bigl(W_z z+W_y\operatorname{onehot}(y)+b\bigr).
$$

Here, $W_z$ and $W_y$ are learned weights, $b$ is a bias, and $\phi$ is an activation function. Changing the label changes the hidden activations, allowing the network to generate a different digit. Training teaches it what each label should produce.

### Conditioning the convolutional reference implementation

The notebook follows the existing DCGAN's 32 × 32 image size and convolutional architecture.

**Generator:** concatenate noise and the one-hot label, then reshape for transposed convolutions.

```text
noise: (B, 100) + one-hot labels: (B, 10)
                  -> (B, 110, 1, 1)
                  -> (B, 256, 4, 4)
                  -> (B, 128, 8, 8)
                  -> (B, 64, 16, 16)
                  -> (B, 1, 32, 32)
```

**Discriminator:** broadcast the one-hot vector across the image grid and concatenate along the channel dimension.

```text
image:                  (B, 1, 32, 32)
label maps:             (B, 10, 32, 32)
combined input:         (B, 11, 32, 32)
discriminator output:   (B, 1)
```

For class 7, its label map is all ones and the other nine label maps are all zeros. The maps encode the requested class, not the shape of the digit.

Broadcasting gives the condition the same spatial dimensions as the image so they can be concatenated as channels. Every local convolutional region then has access to the class condition: the constant maps repeat “the requested class is 7” across the grid. The discriminator learns how to combine that information with image features.

Concatenation is a convenient design choice, not a requirement for conditional generation. Other architectures use learned label embeddings, inject conditions into intermediate layers, or make the final discriminator score depend on the condition. Embeddings are especially useful when there are many classes.

## 4. Pixel range and the generator's Tanh

The dataset transformation is:

```python
transforms.Normalize((0.5,), (0.5,))
```

It maps a pixel $p\in[0,1]$ to $(p-0.5)/0.5=2p-1$. Black becomes −1, mid-gray becomes 0, and white becomes +1. The generator ends with `Tanh` to produce values on the same scale (mathematically between −1 and +1).

For display, undo the transformation with `(images + 1) / 2`. Use a fixed display range of 0 to 1 so that plotting does not independently stretch each image's contrast.

## 5. Two different meanings of label

| Quantity | Values | Role |
|---|---|---|
| Class condition `class_labels` | Integers 0–9 | Input to both networks; says which digit |
| Adversarial target | 1 for real, 0 for fake | Target for binary cross-entropy |

For the generator update, the adversarial target is 1 because we want the generated pair to be accepted as real. This does not mean the requested digit is 1.

## 6. Training, step by step

### A. Update the discriminator

1. Load real images $x$ and their actual class labels $y$.
2. Sample random noise $z$ and generate $G(z,y)$. Reusing the batch's labels keeps real and fake condition frequencies matched.
3. Score real pairs $D(x,y)$ against ones.
4. Score generated pairs $D(G(z,y),y)$ against zeros.
5. Sum these losses, backpropagate, and step only the discriminator optimizer. Detach generated images to prevent gradients flowing into the generator.

### B. Update the generator

1. Sample fresh noise and generate images with the class conditions.
2. Feed the generated images and the same conditions to the discriminator.
3. Compare its output with ones: the generator wants its pairs accepted as real.
4. Backpropagate through the discriminator into the generator and step only the generator optimizer.

Do not detach generated images during step B. The notebook disables discriminator parameter gradients during this step while preserving the gradient path to the images. It also temporarily puts the discriminator in evaluation mode so its BatchNorm running statistics stay fixed during this generator-only update. This is an explicit implementation choice; other GAN implementations keep it in training mode.

## 7. Loss functions

Write $D(x,y)$ for the discriminator's probability estimate. Real pairs $(x,y)$ come from the dataset; generated labels follow the training label distribution and noise is sampled independently.

The discriminator minimizes:

$$
L_D=-\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}\log D(x,y)
-\mathbb{E}_{z,y}\log(1-D(G(z,y),y)).
$$

The non-saturating generator loss is:

$$
L_G=-\mathbb{E}_{z,y}\log D(G(z,y),y).
$$

These have the same structure as the losses in the unconditional scripts; both networks now receive $y$.

The notebook uses **raw discriminator logits with `BCEWithLogitsLoss`**, which combines sigmoid and binary cross-entropy in a numerically stable operation. Therefore, its discriminator does not end with `Sigmoid`. Apply `torch.sigmoid(logits)` only when you want probabilities for inspection. Do not apply sigmoid before passing logits to this loss.

Each network has its own Adam optimizer, with learning rate 0.0002 and `betas=(0.5, 0.999)`. The first beta controls gradient momentum; the second controls the moving average of squared gradients. These settings mirror the existing DCGAN example.

## 8. Conditional GAN versus DCGAN

DCGAN describes a convolutional architecture. Conditional GAN describes conditioning the generation process. A model can be both: the reference notebook is a **conditional DCGAN**.

Relative to the existing script, we retain the MNIST class labels instead of discarding them, add ten condition channels to the generator input and discriminator input, and pass conditions at every call. The basic alternating optimization remains the same.

## 9. Running and evaluating the notebook

Run the notebook from top to bottom in a Python 3 environment with PyTorch, torchvision, and matplotlib. The first dataset run needs internet access to download MNIST. A GPU is useful for the full 30-epoch example; CPU training is slower. The optional smoke mode limits training to two batches and one epoch to check the pipeline, not image quality.

The notebook includes a synthetic single-batch check before the dataset download, class-organized sample grids, loss curves, chosen-digit generation, and checkpoint saving. Generated files go into `conditional_gan_outputs` relative to the notebook's working directory.

Each grid row requests one digit. Columns reuse the same noise vectors across rows, making it easier to inspect the effect of changing the label. This does not guarantee that a column preserves the same handwriting style across classes.

Inspect three things: image realism, agreement with the requested digit, and variety within each digit class. Loss curves alone cannot establish success: adversarial losses may oscillate, and a low generator loss does not guarantee diverse or correctly conditioned images. An independent digit classifier can help quantify class agreement, but is not part of this reference implementation.

## 10. Limitations and discussion questions

Conditioning does not eliminate mode collapse, unstable training, or class imbalance. The generator may produce one style per class, ignore labels, or learn some digits better than others. A fixed seed helps repeat experiments but does not guarantee identical results across hardware and software environments.

Conditions can also be text descriptions, attributes, or other images. Their encoders and architectures change, but the central idea remains generation conditioned on information supplied to the model.

1. Why is giving the label only to the generator insufficient for the intended conditional objective?
2. What happens if we detach the generated image during the generator update?
3. Why does the discriminator output one logit rather than ten class scores?
4. How would you distinguish ignoring noise from ignoring labels using sample grids?
5. What would need to change if real pixels stayed in the range [0, 1]?
