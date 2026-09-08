# Introductory encoder–decoder lab

A 60–90 minute PyTorch exercise aligned with [the encoder–decoder lecture](../../lectures/encoder_decoder/encoder_decoder.md).

- [Student notebook](encoder_decoder_lab.ipynb): four implementation TODOs, shape checks, reconstruction plots, and a bottleneck experiment.
- [Instructor solution](encoder_decoder_solution.ipynb): completed code and conceptual answer guidance. Distribute only the student notebook when assigning the exercise.
- [Optional U-Net pet-segmentation lab](optional_unet_pet_segmentation.md): a Kaggle-friendly semantic-segmentation exercise using the Oxford-IIIT Pet Dataset.

Use a Python 3 Jupyter kernel with `torch`, `torchvision`, and `matplotlib` installed. MNIST downloads into `./data` relative to the notebook working directory on first use. CPU execution is supported; runtime depends on hardware. The exercise uses 5,000 training and 1,000 held-out images, five epochs per run, and no class-label supervision.

The optional extension changes reconstruction into denoising using the same architecture.
