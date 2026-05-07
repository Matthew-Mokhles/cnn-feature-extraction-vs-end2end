"""
members/Matthew_xray/data_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Optional faster pipeline based on tf.data + image_dataset_from_directory.

Used by endtoend_dl.py when ``cfg.USE_TF_DATA`` is True. The legacy
ImageDataGenerator path in preprocessing.py is preserved for steps 2 and 3.

Augmentation here is implemented with built-in Keras layers (RandomFlip,
RandomRotation, RandomZoom, RandomContrast, RandomBrightness) so it can fuse
into the graph and run on GPU when available.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras import layers
import config as cfg

AUTOTUNE = tf.data.AUTOTUNE


class RandomErasing(layers.Layer):
    """Random Erasing (Cutout) — masks a random rectangle with per-channel mean.

    During training, each image has ``probability`` chance of having a random
    rectangular region (area between ``area_range[0]`` and ``area_range[1]``
    fraction of the image) replaced with the per-channel mean of that image.
    This forces the model to rely on global structure rather than local texture
    shortcuts — critical for bridging the train/test domain gap.

    At inference time this layer is a no-op.
    """

    def __init__(self, probability=0.3, area_range=(0.02, 0.20),
                 aspect_range=(0.3, 3.3), **kwargs):
        super().__init__(**kwargs)
        self.probability  = probability
        self.area_range   = area_range
        self.aspect_range = aspect_range

    def call(self, images, training=None):
        if not training:
            return images

        batch_size = tf.shape(images)[0]
        h          = tf.shape(images)[1]
        w          = tf.shape(images)[2]

        # Decide which images in the batch get erased.
        mask = tf.random.uniform([batch_size]) < self.probability

        # Random area fraction and aspect ratio.
        area   = tf.random.uniform([batch_size],
                                   self.area_range[0], self.area_range[1])
        aspect = tf.random.uniform([batch_size],
                                   self.aspect_range[0], self.aspect_range[1])

        eh = tf.cast(tf.math.sqrt(tf.cast(h * w, tf.float32) * area / aspect),
                     tf.int32)
        ew = tf.cast(tf.math.sqrt(tf.cast(h * w, tf.float32) * area * aspect),
                     tf.int32)
        eh = tf.minimum(eh, h)
        ew = tf.minimum(ew, w)

        # Random top-left corner.
        top  = tf.random.uniform([batch_size], 0, h - eh + 1, dtype=tf.int32)
        left = tf.random.uniform([batch_size], 0, w - ew + 1, dtype=tf.int32)

        def erase_single(args):
            img, do_erase, t, l, rh, rw = args
            if not do_erase:
                return img
            # Per-channel mean fill.
            ch_mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
            fill    = tf.broadcast_to(ch_mean, [rh, rw, tf.shape(img)[2]])
            # Pad the fill patch to full image size.
            padded = tf.pad(fill, [[t, h - t - rh], [l, w - l - rw], [0, 0]])
            # Binary mask: 1 where erased.
            ones   = tf.ones([rh, rw, 1])
            bmask  = tf.pad(ones, [[t, h - t - rh], [l, w - l - rw], [0, 0]])
            return img * (1 - bmask) + padded * bmask

        erased = tf.map_fn(
            erase_single,
            (images, mask, top, left, eh, ew),
            fn_output_signature=tf.float32,
        )
        return erased

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "probability":  self.probability,
            "area_range":   self.area_range,
            "aspect_range": self.aspect_range,
        })
        return cfg


def _augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(20.0 / 360.0),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
        RandomErasing(probability=0.3, area_range=(0.02, 0.20)),
    ], name="augmentation")


def _load_split(directory, shuffle, seed=42):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary" if cfg.CLASS_MODE == "binary" else "categorical",
        class_names=cfg.CLASS_NAMES,
        color_mode="rgb",
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMG_SIZE,
        shuffle=shuffle,
        seed=seed,
    )


def _prepare(ds, training, augmenter=None, shuffle_buffer=1024):
    """Apply preprocess_input + (optional) augmentation, then cache & prefetch.

    Order is important:
      cache() -> shuffle() -> map(augment) -> prefetch()

    cache() must come before shuffle() so each epoch shuffles a fresh
    permutation of the cached tensors instead of replaying the same order
    forever. Augmentation must come AFTER cache so the random transforms are
    re-sampled every epoch (caching augmented tensors would freeze the
    augmentation, defeating its purpose).
    """
    preprocess = cfg.PREPROCESS_FN

    def _apply_preproc(x, y):
        x = tf.cast(x, tf.float32)
        x = preprocess(x)
        return x, y

    ds = ds.map(_apply_preproc, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # the *preprocessed* tensors are cached (no augmentation yet)

    if training:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        if augmenter is not None:
            def _apply_aug(x, y):
                return augmenter(x, training=True), y
            ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def get_datasets():
    """Return (train_ds, val_ds, test_ds) as cached + prefetched tf.data.Datasets.

    Each dataset yields (image, label) batches with images already passed
    through ``cfg.PREPROCESS_FN``. Labels are float32 tensors of shape (B, 1)
    in binary mode so they can feed straight into a sigmoid + BCE head.
    """
    raw_train = _load_split(cfg.TRAIN_DIR, shuffle=True)
    raw_val   = _load_split(cfg.VAL_DIR,   shuffle=False)
    raw_test  = _load_split(cfg.TEST_DIR,  shuffle=False)

    augmenter = _augmentation_layer()
    train_ds = _prepare(raw_train, training=True,  augmenter=augmenter)
    val_ds   = _prepare(raw_val,   training=False)
    test_ds  = _prepare(raw_test,  training=False)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        try:
            n_batches = int(ds.cardinality().numpy())
        except Exception:
            n_batches = -1
        print(f"  {name:<5} batches={n_batches}  element_spec={ds.element_spec}")
