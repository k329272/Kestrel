"""Architecture builders for the two Kestrel heads.

Pipeline order (class-first cascade):

  1. class_model(image) -> predicted class value in [-1, 1]  (Rock/Paper/Scissors)
  2. bbox_model(image, class_value) -> predicted xmin of the hand

The bbox head now takes the class prediction as a second input (a small
embedding concatenated onto the image features), so it can condition its box
estimate on what it's looking for . This is why the class model needs to run
first and why its input covers the full frame rather than a small crop
around a not-yet-known box.

Model names follow a YOLO-style convention: `kestrel26n.yaml` -> the
trailing letter (n/s/m/l/x) scales the width of the network, same idea as
ultralytics' n/s/m/l/x checkpoints.
"""

import tensorflow as tf

# width_multiple only -- these are tiny models, depth scaling isn't used.
SCALES = {
    "n": 0.25,  # nano
    "s": 0.50,  # small
    "m": 0.75,  # medium
    "l": 1.00,  # large
    "x": 1.25,  # xlarge
}
DEFAULT_SCALE = "n"


def parse_scale(cfg_path):
    """Extract the trailing scale letter from a filename like 'kestrel26n.yaml'.

    Falls back to DEFAULT_SCALE if there's no recognized letter suffix.
    """
    stem = str(cfg_path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem and stem[-1].lower() in SCALES:
        return stem[-1].lower()
    return DEFAULT_SCALE


def _scaled(base_channels, scale):
    mult = SCALES.get(scale, SCALES[DEFAULT_SCALE])
    return max(1, round(base_channels * mult / SCALES[DEFAULT_SCALE]))


def build_class_model(cfg, scale=DEFAULT_SCALE):
    """Runs FIRST, on the full frame (not a crop -- there's no box to crop to yet)."""
    width, height = cfg.get("imgsz", [160, 120])
    filters = _scaled(cfg.get("cls", {}).get("filters", 8), scale)
    dense_units = _scaled(cfg.get("cls", {}).get("dense", 16), scale)

    inp = tf.keras.Input(shape=(height, width, 1), name="class_image_input")
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="tanh", name="class_output")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="kestrel_class")


def build_bbox_model(cfg, scale=DEFAULT_SCALE):
    """Runs SECOND. Takes the image plus the class prediction from class_model."""
    width, height = cfg.get("imgsz", [160, 120])
    filters = _scaled(cfg.get("bbox", {}).get("filters", 8), scale)
    dense_units = _scaled(cfg.get("bbox", {}).get("dense", 24), scale)
    class_embed = _scaled(cfg.get("bbox", {}).get("class_embed", 8), scale)

    img_in = tf.keras.Input(shape=(height, width, 1), name="bbox_image_input")
    cls_in = tf.keras.Input(shape=(1,), name="bbox_class_input")  # predicted class value, [-1, 1]

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(img_in)
    x = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))(x)
    x = tf.keras.layers.Flatten()(x)

    c = tf.keras.layers.Dense(class_embed, activation="relu", name="class_embed")(cls_in)

    x = tf.keras.layers.Concatenate(name="image_class_concat")([x, c])
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    out = tf.keras.layers.Dense(1, name="bbox_output")(x)
    return tf.keras.Model(inputs=[img_in, cls_in], outputs=out, name="kestrel_bbox")
