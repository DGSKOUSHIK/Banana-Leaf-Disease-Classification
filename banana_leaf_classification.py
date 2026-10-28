"""Banana leaf classification with a CNN and convolutional KernelAttention.

Research-based implementation, not the authors' original training source.

Expected data layout (split BEFORE augmentation; no duplicates across splits):
  dataset/
    train/{Cordana,Healthy,Pestalotiopsis,Sigatoka}/*.jpg
    val/{Cordana,Healthy,Pestalotiopsis,Sigatoka}/*.jpg
    test/{Cordana,Healthy,Pestalotiopsis,Sigatoka}/*.jpg

Install: pip install tensorflow numpy scikit-learn matplotlib
Run: python banana_leaf_classification.py --data dataset --epochs 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class KernelAttention(tf.keras.layers.Layer):
    """Convolutional Q/K/V, sigmoid spatial weighting, and residual projection."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.query_conv = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.query_bn = tf.keras.layers.BatchNormalization()
        self.key_conv = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.key_bn = tf.keras.layers.BatchNormalization()
        self.value_conv = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.value_bn = tf.keras.layers.BatchNormalization()
        self.attention_conv = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
        self.projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.projection = tf.keras.layers.Conv2D(self.filters, 1, padding="same")
        super().build(input_shape)

    def call(self, inputs, training=None):
        q = tf.nn.relu(self.query_bn(self.query_conv(inputs), training=training))
        k = tf.nn.relu(self.key_bn(self.key_conv(inputs), training=training))
        v = tf.nn.relu(self.value_bn(self.value_conv(inputs), training=training))
        attention = self.attention_conv(q + k)
        residual = self.projection(inputs) if self.projection is not None else inputs
        return v * attention + residual

    def get_config(self):
        return {**super().get_config(), "filters": self.filters}


def create_model(num_classes=4):
    inputs = tf.keras.Input((224, 224, 3))
    # Applied only during training.
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
    ], name="training_augmentation")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    for filters in (64, 128, 256):
        x = KernelAttention(filters, name=f"kernel_attention_{filters}")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="banana_kernel_attention_cnn")


def load_split(directory, class_names=None, shuffle=False, batch_size=32):
    options = dict(directory=str(directory), image_size=(224, 224), batch_size=batch_size,
                   label_mode="int", shuffle=shuffle)
    if class_names is not None:
        options["class_names"] = class_names
    return tf.keras.utils.image_dataset_from_directory(**options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("banana_results"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if not (args.data / split).is_dir():
            parser.error(f"Missing {args.data / split}; prepare original-image splits before training")
    train = load_split(args.data / "train", shuffle=True, batch_size=args.batch_size)
    names = train.class_names
    if len(names) != 4:
        parser.error(f"Expected four classes, found: {names}")
    val = load_split(args.data / "val", names, batch_size=args.batch_size)
    test = load_split(args.data / "test", names, batch_size=args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)
    test = test.prefetch(tf.data.AUTOTUNE)
    model = create_model(len(names))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    history = model.fit(train, validation_data=val, epochs=args.epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=8, restore_best_weights=True)])
    model.save(args.output / "banana_model.keras")
    test_loss, test_accuracy = model.evaluate(test, verbose=0)
    from sklearn.metrics import classification_report, confusion_matrix
    truths, predictions, probabilities = [], [], []
    for images, labels in test:
        probs = model.predict_on_batch(images)
        truths.extend(labels.numpy().tolist())
        predictions.extend(np.argmax(probs, axis=1).tolist())
        probabilities.extend(probs.tolist())
    report = classification_report(truths, predictions, labels=list(range(len(names))),
                                   target_names=names, zero_division=0, output_dict=True)
    result = {"class_names": names, "test_loss": float(test_loss),
              "test_accuracy": float(test_accuracy), "confusion_matrix":
              confusion_matrix(truths, predictions, labels=list(range(len(names)))).tolist(),
              "classification_report": report, "history": history.history}
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    try:
        labels_one_hot = label_binarize(np.array(truths), classes=list(range(len(names))))
        result["macro_roc_auc_ovr"] = float(roc_auc_score(labels_one_hot, np.array(probabilities), average="macro"))
    except ValueError:
        result["macro_roc_auc_ovr"] = None
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(f"Test accuracy: {test_accuracy:.4%}. Results saved in {args.output}")
    print("No accuracy is guaranteed; this implementation requires independent training and testing.")


if __name__ == "__main__":
    main()
