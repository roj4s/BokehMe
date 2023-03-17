import os
import tempfile
import tensorflow as tf

def save_tflite(model, output_path, quantize=False) -> None:  # pylint: disable=arguments-differ
        """! Convert and save Keras Model to Tflite."""
        tmpdir = tempfile.mkdtemp()
        temp_model_addr = os.path.join(tmpdir, 'model/1/')
        tf.saved_model.save(model, temp_model_addr)

        converter = tf.lite.TFLiteConverter.from_saved_model(temp_model_addr)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        try:
            os.removedirs(temp_model_addr)
        except OSError:
            pass

        with open(output_path, "wb") as file_desc:
            file_desc.write(tflite_model)
