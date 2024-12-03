import tensorflow as tf
from transformers import TFAlbertModel

def create_model(max_len=768):
    # Ganti dengan model IndoBERT biasa
    bert_model = TFAlbertModel.from_pretrained("indobenchmark/indobert-lite-large-p1", from_pt=True)
    
    # Optimizer, loss, dan metrik
    opt = tf.keras.optimizers.AdamW(learning_rate=2e-5)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    # Input layers
    input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')

    # Embedding layer dari model pretrained
    embeddings = bert_model([input_ids, attention_masks])[1]
    
    # Lapisan tambahan untuk klasifikasi
    x = tf.keras.layers.Dropout(0.2)(embeddings)
    output = tf.keras.layers.Dense(2, activation="softmax")(x)

    # Model akhir
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

    # Kompilasi model dengan metrik custom (BalancedAccuracy)
    model.compile(optimizer=opt, loss=loss, metrics=accuracy)

    return model
