import tensorflow as tf

def build_model(input_shape, classes, learning_rate):
    """
    Builds a custom deep CNN architecture for Covid-19 chest X-ray detection
    matching the specified layer configurations.
    """
    model = tf.keras.models.Sequential([
        # 1. Feature Extraction Layers
        tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), 
            activation="relu", 
            input_shape=input_shape, 
            kernel_initializer='he_normal'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Dropout(0.25),
        
        # 2. Neural Network (Dense Layers)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(classes, activation="softmax")
    ])

    # 3. Model Compilation Engine
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model