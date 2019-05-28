# Batching

Batching is a set of tools to format data for training sequence models.

[![Build Status](https://travis-ci.org/cirick/batching.svg?branch=master)](https://travis-ci.org/cirick/batching)
[![Coverage Status](https://coveralls.io/repos/github/cirick/batching/badge.svg?branch=master)](https://coveralls.io/github/cirick/batching?branch=master)

## Installation
```shell
$ pip install batching
```

## Example usage
Example script exists in sample.py
```python
# Metadata for batch info - including batch IDs and mappings to storage resouces like filenames
storage_meta = StorageMeta(validation_split=0.2)

# Storage for batch data - Memory, Files, S3
storage = BatchStorageMemory(storage_meta)

# Create batches - configuration contains feature names, windowing config, timeseries spacing
batch_generator = Builder(storage, 
                          feature_set, 
                          look_back, 
                          look_forward, 
                          batch_seconds, 
                          batch_size=128)
batch_generator.generate_and_save_batches(list_of_dataframes)

# Generator for feeding batches to training - tf.keras.model.fit_generator
train_generator = BatchGenerator(storage)
validation_generator = BatchGenerator(storage, is_validation=True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, activation='sigmoid')
model.compile(loss=tf.keras.losses.binary_crossentropy, 
              optimizer=tf.keras.optimizers.Adam(), 
              metrics=['accuracy'])
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=epochs)
```

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2015 © <a href="http://fvcproductions.com" target="_blank">FVCproductions</a>.