from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import sys
import tensorflow as tf
import tensorflow.keras as keras
from bert.tokenization.bert_tokenization import FullTokenizer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import (
    TensorBoard
)
import tensorflow_addons as tfa
from lr_warmup import *
from utils import *
from official.nlp import optimization as opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['AUTOGRAPH_VERBOSITY'] = "0"

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

pretrained_path = '/path/to/bert/pretrained/uncased/uncased_L-12_H-768_A-12'
bert_config_file = os.path.join(pretrained_path, 'bert_config.json')
bert_ckpt_file = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
tokenizer = FullTokenizer(vocab_file=vocab_path)

AUTO = tf.data.AUTOTUNE
batch_size = 32
contrastive_lr = 2e-5
classifier_lr = 5e-5
image_size = 224
max_seq_len = 197
EPOCHS = 50
channels = 3
temperature = 1.0
width = 768

encoder_weights_dir = "/path/to/weights/encoder/"

path_train = '/data/sbakka01/tensorflow_datasets/rvlcdip/'
weights_name = "CP-SupCon_CMHFL.ckpt"
model_weights_name = "SupCon_CMHFL"
NAME = "SupCon_CMHFL." + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
encoder_multimodal_weights_dir = "/path/to/encoder/"
classifier_multimodal_weights_dir = "/path/to/classifier/"
history_dir = "/path/to/history/"
checkpoint_logs = "/path/to/logs/"
checkpoint_dir = "/path/to/checkpoints/"
checkpoint_csv = "/path/to/results/"

tensorboard = TensorBoard(log_dir=checkpoint_logs + NAME, histogram_freq=1)
early_stop = EarlyStopping(monitor='val_Image_Accuracy', patience=5, mode='max', verbose=1, restore_best_weights=True),
checkpoints = ModelCheckpoint(filepath=checkpoint_dir + weights_name,
                              save_weights_only=True,
                              save_best_only=True,
                              monitor='val_Image_Accuracy',
                              mode='max',
                              verbose=1,
                              save_freq="epoch",
                              )
csv_logger = CSVLogger(checkpoint_csv + model_weights_name + ".csv", append=True, separator=",")
lr_decay = ReduceLROnPlateau(factor=np.sqrt(0.5), monitor="val_Classifier_loss", mode="min", cooldown=0, patience=3,
                             min_lr=0.5e-5, verbose=1)

if not os.path.exists(encoder_multimodal_weights_dir):
    os.makedirs(encoder_multimodal_weights_dir)
if not os.path.exists(classifier_multimodal_weights_dir):
    os.makedirs(classifier_multimodal_weights_dir)
if not os.path.exists(history_dir):
    os.makedirs(history_dir)
if not os.path.exists(checkpoint_logs):
    os.makedirs(checkpoint_logs)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(checkpoint_csv):
    os.makedirs(checkpoint_csv)

Callbacks = [checkpoints, csv_logger, early_stop]    

data_train = create_rvlcdip_dataset(path_train, "Valid_Data")
data_valid = create_rvlcdip_dataset(path_train, "Valid_Data")
data_test = create_rvlcdip_dataset(path_train, "Test_Data")

def train_batchgen_gen():
    while True:
        for samples in data_train:
            image_train, text_train, labels = samples[0], samples[1], samples[2]
            image_train, text_train = Read_Image(image_train, image_size), Preprocess_Text(text_train, tokenizer, max_seq_len)
            yield image_train, text_train, labels, labels    
            
def valid_batchgen_gen():
    while True:
        for samples in data_valid:
            image_valid, text_valid, labels = samples[0], samples[1], samples[2]
            image_valid, text_valid = Read_Image(image_valid, image_size), Preprocess_Text(text_valid, tokenizer, max_seq_len)
            yield image_valid, text_valid, labels, labels            

def test_batchgen_gen():
    while True:
        for samples in data_test:
            image_test, text_test, labels = samples[0], samples[1], samples[2]
            image_test, text_test = Read_Image(image_test, image_size), Preprocess_Text(text_test, tokenizer, max_seq_len)
            yield image_test, text_test, labels, labels
            
class ContrastiveModel(tf.keras.Model):
    def __init__(self, encoder, classifier):
        super(ContrastiveModel, self).__init__()
        self.temperature = temperature
        self.encoder = encoder
        
        self.classifier = classifier
        self.intra_image_projection_head = tf.keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation=tf.nn.gelu),
                layers.Dense(width),
            ],
            name="intra_image_projection_head",
        )
        self.inter_image_projection_head = tf.keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation=tf.nn.gelu),
                layers.Dense(width),
            ],
            name="inter_image_projection_head",
        )
        self.intra_text_projection_head = tf.keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation=tf.nn.gelu),
                layers.Dense(width),
            ],
            name="intra_text_projection_head",
        )
        self.inter_text_projection_head = tf.keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation=tf.nn.gelu),
                layers.Dense(width),
            ],
            name="inter_text_projection_head",
        )
        
    def compile(self, contrastive_optimizer, classifier_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.classifier_optimizer = classifier_optimizer
        
        # self.contrastive_loss will be defined as a method
        self.SupCon_loss_tracker = tf.keras.metrics.Mean(name="SupCon_loss")
        self.Image_SupCon_loss_tracker = tf.keras.metrics.Mean(name="Image_SupCon_loss")
        self.Text_SupCon_loss_tracker = tf.keras.metrics.Mean(name="Text_SupCon_loss")
        
        self.Classifier_loss_tracker = tf.keras.metrics.Mean(name="Classifier_loss")
        self.Image_loss_tracker = tf.keras.metrics.Mean(name="Image_Classifier_Loss")
        self.Text_loss_tracker = tf.keras.metrics.Mean(name="Text_Classifier_Loss")        
        
        self.Image_supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                                   reduction=tf.keras.losses.Reduction.NONE)
        self.Text_supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                                  reduction=tf.keras.losses.Reduction.NONE)
        self.Image_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="Image_Accuracy")
        self.Text_accuracy =  tf.keras.metrics.SparseCategoricalAccuracy(name="Text_Accuracy")
        
    @property
    def metrics(self):
        return [self.SupCon_loss_tracker,
                self.Image_SupCon_loss_tracker,
                self.Text_SupCon_loss_tracker,
                self.Classifier_loss_tracker,
                self.Image_loss_tracker,
                self.Text_loss_tracker,
                self.Image_accuracy,
                self.Text_accuracy,
                ]
    
    def Image_SupCon_loss(self, labels, image_feature_vectors):
        # Normalize feature vectors
        image_feature_vectors_normalized = tf.math.l2_normalize(image_feature_vectors, axis=1)        
        # Compute logits
        image_logits = tf.divide(
            tf.matmul(
                image_feature_vectors_normalized, tf.transpose(image_feature_vectors_normalized)
            ),
            self.temperature,
        )
        image_contrastive_loss = tfa.losses.npairs_loss(tf.squeeze(labels), image_logits)
                                 
        return image_contrastive_loss
    
    def Image_Cross_SupCon_loss(self, labels, image_feature_vectors, text_feature_vectors):
        # Normalize feature vectors
        image_feature_vectors_normalized = tf.math.l2_normalize(image_feature_vectors, axis=1)
        text_feature_vectors_normalized = tf.math.l2_normalize(text_feature_vectors, axis=1)
        
        # Compute logits
        text_logits = tf.matmul(image_feature_vectors_normalized, tf.transpose(text_feature_vectors_normalized))

        image_cross_contrastive_loss = tfa.losses.npairs_loss(tf.squeeze(labels), text_logits)
        return image_cross_contrastive_loss
    
    def Text_SupCon_loss(self, labels, text_feature_vectors):
        # Normalize feature vectors
        text_feature_vectors_normalized = tf.math.l2_normalize(text_feature_vectors, axis=1)
        
        # Compute logits        
        text_logits = tf.divide(
            tf.matmul(
                text_feature_vectors_normalized, tf.transpose(text_feature_vectors_normalized)
            ),
            self.temperature,
        )
        
        text_contrastive_loss = tfa.losses.npairs_loss(tf.squeeze(labels), text_logits)
        return text_contrastive_loss
    
    def Text_Cross_SupCon_loss(self, labels, text_feature_vectors, image_feature_vectors):
        # Normalize feature vectors
        text_feature_vectors_normalized = tf.math.l2_normalize(text_feature_vectors, axis=1)
        image_feature_vectors_normalized = tf.math.l2_normalize(image_feature_vectors, axis=1)
        
        # Compute logits        
        text_cross_logits = tf.divide(
            tf.matmul(
                image_feature_vectors_normalized, tf.transpose(image_feature_vectors_normalized)
            ),
            self.temperature,
        )
        
        text_cross_contrastive_loss = tfa.losses.npairs_loss(tf.squeeze(labels), text_cross_logits)
        return text_cross_contrastive_loss

    def train_step(self, data):
        # Unpack data
        image_inputs, text_inputs, image_labels, text_labels = data

        with tf.GradientTape() as tape:
            # Forward pass of image
            image_features, text_features = self.encoder([image_inputs, text_inputs], training=True)
            # Compute losses
            image_intra_projections = self.intra_image_projection_head(image_features, training=True)
            image_inter_projections = self.inter_image_projection_head(image_features, training=True)
            text_intra_projections = self.intra_text_projection_head(text_features, training=True)
            text_inter_projections = self.inter_text_projection_head(text_features, training=True)
            
            intra_image_loss = self.Image_SupCon_loss(image_labels, image_intra_projections)
            inter_image_loss = self.Image_Cross_SupCon_loss(image_labels, image_inter_projections, text_inter_projections)
            
            intra_text_loss = self.Text_SupCon_loss(text_labels, text_intra_projections)
            inter_text_loss = self.Text_Cross_SupCon_loss(text_labels, text_inter_projections, image_inter_projections)
            
            image_loss = intra_image_loss + inter_image_loss
            text_loss = intra_text_loss + inter_text_loss
            
            SupCon_loss = image_loss + text_loss

        # Compute gradients and update the parameters.
        supcon_learnable_params = self.encoder.trainable_variables + \
                                  self.intra_image_projection_head.trainable_variables + \
                                  self.intra_text_projection_head.trainable_variables + \
                                  self.inter_image_projection_head.trainable_variables + \
                                  self.inter_text_projection_head.trainable_variables
        
        supcon_gradients = tape.gradient(SupCon_loss, supcon_learnable_params)
        # Update weights
        self.contrastive_optimizer.apply_gradients(zip(supcon_gradients, supcon_learnable_params))

        # Update the metrics configured in `compile()`.
        self.SupCon_loss_tracker.update_state(SupCon_loss)
        self.Image_SupCon_loss_tracker.update_state(image_loss)
        self.Text_SupCon_loss_tracker.update_state(text_loss)
        
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            supcon_image_features, supcon_text_features = self.encoder([image_inputs, text_inputs], training=False)
            image_class_logits, text_class_logits = self.classifier([supcon_image_features, supcon_text_features], training=True)
            
            image_probe_loss = self.Image_supervised_loss(image_labels, image_class_logits)
            text_probe_loss = self.Text_supervised_loss(text_labels, text_class_logits)
            
            total_loss = image_probe_loss + text_probe_loss
        
        # Compute gradients and update the parameters.
        classifier_learnable_params = self.classifier.trainable_variables
        classifier_gradients = tape.gradient(total_loss, classifier_learnable_params)
        self.classifier_optimizer.apply_gradients(zip(classifier_gradients, classifier_learnable_params))
        
        self.Classifier_loss_tracker.update_state(total_loss)
        self.Image_loss_tracker.update_state(image_probe_loss)
        self.Text_loss_tracker.update_state(text_probe_loss)
        self.Image_accuracy.update_state(image_labels, image_class_logits)
        self.Text_accuracy.update_state(text_labels, text_class_logits)

        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        image_inputs, text_inputs, image_labels, text_labels = data

        supcon_image_features, supcon_text_features = self.encoder([image_inputs, text_inputs], training=False)
        image_class_logits, text_class_logits = self.classifier([supcon_image_features, supcon_text_features], training=False)

        image_probe_loss = self.Image_supervised_loss(image_labels, image_class_logits)
        text_probe_loss = self.Text_supervised_loss(text_labels, text_class_logits)

        total_loss = image_probe_loss + text_probe_loss        
        self.Classifier_loss_tracker.update_state(total_loss)
        self.Image_loss_tracker.update_state(image_probe_loss)
        self.Text_loss_tracker.update_state(text_probe_loss)
        self.Image_accuracy.update_state(image_labels, image_class_logits)
        self.Text_accuracy.update_state(text_labels, text_class_logits)
        
        return {m.name: m.result() for m in self.metrics[3:]}
    
def main():
    train_dataset = tf.data.Dataset.from_generator(train_batchgen_gen,
                                                   output_types=(tf.float32, tf.int64, tf.float32, tf.float32))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(AUTO)
    train_dataset = train_dataset.with_options(options)
        
    valid_dataset = tf.data.Dataset.from_generator(valid_batchgen_gen, 
                                                   output_types=(tf.float32, tf.int64, tf.float32, tf.float32))
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.prefetch(AUTO)
    valid_dataset = valid_dataset.with_options(options)
    
    test_dataset = tf.data.Dataset.from_generator(test_batchgen_gen, 
                                                  output_types=(tf.float32, tf.int64, tf.float32, tf.float32))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(AUTO)
    test_dataset = test_dataset.with_options(options)
    
    num_training_samples = len(data_train)
    train_steps_per_epoch = num_training_samples // batch_size
    
    num_validation_samples = len(data_valid)
    valid_steps_per_epoch = num_validation_samples // batch_size
    
    num_test_samples = len(data_valid)
    test_steps_per_epoch = num_test_samples // batch_size
    
    num_train_steps = train_steps_per_epoch * EPOCHS
    warmup_steps = EPOCHS * num_training_samples * 0.1 // batch_size
      
    with strategy.scope():
        encoder = get_multimodal_encoder()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=encoder_weights_dir)
        print('[INFO] Loading trained model ...', latest_checkpoint)
        encoder.load_weights(latest_checkpoint).expect_partial()
        
        classifier = get_VL_finetuning_classifier()
        trainer = ContrastiveModel(encoder=encoder, 
                                              classifier=classifier)
        
        contrastive_optimizer = opt.create_optimizer(init_lr=contrastive_lr,
                                                     num_train_steps=num_train_steps,
                                                     num_warmup_steps=warmup_steps,
                                                     optimizer_type='adamw')
        
        classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=classifier_lr)
        
        trainer.compile(contrastive_optimizer=contrastive_optimizer,
                        classifier_optimizer=classifier_optimizer,
                        )

    print('[INFO] Training...')
    
    history = trainer.fit(train_dataset,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_data=train_dataset,
                         validation_steps=train_steps_per_epoch,
                         epochs=EPOCHS,
                         shuffle=False,
                         callbacks=Callbacks,
                         verbose=1,
                         initial_epoch=0,
                         )
        
    print("Maximal Image validation accuracy: {:.2f}%".format(max(history.history["val_Image_Accuracy"]) * 100))
    print("Maximal Text validation accuracy: {:.2f}%".format(max(history.history["val_Text_Accuracy"]) * 100))
    
    print('[INFO] Evaluating...')
    
    scores = trainer.evaluate(test_dataset, steps=test_steps_per_epoch, verbose=1)
    for score, metric_name in zip(scores, trainer.metrics_names):
        print("%s : %0.6f" % (metric_name, score))
        
    print("[INFO] =======>>>>> Saving Model weights")
    # Extract the backbone Image Vision Transformer.
    encoder_backbone = tf.keras.Model(trainer.encoder.input, trainer.encoder.output)
    classifier_backbone = tf.keras.Model(trainer.classifier.input, trainer.classifier.output)

    encoder_backbone.save_weights(filepath=encoder_multimodal_weights_dir+model_weights_name,
                                  overwrite=True,
                                  save_format='tf') 
    
    classifier_backbone.save_weights(filepath=classifier_multimodal_weights_dir+model_weights_name,
                                     overwrite=True,
                                     save_format='tf') 
    
    print("[INFO] =======>>>>> Weights have been saved correctly")
    
if __name__ == '__main__':
    main() 
    