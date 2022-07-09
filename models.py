from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from tensorflow.keras import layers
from vit_keras import vit

from utils import *

pretrained_path = '/path/to/bert/uncased/model/uncased_L-12_H-768_A-12'
bert_config_file = os.path.join(pretrained_path, 'bert_config.json')
bert_ckpt_file = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
tokenizer = FullTokenizer(vocab_file=vocab_path)

max_seq_len = 197
image_size = 224
channels = 3
PROJECT_DIM = 768
dropout = 0.1
num_heads = 2
mlp_dim = 768

def get_vit_encoder():
    model_vit = vit.vit_b16(image_size=image_size,
                            activation='softmax',
                            pretrained=True,
                            include_top=False,
                            pretrained_top=False,
                            )
    model = tf.keras.Model(inputs=model_vit.input, outputs=model_vit.layers[-2].output, name="Image_Encoder")
    return model


def get_text_encoder():
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, trainable=True, name="bert")
    input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    bert_output = bert(input_ids)
    model = tf.keras.Model(inputs=input_ids, outputs=bert_output, name="Text_Encoder")
    model.build(input_shape=(None, max_seq_len))
    load_stock_weights(bert, bert_ckpt_file)
    return model

def get_multimodal_encoder():
    image_inputs = tf.keras.Input(shape=(image_size, image_size, channels), dtype='int32', name="Input_Image")
    image_encoder = get_vit_encoder()
    image_encoder = image_encoder(image_inputs)

    text_inputs = tf.keras.Input(shape=(max_seq_len,), dtype='int32', name="Input_Text")
    text_encoder = get_text_encoder()
    text_encoder = text_encoder(text_inputs)

    image_attention_00, _ = Image_MultiHeadSelfAttentio(nnum_heads=num_heads,
                                                    mlp_dim=mlp_dim,
                                                    dropout=dropout)(image_encoder)
    text_attention_00, _ = Image_MultiHeadSelfAttentio(nnum_heads=num_heads,
                                                    mlp_dim=mlp_dim,
                                                    dropout=dropout)(text_encoder)

    image_attention_01, text_attention_01, _, _ = Cross_TransformerBlock_01(num_heads=num_heads,
                                                                            mlp_dim=mlp_dim,
                                                                            dropout=dropout)(image_attention_00, text_attention_00)

    image_attention_02 = tf.multiply([image_attention_00, image_attention_01])
    image_attention_03 = tf.add([image_attention_00, image_attention_02])
    image_attention_04 = layers.Dense(PROJECT_DIM, activation=tf.nn.gelu)(image_attention_03)

    text_attention_02 = tf.multiply([text_attention_00, text_attention_01])
    text_attention_03 = tf.add([text_attention_00, text_attention_02])
    text_attention_04 = layers.Dense(PROJECT_DIM, activation=tf.nn.gelu)(text_attention_03)

    image_attention_05, _ = Image_MultiHeadSelfAttentio(nnum_heads=num_heads,
                                                    mlp_dim=mlp_dim,
                                                    dropout=dropout)(image_attention_04)
    text_attention_05, _ = Text_MultiHeadSelfAttentio(nnum_heads=num_heads,
                                                    mlp_dim=mlp_dim,
                                                    dropout=dropout)(text_attention_04)
    
    image_attention_06, text_attention_06, _, _ = Cross_TransformerBlock_01(num_heads=num_heads,
                                                                      mlp_dim=mlp_dim,
                                                                      dropout=dropout)(image_attention_05, text_attention_05)

    image_attention_07 = tf.multiply([image_attention_05, image_attention_06])
    image_attention_08 = tf.add([image_attention_05, image_attention_07])
    image_attention_09 = layers.Dense(PROJECT_DIM, activation=tf.nn.gelu)(image_attention_08)

    text_attention_07 = tf.multiply([text_attention_05, text_attention_06])
    text_attention_08 = tf.add([text_attention_05, text_attention_07])
    text_attention_09 = layers.Dense(PROJECT_DIM, activation=tf.nn.gelu)(text_attention_08)

    image_features = layers.GlobalAveragePooling1D()(image_attention_09)
    image_features = layers.Dropout(dropout)(image_features)
    image_features = layers.Dense(PROJECT_DIM, activation=tf.nn.gelu)(image_features)

    text_features = layers.GlobalAveragePooling1D()(text_attention_09)
    text_features = layers.Dropout(dropout)(text_features)
    text_features = layers.Dense(PROJECT_DIM, activation=tf.nn.gelu)(text_features)

    model = tf.keras.Model(inputs=[image_inputs, text_inputs],
                           outputs=[image_features, text_features],
                           name="Attention_Model")
    return model

def get_VL_finetuning_classifier():
    
    image_inputs, text_inputs = layers.Input(shape=(PROJECT_DIM,)), layers.Input(shape=(PROJECT_DIM,))

    image_features = layers.Dense(PROJECT_DIM, activation="tanh")(image_inputs)
    text_features = layers.Dense(PROJECT_DIM, activation="tanh")(text_inputs)
    
    average_features = layers.average([image_features, text_features])
    
    image_average = layers.average([image_features, average_features])
    image_dropout = layers.Dropout(dropout)(image_average)
    image_outputs = layers.Dense(output_classes, activation="softmax", name="Image")(image_dropout)
    
    text_average = layers.average([text_features, average_features])
    text_dropout = layers.Dropout(dropout)(text_average) 
    text_outputs = layers.Dense(output_classes, activation="softmax", name="Text")(text_dropout)
    
    inputs = [image_inputs, text_inputs]
    outputs = [image_outputs, text_outputs]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VL_Modality_Adaptive_Attention")
    return model