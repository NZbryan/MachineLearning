import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import numpy as np
import re
import random
import math

class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output
    

def tokenize_text(text_input):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_input))


if __name__ == '__main__': 
    
    # hyper parameters
    BATCH_SIZE = 32
    EMB_DIM = 200
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 10
    DROPOUT_RATE = 0.2
    NB_EPOCHS = 5
    
    # raw data
    df_raw = pd.read_csv("data.txt",sep="\t",header=None,names=["text","label"])
    
    # label
    df_label = pd.DataFrame({"label":["财经","房产","股票","教育","科技","社会","时政","体育","游戏","娱乐"],"y":list(range(10))})
    df_raw = pd.merge(df_raw,df_label,on="label",how="left")
    y = np.array(df_raw["y"].tolist())
    
    # Creating a BERT Tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("bert_zh_L-12_H-768_A-12_2",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    
    # Tokenize all the text
    text = []
    sentences = list(df_raw['text'])
    for sen in sentences:
        text.append(sen)
    tokenized_text = [tokenize_text(i) for i in text]
    
    # Prerparing Data For Training
    text_with_len = [[text, y[i], len(text)]
                 for i, text in enumerate(tokenized_text)]
    random.shuffle(text_with_len)
    text_with_len.sort(key=lambda x: x[2])
    sorted_text_labels = [(text_lab[0], text_lab[1]) for text_lab in text_with_len]
    
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_text_labels, output_types=(tf.int32, tf.int32))
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
    
    TOTAL_BATCHES = math.ceil(len(sorted_text_labels) / BATCH_SIZE)
    TEST_BATCHES = TOTAL_BATCHES // 10
    batched_dataset.shuffle(TOTAL_BATCHES)
    test_data = batched_dataset.take(TEST_BATCHES)
    train_data = batched_dataset.skip(TEST_BATCHES)
    
    VOCAB_LENGTH = len(tokenizer.vocab)
    text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                            embedding_dimensions=EMB_DIM,
                            cnn_filters=CNN_FILTERS,
                            dnn_units=DNN_UNITS,
                            model_output_classes=OUTPUT_CLASSES,
                            dropout_rate=DROPOUT_RATE)
    
    if OUTPUT_CLASSES == 2:
        text_model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])
    else:
        text_model.compile(loss="sparse_categorical_crossentropy",
                           optimizer="adam",
                           metrics=["sparse_categorical_accuracy"])
        
    text_model.fit(train_data, epochs=NB_EPOCHS)
    # test test data
    results = text_model.evaluate(test_data)
    print(results)
