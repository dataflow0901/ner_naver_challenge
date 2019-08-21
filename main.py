# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import argparse
import random
import os
from model import Model, ConvModel
from dataset_batch import Dataset
from data_loader import data_loader
from data_loader import test_data_loader
from evaluation import diff_model_label
from evaluation import calculation_measure
from config import Config
from tensorflow.python.tools import inspect_checkpoint as chkp


def iteration_model(model, dataset, parameter, train=True):
    precision_count = np.array([0., 0.])
    recall_count = np.array([0., 0.])

    # 학습
    avg_cost = 0.0
    avg_correct = 0.0
    total_labels = 0.0

    for morph, ne_dict, character, seq_len, char_len, label, step in dataset.get_data_batch_size(parameter["batch_size"], train):
        feed_dict = {model.morph: morph,
                     model.ne_dict: ne_dict,
                     model.character: character,
                     model.sequence: seq_len,
                     model.character_len: char_len,
                     model.label: label,
                     model.dropout_rate: parameter["keep_prob"],
                     model.weight_dropout_keep_prob: parameter["weight_keep_prob"],
                     model.lstm_dropout_keep_prob: parameter["lstm_keep_prob"],
                     model.emb_dropout_keep_prob: parameter["emb_keep_prob"],
                     model.dense_dropout_keep_prob: parameter["dense_keep_prob"],
                     model.learning_rate: parameter["learning_rate"]
                     }

        if train:
            cost, tf_viterbi_sequence, _ = sess.run([model.cost, model.viterbi_sequence, model.train_op],
                                                    feed_dict=feed_dict)
        else:
            cost, tf_viterbi_sequence = sess.run(
                [model.cost, model.viterbi_sequence], feed_dict=feed_dict)
        avg_cost += cost

        mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                np.expand_dims(seq_len, axis=1))
        total_labels += np.sum(seq_len)

        correct_labels = np.sum((label == tf_viterbi_sequence) * mask)
        avg_correct += correct_labels
        precision_count, recall_count = diff_model_label(dataset, precision_count, recall_count, tf_viterbi_sequence,
                                                         label, seq_len)
        if train and step % 5 == 0:
            print('[Train step: {:>4}] cost = {:>.9} Accuracy = {:>.6}'.format(step + 1, avg_cost / (step + 1),
                                                                               100.0 * avg_correct / float(
                                                                                   total_labels)))
        else:
            if step % 5 == 0:
                print('[Dev step: {:>4}] cost = {:>.9} Accuracy = {:>.6}'.format(step + 1, avg_cost / (step + 1),
                                                                                 100.0 * avg_correct / float(
                                                                                     total_labels)))

        if step > 5:
            break

    return avg_cost / (step + 1), 100.0 * avg_correct / float(total_labels), precision_count, recall_count

def iteration_model_Test(dataset, parameter,param, train=True):
    precision_count = np.array([0., 0.])
    recall_count = np.array([0., 0.])

    # 학습
    avg_cost = 0.0
    avg_correct = 0.0
    total_labels = 0.0

    for morph, ne_dict, character, seq_len, char_len, label, step in dataset.get_data_batch_size(parameter["batch_size"], train):
        feed_dict = {param["morph"]: morph,
                     param["ne_dict"]: ne_dict,
                     param["character"]: character,
                     param["sequence"]: seq_len,
                     param["character_len"]: char_len,
                     param["label"]: label,
                     param["dropout_rate"]: parameter["keep_prob"],
                     param["weight_dropout_keep_prob"]: parameter["weight_keep_prob"],
                     param["lstm_dropout"]: parameter["lstm_keep_prob"],
                     param["emb_dropout_keep_prob"]: parameter["emb_keep_prob"],
                     param["dense_dropout_keep_prob"]: parameter["dense_keep_prob"],
                     param["learning_rate"]: parameter["learning_rate"]
                     }

        if train:
            cost, tf_viterbi_sequence, _ = sess.run([param["cost"], param["viterbi_sequence"], param["train_op"]],
                                                    feed_dict=feed_dict)
        else:
            cost, tf_viterbi_sequence = sess.run(
                [param["cost"], param["viterbi_sequence"]], feed_dict=feed_dict)
        avg_cost += cost

        mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                np.expand_dims(seq_len, axis=1))
        total_labels += np.sum(seq_len)

        correct_labels = np.sum((label == tf_viterbi_sequence) * mask)
        avg_correct += correct_labels
        precision_count, recall_count = diff_model_label(dataset, precision_count, recall_count, tf_viterbi_sequence,
                                                         label, seq_len)
        if train and step % 5 == 0:
            print('[Train step: {:>4}] cost = {:>.9} Accuracy = {:>.6}'.format(step + 1, avg_cost / (step + 1),
                                                                               100.0 * avg_correct / float(
                                                                                   total_labels)))
        else:
            if step % 5 == 0:
                print('[Dev step: {:>4}] cost = {:>.9} Accuracy = {:>.6}'.format(step + 1, avg_cost / (step + 1),
                                                                                 100.0 * avg_correct / float(
                                                                                     total_labels)))

        if step > 20:
            break

    return avg_cost / (step + 1), 100.0 * avg_correct / float(total_labels), precision_count, recall_count

def NER_test():
    tf.reset_default_graph()
    config = Config()
    parser = argparse.ArgumentParser(description=sys.argv[0] + " description")
    parser = config.parse_arg(parser)
    print("[NER_test...]")

    try:
        parameter = vars(parser.parse_args())
    except:
        parser.print_help()
        sys.exit(0)

    parameter["mode"] = "test"

    # data_loader를 이용해서 전체 데이터셋 가져옴
    DATASET_PATH = './data' #test data path

    extern_data = []

    if parameter["mode"] == "test":
        extern_data = test_data_loader(DATASET_PATH)
    # 가져온 문장별 데이터셋을 이용해서 각종 정보 및 학습셋 구성

    dataset = Dataset(parameter, extern_data)
    # dev_dataset = Dataset(parameter, extern_data)
    # # Model 불러오기
    #     # if parameter["use_conv_model"]:
    #     #     model = ConvModel(dataset.parameter)
    #     #     print("[Use Conv with lstm...]")
    #     # else:
    #     #     model = Model(dataset.parameter)
    #     #     print("[Use original lstm...]")
    #     # #
    #     # # model.build_model_test()

    dev_size = config.dev_size
    #train_extern_data, dev_extern_data = extern_data[:-dev_size], extern_data[-dev_size:]
    # dataset.make_input_data(train_extern_data)
    #dev_dataset.make_input_data(extern_data)  # For Test set


    best_dev_f1 = 0
    cur_patience = 0
    # use_lr_decay = False

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('./saved/testmodel.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./saved/'))
        graph = tf.get_default_graph()
        # print all tensors in checkpoint file
        #chkp.print_tensors_in_checkpoint_file("./saved/checkpoint.ckpt", tensor_name='', all_tensors=True)

        # placeholder
        param = {}
        param["morph"] = graph.get_tensor_by_name('morph')
        param["ne_dict"]  = graph.get_tensor_by_name('ne_dict')
        param["character"]  = graph.get_tensor_by_name('character')
        param["dropout_rate"]  = graph.get_tensor_by_name('dropout_rate')
        param["weight_dropout_keep_prob"]  = graph.get_tensor_by_name('weight_dropout_keep_prob')
        param["lstm_dropout"]  = graph.get_tensor_by_name('lstm_dropout')
        param["label"] = graph.get_tensor_by_name('label')
        param["sequence"]  = graph.get_tensor_by_name('sequence')
        param["character_len"]  = graph.get_tensor_by_name('character_len')
        param["global_step"]  = graph.get_tensor_by_name('global_step')
        param["emb_dropout_keep_prob"]  = graph.get_tensor_by_name('emb_dropout_keep_prob')
        param["dense_dropout_keep_prob"]  = graph.get_tensor_by_name('dense_dropout_keep_prob')
        param["learning_rate"]  = graph.get_tensor_by_name('learning_rate')


        #tensor   model.cost, model.viterbi_sequence, model.train_op
        param["cost"] = graph.get_tensor_by_name('cost')
        param["viterbi_sequence"] = graph.get_tensor_by_name('viterbi_sequence')
        param["train_op"] = graph.get_tensor_by_name('train_op')

    #for epoch in range(parameter["epochs"]):
    #random.shuffle(extern_data)  # 항상 train set shuffle 시켜주자
        dataset.make_input_data(extern_data)

        # Check for test set
        de_avg_cost, de_avg_correct, de_precision_count, de_recall_count = iteration_model_Test( dataset, parameter,param,
                                                                                           train=False)
        print('[Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, de_avg_cost, de_avg_correct))
        de_f1Measure, de_precision, de_recall = calculation_measure(de_precision_count, de_recall_count)
        print('[Test] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(de_f1Measure, de_precision, de_recall))

        # # Early stopping
        # if best_dev_f1 < de_f1Measure:
        #     best_dev_f1 = de_f1Measure
        #     cur_patience = 0  # Make current patience into zero
        # else:
        #     # Check whether current patience came to limit
        #     if config.patience == cur_patience:
        #         break
        #     else:
        #         cur_patience += 1

        # lr_decay
        # parameter["learning_rate"] = parameter["learning_rate"] * config.lr_decay

if __name__ == '__main__':
    tf.reset_default_graph()
    config = Config()
    parser = argparse.ArgumentParser(description=sys.argv[0] + " description")
    parser = config.parse_arg(parser)

    try:
        parameter = vars(parser.parse_args())
    except:
        parser.print_help()
        sys.exit(0)

    # data_loader를 이용해서 전체 데이터셋 가져옴
    DATASET_PATH = './data'

    extern_data = []

    # 가져온 문장별 데이터셋을 이용해서 각종 정보 및 학습셋 구성
    if parameter["mode"] == "train" and not os.path.exists(parameter["necessary_file"]):
        extern_data = data_loader(DATASET_PATH)
    dataset = Dataset(parameter, extern_data)
    dev_dataset = Dataset(parameter, extern_data)

    # Model 불러오기
    if parameter["use_conv_model"]:
        model = ConvModel(dataset.parameter)
        print("[Use Conv with lstm...]")
    else:
        model = Model(dataset.parameter)
        print("[Use original lstm...]")

    model.build_model()

    # tensorflow session 생성 및 초기화
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # 학습
    if parameter["mode"] == "train":
        extern_data = data_loader(DATASET_PATH)
        # random.shuffle(extern_data)  # shuffle input data
        dev_size = config.dev_size
        train_extern_data, dev_extern_data = extern_data[:-dev_size], extern_data[-dev_size:]
        # dataset.make_input_data(train_extern_data)
        dev_dataset.make_input_data(dev_extern_data)  # For Dev set

        best_dev_f1 = 0
        cur_patience = 0
        # use_lr_decay = False

        for epoch in range(parameter["epochs"]):
            random.shuffle(train_extern_data)  # 항상 train set shuffle 시켜주자
            dataset.make_input_data(train_extern_data)

            avg_cost, avg_correct, precision_count, recall_count = iteration_model(model, dataset, parameter)
            print('[Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, avg_cost, avg_correct))
            f1Measure, precision, recall = calculation_measure(precision_count, recall_count)
            print('[Train] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(f1Measure, precision, recall))

            # Check for dev set
            de_avg_cost, de_avg_correct, de_precision_count, de_recall_count = iteration_model(model, dev_dataset, parameter, train=False)
            print('[Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, de_avg_cost, de_avg_correct))
            de_f1Measure, de_precision, de_recall = calculation_measure(de_precision_count, de_recall_count)
            print('[Dev] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(de_f1Measure, de_precision, de_recall))

            # Early stopping
            if best_dev_f1 < de_f1Measure:
                best_dev_f1 = de_f1Measure
                cur_patience = 0  # Make current patience into zero
            else:
                # Check whether current patience came to limit
                if config.patience == cur_patience:
                    break
                else:
                    cur_patience += 1

            # lr_decay
            parameter["learning_rate"] = parameter["learning_rate"] * config.lr_decay

        temp = {}
        temp["cost"] = model.cost
        temp["viterbi_sequence"] = model.viterbi_sequence
        temp["train_op"] = model.train_op


        temp["morph"] = model.morph
        temp["ne_dict"] = model.ne_dict
        temp["character"] = model.character
        temp["dropout_rate"] = model.dropout_rate
        temp["weight_dropout_keep_prob"] = model.weight_dropout_keep_prob
        temp["lstm_dropout"] = model.lstm_dropout_keep_prob
        temp["label"] = model.label
        temp["sequence"] = model.sequence
        temp["character_len"] = model.character_len
        temp["emb_dropout_keep_prob"] = model.emb_dropout_keep_prob
        temp["dense_dropout_keep_prob"] = model.dense_dropout_keep_prob
        temp["learning_rate"] = model.learning_rate

        saver.save(sess, './saved/testmodel', temp)
    sess.close()

    NER_test()
