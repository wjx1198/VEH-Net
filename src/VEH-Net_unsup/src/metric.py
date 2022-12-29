# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
import logging
import string
import random
import cPickle as pkl
import math


def get_iou(item1, item2):
    intersection = 0
    for item in item1:
        if item in item2:
            intersection += 1
    union = len(item1) + len(item2) - intersection
    return intersection * 1.0 / union, intersection


def get_new_iou(item1, item2, l):
    intersection = 0
    for item in item2:
        if item in item1:
            intersection += 1
    union = l + len(item2) - intersection
    return intersection * 1.0 / union, intersection


# def analysis_iou(result,epoch,logging):
#     all_iou = []
#     for content in result:
#         current_iou_list = []
#         choose_index_list = content[3]
#         predict_index = content[5]
#         for person in choose_index_list:
#             person_choose_index = []
#             for jj in range(len(person)):
#                 if person[jj] > 0:
#                     person_choose_index.append(jj)
#             iou = get_iou(person_choose_index,predict_index)
#             current_iou_list.append(iou)
#         all_iou.append(np.max(current_iou_list))
#     logging.info('epoch_num = {:d}, mean_iou = {:f}'.format(epoch, np.mean(all_iou)))


# def analysis_prf(result,epoch,logging):
#     all_precision = []
#     all_recall = []
#     all_f1 = []
#     for content in result:
#         ground_person_choose_list = content[3]
#         sorted_index = content[5]
#         final_predict = sorted_index
#         print final_predict
#         person_precision_list = []
#         person_recall_list = []
#         person_f1_list = []
#         for person_choose in ground_person_choose_list:
#             hit = 0
#             for idx in final_predict:
#                 if person_choose[idx] > 0.0:
#                     hit+=1
#             if len(final_predict) == 0:
#                 precision = 0.0
#                 recall = 0.0
#             else:
#                 precision = hit * 1.0 / len(final_predict)
#                 recall = hit * 1.0 / np.sum(person_choose)
#             if precision == 0.0 and recall == 0.0:
#                 f1 = 0.0
#             else:
#                 f1 = 2 * precision * recall / (precision + recall)
#             person_precision_list.append(precision)
#             person_recall_list.append(recall)
#             person_f1_list.append(f1)
#         all_precision.append(np.max(person_precision_list))
#         all_recall.append(np.max(person_recall_list))
#         all_f1.append(np.max(person_f1_list))
#     logging.info('epoch_num = {:d}, mean_precision = {:f}'.format(epoch,np.mean(all_precision)))
#     logging.info('epoch_num = {:d}, mean_recall= {:f}'.format(epoch,np.mean(all_recall)))
#     logging.info('epoch_num = {:d}, mean_f1 = {:f}'.format(epoch,np.mean(all_f1)))


def analysis_iou(result, epoch, logging):
    all_iou = []
    n = len(result)
    hit1 = 0.0
    hit2 = 0.0
    hit3 = 0.0
    hit4 = 0.0
    hit5 = 0.0
    for content in result:
        current_iou_list = []
        current_intersection_list = []
        choose_index_list = content[3]
        predict_index = content[5]
        for person in choose_index_list:
            person_choose_index = []
            for jj in range(len(person)):
                if person[jj] > 0:
                    person_choose_index.append(jj)
            iou, intersection = get_iou(person_choose_index, predict_index)
            current_iou_list.append(iou)
            current_intersection_list.append(intersection)
        all_iou.append(np.max(current_iou_list))
        if max(current_intersection_list) == 5:
            hit5 = hit5 + 1.0
            hit4 = hit4 + 1.0
            hit3 = hit3 + 1.0
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif max(current_intersection_list) == 4:
            hit4 = hit4 + 1.0
            hit3 = hit3 + 1.0
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif max(current_intersection_list) == 3:
            hit3 = hit3 + 1.0
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif max(current_intersection_list) == 2:
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif max(current_intersection_list) == 1:
            hit1 = hit1 + 1.0
        else:
            continue

    print('forward: epoch_num = {:d}, mean_iou = {:f}'.format(epoch, np.mean(all_iou)))
    logging.info('forward: epoch_num = {:d}, hit5 = {:f}, hit@5 = {:f}'.format(epoch, hit5, hit5 / n))
    logging.info('forward: epoch_num = {:d}, hit4 = {:f}, hit@4 = {:f}'.format(epoch, hit4, hit4 / n))
    logging.info('forward: epoch_num = {:d}, hit3 = {:f}, hit@3 = {:f}'.format(epoch, hit3, hit3 / n))
    logging.info('forward: epoch_num = {:d}, hit2 = {:f}, hit@2 = {:f}'.format(epoch, hit2, hit2 / n))
    logging.info('forward: epoch_num = {:d}, hit1 = {:f}, hit@1 = {:f}'.format(epoch, hit1, hit1 / n))


def analysis_new_iou(result, epoch, logging):
    all_iou = []
    n = len(result)
    hit1 = 0.0
    hit2 = 0.0
    hit3 = 0.0
    hit4 = 0.0
    hit5 = 0.0
    for content in result:
        current_len_list = []
        choose_index_list = content[3]
        predict_index = content[5]
        person_choose_index_all = []
        for person in choose_index_list:
            person_choose_index = []
            for jj in range(len(person)):
                if person[jj] > 0:
                    person_choose_index.append(jj)
                    person_choose_index_all.append(jj)
            current_len_list.append(len(person_choose_index))
        iou, intersection = get_new_iou(person_choose_index_all, predict_index, max(current_len_list))
        all_iou.append(iou)
        # current_iou_list.append(iou)
        # all_iou.append(np.max(current_iou_list))
        if intersection == 5:
            hit5 = hit5 + 1.0
            hit4 = hit4 + 1.0
            hit3 = hit3 + 1.0
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif intersection == 4:
            hit4 = hit4 + 1.0
            hit3 = hit3 + 1.0
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif intersection == 3:
            hit3 = hit3 + 1.0
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif intersection == 2:
            hit2 = hit2 + 1.0
            hit1 = hit1 + 1.0
        elif intersection == 1:
            hit1 = hit1 + 1.0
        else:
            continue
    logging.info('forward: epoch_num = {:d}, mean_new_iou = {:f}'.format(epoch, np.mean(all_iou)))
    logging.info('forward: epoch_num = {:d}, hit@5 = {:f}'.format(epoch, hit5 / n))
    logging.info('forward: epoch_num = {:d}, hit@4 = {:f}'.format(epoch, hit4 / n))
    logging.info('forward: epoch_num = {:d}, hit@3 = {:f}'.format(epoch, hit3 / n))
    logging.info('forward: epoch_num = {:d}, hit@2 = {:f}'.format(epoch, hit2 / n))
    logging.info('forward: epoch_num = {:d}, hit@1 = {:f}'.format(epoch, hit1 / n))


def analysis_prf(result, epoch, logging):
    all_precision = []
    all_recall = []
    all_f1 = []
    for content in result:
        ground_person_choose_list = content[3]
        final_predict = content[5]
        person_precision_list = []
        person_recall_list = []
        person_f1_list = []
        for person_choose in ground_person_choose_list:
            hit = 0
            for idx in final_predict:
                if person_choose[idx] > 0.0:
                    hit += 1
            if len(final_predict) == 0:
                precision = 0.0
                recall = 0.0
            else:
                precision = hit * 1.0 / len(final_predict)
                recall = hit * 1.0 / np.sum(person_choose)
            if precision == 0.0 and recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            person_precision_list.append(precision)
            person_recall_list.append(recall)
            person_f1_list.append(f1)
        all_precision.append(np.max(person_precision_list))
        all_recall.append(np.max(person_recall_list))
        all_f1.append(np.max(person_f1_list))
    print('forward epoch_num = {:d}, mean_precision = {:f}'.format(epoch, np.mean(all_precision)))
    print('forward epoch_num = {:d}, mean_recall= {:f}'.format(epoch, np.mean(all_recall)))
    print('forward epoch_num = {:d}, mean_f1 = {:f}'.format(epoch, np.mean(all_f1)))
