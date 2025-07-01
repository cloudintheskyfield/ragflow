#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import copy
import re
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper


def build_post_process(config, global_config=None):
    support_dict = {'DBPostProcess': DBPostProcess, 'CTCLabelDecode': CTCLabelDecode}

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    module_class = support_dict.get(module_name)
    if module_class is None:
        raise ValueError(
            'post process only support {}'.format(list(support_dict)))
    return module_class(**config)


class DBPostProcess:
    """
    The post process for Differentiable Binarization (DB).
    微分二值化（Differentiable Binarization）：文本检测任务的技术，设计了一个可微分的二值化模块，继承到深度神经网络中
        传统文本检测：二值化操作（将图像转换为只有黑白两种两种颜色的过程）不可微，使得模型训练时难以进行端到端的优化。

        关键步骤：
            1. 特征提取：卷积（ResNet、MobileNet）从输入图像中提起特征图
            2. 概率图生成：通过卷积层生成概率图，概率图中每个像素值表示该位置属于文本区域的概率
            3. 阈值图生成：通过卷积层生成阈值图，该图为每个像素提供自适应的二值化阈值
            4. 可微分二值化：借助提出的可微分二值化函数，结合概率图和阈值图生成近似二值化图。 函数可微可以在反向传播更新网络参数
            5. 文本框提取，基于近视二值化图提取文本框

    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}  二值图中提取四边形文本框
        '''

        bitmap = _bitmap
        height, width = bitmap.shape
        # 从二值图中提取所有文本区域的轮廓 RETR_LIST为不构建层次关系 CHAIN_APPROX_SIMPLE压缩轮廓点 True -> 255 True 0 False. 无符号8位整数. *255后得到OpenCV接受的标准的二值图（像素值0或255）
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]  # _：包含每个轮廓与其他轮廓的关系信息（Next、Previous、FirstChild、Parent）

        num_contours = min(len(contours), self.max_candidates)  # 返回的轮廓已按面积排序、因此重要的文本区域在前侧、限制处理的轮廓数量

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":  # 快速评分、使用文本框的外界矩形进行评分、计算速度快、对倾斜或不规则文本、可能导致不完全精确
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)  # 文本的精确多边形轮廓进行评分
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)   # sside: 文本框的最短边长度
            if sside < self.min_size + 2:  # 删除过小可能为噪声的文本框 个工程上的容差
                continue
            box = np.array(box)
            # 0: 裁剪下限、 dest_width：裁剪上限。 x坐标归一化
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):  # 将初步检测到的文本框向外扩展、确保能完整的包围整个区域、可能导致初始轮廓比实际文本区域小 会切掉字符的边缘
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length  # 自适应扩展策略、根据文本的面积和周长来扩展距离
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # 偏移器创建
        expanded = np.array(offset.Execute(distance))  # 将原始多边形的每条边向外平移distance的距离、从而生成一个更大的新多边形
        return expanded

    def get_mini_boxes(self, contour):  # # 从不规则轮廓中提取标准化的四边形边界框  Counter
        bounding_box = cv2.minAreaRect(contour)  # 包含轮廓的最小面积矩形
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])  # 获取矩形的四个顶点

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:  # 确定左侧两点中哪个是左上角，哪个是左下角
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:  # 确定右侧两点中哪个是右上角，哪个是右下角
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        # 左上→右上→右下→左下（顺时针方向）
        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]  # 提取概率
        segmentation = pred > self.thresh  # 分割图、浮点型、布尔型或其他

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(  # 膨胀 图像中的白色区域变大 分割结果边缘太细 膨胀一下更明显
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.box_type == 'poly':  # 生成不规则的多边形边界框 适合弯曲文本
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          mask, src_w, src_h)
            elif self.box_type == 'quad':  # quad生成四边形边界框 适合正常垂直文本
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                       src_w, src_h)
            else:
                raise ValueError(
                    "box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({'points': boxes})
        return boxes_batch


class BaseRecLabelDecode:
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index
        Connectionist Temporal Classification 连接时序分类，一种深度学习中处理序列数据的损失函数与解码算法，主要
            用于解决输入序列和输出序列在时序上难以对齐的问题，常用于语音识别、光学字符识别（OCR）。

        语音识别中：输入音频信号是连续的长序列，而输出的文本是离散的单词或字符序列，难以预先知道每个字符对应音频中
            的具体位置； OCR任务中，输入的图像里文字的位置和数量也不固定，难以将图像特征和文本字符一一对应，CTC
            能有效处理这类输入输出序列长度不一致、对齐困难的问题。

        损失函数：CTC损失函数用于衡量模型预测结果与真实标签之间的差异，通过引入一个特殊的“空白”（blank）符号，允许模型
            在预测时跳过部分输入帧率，从而解决输入输出序列长度不匹配的问题。训练时，模型会尝试最小化CTC，来学习输入序列
            到输出序列的映射关系
        解码算法：训练完成后，需要将模型输出的概率分布转换为最终的标签序列。常见的CTC解码有贪心（Greedy Decoding）和
            束搜索解码（Beam Search Decoding）
     """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
