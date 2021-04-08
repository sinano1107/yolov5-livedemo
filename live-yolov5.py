import math
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import torchvision
import numpy as np
from numpy import random
import cv2
from threading import Thread
from pathlib import Path

# モデルのパス
weight = 'yolov5s.pt'
imgSize = 640


class Ensemble(nn.ModuleList):
    # モデルのインセンブル
    def __init__(self):
        super(Ensemble, self).__init__()


def autopad(k, p=None):  # カーネル、パディング
    # パッドから「同じ」へ
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # 標準的なコンボリューション
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def attempt_load(map_location=None):
    # モデルのロード
    print('モデルのロードを試みます')
    model = Ensemble()
    ckpt = torch.load(weight, map_location=map_location)
    model.append(ckpt['ema' if ckpt.get('ema')
                      else 'model'].float().fuse().eval())  # FP32 model

    # pytorch互換の変更
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True    # pytorch 1.7.0 との互換
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()   # pytorch 1.6.0 との互換

    # モデルの重み数が1を想定するため最後の一つを返す
    return model[-1]


def make_divisible(x, divisor):
    # 割り切れる数xを返す
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # img_sizeがストライドの倍数であることを確認する
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print('警告: imgSize %g は %g の倍数でなければならないため、%g に変更して実行します' %
              (img_size, s, new_size))
    return new_size


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # ストライドマルチの制約をクリアしながら、画像のリサイズやパディングを行うことがでいます
    shape = img.shape[:2]   # 現在の形状[高さ, 幅]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # スケール比(new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # スケールダウンのみで、スケールアップはしない
        r = min(r, 1.0)

    # paddingの計算
    ratio = r, r    # 幅, 高さ比
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # whiteパディング
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # ストレッチ
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 幅, 高さ比

    dw /= 2  # paddingを2で割る
    dh /= 2

    if shape[::-1] != new_unpad:  # リサイズ
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # ボーダーを追加
    return img, ratio, (dw, dh)


class LoadStreams:  # カメラ
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            cap = cv2.VideoCapture(eval(s))
            assert cap.isOpened(), f'{s} を開くことができませんでした'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()
            thread = Thread(target=self.update, args=([0, cap]), daemon=True)
            print(f' 成功 ({w}*{h} at {fps:.2f} FPS)')
            thread.start()
        print('')   # newline

        # 共通する形状のチェック
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[
                     0].shape for x in self.imgs], 0)   # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            print('警告: 異なる形状のストリームを検出しました。最適なパフォーマンスを得るためには、同じような形状のストリームを供給してください')

    def update(self, index, cap):
        # デーモンスレッドで次のストリームフレームを読む
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n == 4:  # 4フレームごとに読む
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # 待ち時間

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect,
                         stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        # BGR to RGB, to bsx3x416x416
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0


def time_synchronized():
    # pytorch 正確な時間
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def xywh2xyxy(x):
    # nx4個のボックスを[x, y, w, h]から[x1, y1, x2, y2]（xy1=左上、xy2=右下）に変換する
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    ボックスの交差-結合（Jaccard index）を返します。
    両セットのボックスは(x1, y1, x2, y2)形式であることが期待されます。
    引数:
        box1 (Tensor[N, 4])
        box2 (Tensor[M. 4])
    返り値:
        iou (Tensor[N, M]): box1とbox2の各要素のペアワイズIoU値を含むNxM行列です。
            ボックス1とボックス2の各要素のIoU値を含むNxM行列
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """
    推論結果にNon-Maximum-Suppression(NMS)を実行する

    Returns:
        画像ごとに(n,6)個のテンソルで検出されたリスト [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # クラス数
    xc = prediction[..., 4] > conf_thres  # 候補者

    # 設定
    # (ピクセル) ボックスの最小・最大の幅と高さ
    min_wh, max_wh = 2, 4096
    max_det = 300  # 1画像あたりの最大検出数
    max_nms = 30000  # torchvision.ops.nms()の最大ボックス数
    time_limit = 10.0  # 10秒後に終了します
    redundant = True  # 冗長な検出が必要
    multi_label &= nc > 1  # 1箱に複数のラベルを貼る(0.5ms/img追加)
    merge = False  # マージNMSを使用するか

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # 画像インデックス, 画像推論
        # 制約を適用する
        x = x[xc[xi]]  # 信頼度

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # 残っていなければ次の画像を処理する
        if not x.shape[0]:
            continue

        # 信頼度を計算
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box(center x, center y, width, height) を (x1, y1, x2, y2) に変換
        box = xywh2xyxy(x[:, :4])

        # 検出マトリクス nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], [i,  + 5, None], j[:, None].float()), 1)
        else:  # ベストクラスのみ
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres
            ]

        # クラスで絞り込む
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes,  device=x.device)).any(1)]

        # シェイプをチェック
        n = x.shape[0]  # ボックス数
        if not n:  # ボックスがない
            continue
        elif n > max_nms:  # 余剰ボックス
            # 信頼度でソート
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # バッチ式NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # クラス
        # ボックス（クラスごとにオフセット）、スコア
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # 検出数が限界を超えている場合
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # NMSの統合（加重平均でボックスを統合)
            # boxを更新します。 boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou マトリックス
            weights = iou * scores[None]  # boxの重さ
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # 合わせたボックス
            if redundant:
                i = i[iou.sum(1) > 1]  # 冗長性が必要

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'警告: NMSの制限時間 {time_limit}s を超えました')
            break  # 制限時間超過

    return output


def clip_coords(boxes, img_shape):
    # xyxyのバウンディングボックスを画像の形（高さ、幅）に合わせて切り取る
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # img1_shapeからimg0_shapeへの座標（xyxy）のリスケール
    if ratio_pad is None:  # img0_shapeから計算
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh パディング
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # 画像imgに1つのバウンディングボックスをプロットする
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # フォントの太さ
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # フィル
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect():
    # 推論関数
    print('推論を開始します')

    # 初期化
    device = torch.device('cpu')
    print(device)

    # モデルのロード
    model = attempt_load(map_location=device)
    stride = int(model.stride.max())    # モデル ストライド
    imgsz = check_img_size(imgSize, s=stride)   # img_sizeをチェック

    # データローダの設定
    cudnn.benchmark = True  # 一定の画像サイズの推論を高速化する場合Trueを設定する
    dataset = LoadStreams('0', img_size=imgsz, stride=stride)

    # 名前と色の取得
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 推論の実行
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推論
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # NMSの適用
        pred = non_max_suppression(
            pred,
            0.25,
            0.45,
            classes=None,
            agnostic=False,
        )
        t2 = time_synchronized()

        # プロセス検出
        for i, det in enumerate(pred):  # 画像あたりの検出数
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
            ), dataset.count

            p = Path(p)  # パス
            s += '%g✖︎%g ' % img.shape[2:]  # 表示する文字列
            if len(det):
                # ボックスをimg_sizeからim0サイズにリスケールする
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # リザルトの表示
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # クラスごとの検出数
                    # 文字列を追加
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # リザルトを書き込む
                for *xyxy, conf, cls in reversed(det):
                    # bboxを追加
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)], line_thickness=3)

            # 時間を表示 (inference + NMS)
            print(f'{s}完了！({t2 - t1:.3f}s)')

            # 結果を流す
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

    print(f'完了! ({time.time() - t0:.3f}s)')


# 勾配計算を凍結して実行
with torch.no_grad():
    detect()
