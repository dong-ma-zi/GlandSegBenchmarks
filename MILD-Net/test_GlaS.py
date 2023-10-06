import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage import measure, io
from models.mildNet import MILDNet
import utils
from metrics import ObjectHausdorff, ObjectDice, ObjectF1score
# from vis import draw_overlay_rand
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description="Train MILD-Net Model")
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers to load images')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=10, help='epoch to save checkpoints')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_dir', type=str, default='./experiments')
parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Annotation')
parser.add_argument('--model_path', type=str, default="/home/data1/my/Project/GlandSegBenchmark/MILD-Net/experiments/GlaS/140/checkpoints/checkpoint_140.pth.tar")
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[2,], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=100, help='minimum area for an object')
parser.add_argument('--radius', type=int, default=4)
args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    save_dir = "%s/%s" % (args.save_dir, args.dataset)
    model_path = args.model_path
    save_flag = True
    tta = False

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # data transforms
    test_transform = transforms.Compose([transforms.ToTensor()])

    # load model
    model = MILDNet(n_class=2, split='test')

    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)

    model.load_state_dict(best_checkpoint['state_dict'])
    epoch = best_checkpoint['epoch']
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area
    accumulated_metrics = np.zeros(11)
    all_results = dict()

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}/'.format(args.save_dir, args.dataset, strs[-1])
        seg_folder = '{:s}/{:s}/{:s}_segmentation'.format(args.save_dir, args.dataset, strs[-1])

        before_contour_folder = '{:s}/{:s}/contour-unaware'.format(args.save_dir, args.dataset)
        seg_contour_folder = '{:s}/{:s}/contour'.format(args.save_dir, args.dataset)
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)

        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        # img_overlay = Image.open(os.path.join('/home/data2/MedImg/GlandSeg/GlaS/test/Overlay/', name + '.png')).convert("RGB")

        if eval_flag:
            ## GlaS
            label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
            ## CRAG
            # label_path = '{:s}/{:s}.png'.format(label_dir, name)
            label_img = io.imread(label_path)

        input = test_transform(img).unsqueeze(0)

        print('\tComputing output probability maps...')
        prob_o_maps, prob_c_maps = get_probmaps(input, model)

        ## RTS model
        if tta:
            img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_hf = test_transform(img_hf).unsqueeze(0)  # horizontal flip input
            input_vf = test_transform(img_vf).unsqueeze(0)  # vertical flip input
            input_hvf = test_transform(img_hvf).unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_hf_o, prob_maps_hf_c = get_probmaps(input_hf, model)
            prob_maps_vf_o, prob_maps_vf_c = get_probmaps(input_vf, model)
            prob_maps_hvf_o, prob_maps_hvf_c = get_probmaps(input_hvf, model)

            # re flip
            prob_maps_hf_o = np.flip(prob_maps_hf_o, 2)
            prob_maps_vf_o = np.flip(prob_maps_vf_o, 1)
            prob_maps_hvf_o = np.flip(np.flip(prob_maps_hvf_o, 1), 2)
            prob_maps_hf_c = np.flip(prob_maps_hf_c, 2)
            prob_maps_vf_c = np.flip(prob_maps_vf_c, 1)
            prob_maps_hvf_c = np.flip(np.flip(prob_maps_hvf_c, 1), 2)

            # rotation 90 and flips
            img_r90 = img.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_r90 = test_transform(img_r90).unsqueeze(0)
            input_r90_hf = test_transform(img_r90_hf).unsqueeze(0)  # horizontal flip input
            input_r90_vf = test_transform(img_r90_vf).unsqueeze(0)  # vertical flip input
            input_r90_hvf = test_transform(img_r90_hvf).unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_r90_o, prob_maps_r90_c = get_probmaps(input_r90, model)
            prob_maps_r90_hf_o, prob_maps_r90_hf_c = get_probmaps(input_r90_hf, model)
            prob_maps_r90_vf_o, prob_maps_r90_vf_c = get_probmaps(input_r90_vf, model)
            prob_maps_r90_hvf_o, prob_maps_r90_hvf_c = get_probmaps(input_r90_hvf, model)

            # re flip
            prob_maps_r90_o = np.rot90(prob_maps_r90_o, k=3, axes=(1, 2))
            prob_maps_r90_hf_o = np.rot90(np.flip(prob_maps_r90_hf_o, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf_o = np.rot90(np.flip(prob_maps_r90_vf_o, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf_o = np.rot90(np.flip(np.flip(prob_maps_r90_hvf_o, 1), 2), k=3, axes=(1, 2))
            prob_maps_r90_c = np.rot90(prob_maps_r90_c, k=3, axes=(1, 2))
            prob_maps_r90_hf_c = np.rot90(np.flip(prob_maps_r90_hf_c, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf_c = np.rot90(np.flip(prob_maps_r90_vf_c, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf_c = np.rot90(np.flip(np.flip(prob_maps_r90_hvf_c, 1), 2), k=3, axes=(1, 2))


            prob_o_maps = (prob_o_maps + prob_maps_hf_o + prob_maps_vf_o + prob_maps_hvf_o
                         + prob_maps_r90_o + prob_maps_r90_hf_o + prob_maps_r90_vf_o + prob_maps_r90_hvf_o) / 8
            prob_c_maps = (prob_c_maps + prob_maps_hf_c + prob_maps_vf_c + prob_maps_hvf_c
                         + prob_maps_r90_c + prob_maps_r90_hf_c + prob_maps_r90_vf_c + prob_maps_r90_hvf_c) / 8

        #pred_o = np.argmax(prob_o_maps, axis=0)
        #pred_c = np.argmax(prob_c_maps, axis=0)
        pred_o = np.where(prob_o_maps > 0.5, 1, 0)
        pred_c = np.where(prob_c_maps > 0.1, 1, 0)
        pred = np.where(prob_o_maps > 0.5, 1, 0) * np.where(prob_c_maps < 0.1, 1, 0)
        pred_inside = pred == 1
        pred2 = morph.remove_small_objects(pred_inside, args.min_area)  # remove small object

        # if 'scale' in opt.transform['test']:
        #     pred2 = Image.fromarray(pred2.astype(np.uint8) * 255).resize((ori_w, ori_h), resample=Image.BILINEAR)
        #     pred2 = np.array(pred2)
        #     # pred2 = misc.imresize(pred2.astype(np.uint8) * 255, (ori_h, ori_w), interp='bilinear')
        #     pred2 = (pred2 > 127.5)


        pred_labeled = measure.label(pred2)   # connected component labeling
        # pred_labeled = morph.dilation(pred_labeled, footprint=morph.disk(args.radius))

        # res_folder = './res_ck'
        # filename = '{:s}/{:s}.png'.format(res_folder, name)
        # io.imsave(filename, pred_labeled)
        if eval_flag:
            print('\tComputing metrics...')
            result = utils.accuracy_pixel_level(np.expand_dims(pred_labeled > 0, 0), np.expand_dims(label_img > 0, 0))
            pixel_accu = result[0]

            # single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img)
            objF1, _, _, _ = ObjectF1score(pred_labeled, label_img)
            objDice = ObjectDice(pred_labeled, label_img)
            objHaus = ObjectHausdorff(pred_labeled, label_img)
            single_image_result = (objF1, objDice, objHaus)
            accumulated_metrics += utils.gland_accuracy_object_level_all_images(pred_labeled, label_img)
            all_results[name] = tuple([pixel_accu, *single_image_result])

            # 打印每张test的指标
            print('Pixel Acc: {r[0]:.4f}\n'
                  'F1: {r[1]:.4f}\n'
                  'dice: {r[2]:.4f}\n'
                  'haus: {r[3]:.4f}'.format(r=[pixel_accu, objF1, objDice, objHaus]))

        # save image
        if save_flag:
            print('\tSaving image results...')
            final_pred = pred_labeled.astype(np.uint8) * 100
            before_pred = pred_o.astype(np.uint8) * 255
            contour_pred = pred_c.astype(np.uint8) * 255
            final_pred = Image.fromarray(np.concatenate([final_pred, before_pred, contour_pred], axis=1))
            final_pred.save('{:s}/{:s}_seg.jpg'.format(seg_folder, name))

            # save colored objects
            # blank = np.ones(shape=(ori_h, 5, 3)) * 255
            # pred_overlay = draw_overlay_rand(np.array(img), pred_labeled)
            # pred_colored = np.zeros((ori_h, ori_w, 3))
            # for k in range(1, pred_labeled.max() + 1):
            #     pred_colored[pred_labeled == k, :] = np.array(utils.get_random_color())
            # pred_colored = (pred_colored * 255).astype(np.uint8)
            # # filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            # # io.imsave(filename, pred_colored)

            # img_overlay = np.array(img_overlay)
            # conbine_img = np.concatenate([img_overlay, blank, pred_overlay], axis=1)
            # filename = '{:s}/{:s}_seg_concat.png'.format(seg_folder, name)
            # io.imsave(filename, (conbine_img).astype(np.uint8))

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    avg_pq = []
    avg_f1 = []
    avg_dice = []
    avg_haus = []
    for name in all_results:
        pq, f1, dice, haus = all_results[name]
        avg_pq += [pq]
        avg_f1 += [f1]
        avg_dice += [dice]
        avg_haus += [haus]
    avg_pq = np.nanmean(avg_pq)
    avg_f1 = np.nanmean(avg_f1)
    avg_dice = np.nanmean(avg_dice)
    avg_haus = np.nanmean(avg_haus)
    header = ['pixel_acc', 'objF1', 'objDice', 'objHaus']
    save_results(header, [avg_pq, avg_f1, avg_dice, avg_haus], all_results,
                 '{:s}/test_result_epoch{}.txt'.format(save_dir, epoch))

    TP, FP, FN, dice_g, dice_s, iou_g, iou_s, hausdorff_g, hausdorff_s, \
    gt_objs_area, pred_objs_area = accumulated_metrics

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    dice = (dice_g / gt_objs_area + dice_s / pred_objs_area) / 2
    iou = (iou_g / gt_objs_area + iou_s / pred_objs_area) / 2
    haus = (hausdorff_g / gt_objs_area + hausdorff_s / pred_objs_area) / 2

    avg_pixel_accu = -1
    avg_results = [avg_pixel_accu, recall, precision, F1, dice, iou, haus]

    print('=> Processed all {:d} images'.format(counter))
    if eval_flag:
        print('Average of all images:\n'
              'recall: {r[1]:.4f}\n'
              'precision: {r[2]:.4f}\n'
              'F1: {r[3]:.4f}\n'
              'dice: {r[4]:.4f}\n'
              'iou: {r[5]:.4f}\n'
              'haus: {r[6]:.4f}'.format(r=avg_results))

        strs = img_dir.split('/')
        header = ['pixel_acc','recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
        save_results(header, avg_results, all_results,
                     '{:s}/{:s}_test_result_ck_epoch{}.txt'.format(save_dir, strs[-1], epoch))


def get_probmaps(input, model):
    with torch.no_grad():
        o_output, _, c_output, _ = model(input.cuda())

    prob_o_maps = F.softmax(o_output, dim=1).cpu().numpy()
    prob_c_maps = F.softmax(c_output, dim=1).cpu().numpy()
    # pred_o = np.argmax(prob_o_maps, axis=0)
    # pred_c = np.argmax(prob_c_maps, axis=0)
    # pred = np.where(pred_o > 0.5, 1, 0) * np.where(pred_c < 0.5, 1, 0)
    return prob_o_maps[0, 1, :, :], prob_c_maps[0, 1, :, :]


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    main()
