import torch.nn.functional as F
import skimage.morphology as morph
from skimage.measure import label
from skimage import measure, io
import utils
from metrics import dice_coefficient, iou_metrics
from transforms import ResizeLongestSide
from scipy import ndimage
# import torchvision.transforms as transforms
import argparse
from multiprocessing import Array, Process
from models.sam import SamPredictor, sam_model_registry
import cv2
from utils import *

parser = argparse.ArgumentParser(description="Testing oeem segmentation Model")

parser.add_argument('--save_dir', type=str, default='./experimentsP')

parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Annotation')

# parser.add_argument('--img_dir', type=str, default='/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_cls/2.validation/img')
# parser.add_argument('--label_dir', type=str, default='/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_cls/2.validation/mask')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Images')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Annotation/')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/wzh/valid/480x480/Images/')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/wzh/valid/480x480/Annotation/')

parser.add_argument('--desc', type=str,
                    default='SAM-vit-h-Adapt',

                    )

parser.add_argument('--model_path', type=str,
                    # default="/home/data1/my/Project/GlandSegBenchmark/SAM/experimentsP3/GlaS_Vit-SAM-ft-all/checkpoints/checkpoint_100.pth.tar"
                    default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs/sam-h-1024-16-256_2023_10_25_14_22_04/Model/best_checkpoint_2"
                    )

parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[1, ], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=400, help='minimum area for an object')
# parser.add_argument('--radius', type=int, default=4)
args = parser.parse_args()


def img_preprocessing(image, sam):
    original_image_size = (image.shape[0], image.shape[1])
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(np.array(image))
    input_image = torch.as_tensor(input_image, dtype=torch.float32, device=sam.device) # set to float32
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_size = input_image.shape[-2:]
    # input_image = sam.preprocess(input_image) # do not need padding here
    return input_image, original_image_size, input_size

def get_scaled_prompt(points, sam, original_image_size, if_transform: bool = True):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    points = transform.apply_coords(points, original_image_size) if if_transform else points
    points = torch.as_tensor(points, device=sam.device).unsqueeze(1)
    points = (points, torch.ones(points.shape[0], 1))
    return points

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    save_dir = "%s/%s_%s" % (args.save_dir, args.dataset, args.desc)
    model_path = args.model_path
    save_flag = True

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    epoch = os.path.basename(model_path).split('.')[0].split('_')[-1]
    print("=> loaded model at epoch {}".format(epoch))


    model = sam_model_registry['vit_h'](args).cuda()
    if args.model_path:
        weights = torch.load(args.model_path, map_location='cpu')["state_dict"]
        model.load_state_dict(weights, strict=True)

    # switch to evaluate mode
    model.eval()
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area

    all_results = dict()

    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prob_maps_folder = '{:s}/{:s}_{:s}/{:s}'.format(args.save_dir, args.dataset, args.desc, 'mask_pred')
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        vis_maps_folder = '{:s}/{:s}_{:s}/{:s}'.format(args.save_dir, args.dataset, args.desc, 'vis_pred')
        if not os.path.exists(vis_maps_folder):
            os.mkdir(vis_maps_folder)

    imgsize = 224
    stride = 224
    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        orig_img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        ########################### esenbling from multi-scale #################################

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        h, w, _ = image.shape
        label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
        label_img = np.array(Image.open(label_path))

        padding_h = int(math.ceil(h / stride) * stride)
        padding_w = int(math.ceil(w / stride) * stride)
        padding_output = np.zeros((1, 1, padding_h, padding_w))
        padding_weight_mask = np.zeros_like(padding_output)

        for h_ in range(0, padding_h - stride + 1, int(stride // 1)):
            for w_ in range(0, padding_w - stride + 1, int(stride // 1)):
                img_padding = np.zeros((imgsize, imgsize, 3), np.uint8)
                slice = image[h_:h_ + imgsize, w_:w_ + imgsize, :]
                t_h, t_w, _ = slice.shape
                img_padding[:t_h, :t_w] = slice

                label_padding = np.zeros((imgsize, imgsize), np.uint8)
                slice = label_img[h_:h_ + imgsize, w_:w_ + imgsize]
                t_h, t_w = slice.shape
                label_padding[:t_h, :t_w] = slice

                ''' preprocess '''
                input_image, original_image_size, input_size = img_preprocessing(img_padding, model)
                pts, masks = generate_click_prompt_all_inst(label_padding)
                pt = get_scaled_prompt(pts, model, original_image_size)
                # masks = torch.as_tensor(masks, dtype=torch.float32).cuda()

                ''' forward '''
                # --------------------- get prompt --------------------------#
                with torch.no_grad():
                    imge= model.image_encoder(input_image)
                    se, de = model.prompt_encoder(
                        points=pt,
                        # points=None,
                        boxes=None,
                        masks=None,
                    )

                    output, _ = model.mask_decoder(
                        image_embeddings=imge,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )

                ''' postprocess '''
                # msk
                upscaled_masks = model.postprocess_masks(output, input_size, original_image_size).squeeze(1)
                upscaled_masks = (upscaled_masks > 0.5).float().detach().cpu().numpy()

                res_mask = np.zeros((imgsize, imgsize), np.uint8)
                for i in range(upscaled_masks.shape[0]):
                    res_mask[upscaled_masks[i] == 1] = 1

                padding_output[:, :, h_:h_ + imgsize, w_:w_ + imgsize] += res_mask
                # weight
                weight_mask = np.ones_like(res_mask)
                padding_weight_mask[:, :, h_:h_ + imgsize, w_:w_ + imgsize] += weight_mask
                # del output, img, img_padding, slice

        padding_output /= padding_weight_mask
        pred = padding_output[:, :, :h, :w]
        pred = pred[0][0].astype(int)


        ############################### post proc ################################################
        # 将二值图像转换为标签图像
        label_image = label(pred)
        # 根据标记删除小连通域
        filtered_label_image = morph.remove_small_objects(label_image, min_size=args.min_area)
        # 将标签图像转换回二值图像
        pred = (filtered_label_image > 0).astype(np.uint8)
        # fill holes
        pred = ndimage.binary_fill_holes(pred)
        ################################################################################################ # remove small object

        if eval_flag:


            # label_img = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']
            label_img = cv2.imread('{:s}/{:s}_anno.bmp'.format(label_dir, name))[:, :, 0]
            label_img = np.array(label_img != 0, dtype=np.uint8)

        if eval_flag:
            img_show = np.concatenate([np.array(orig_img),
                                       np.stack((label_img * 255, label_img * 255, label_img * 255), axis=-1),
                                       np.stack((pred * 255, pred * 255, pred * 255), axis=-1)],
                                       axis=1)
            cv2.imwrite('{:s}/{}.png'.format(vis_maps_folder, name), img_show)

            np.save('{:s}/{}.npy'.format(prob_maps_folder, name), pred)
            print('\tComputing metrics...')
            result = utils.accuracy_pixel_level(np.expand_dims(pred > 0, 0), np.expand_dims(label_img > 0, 0))
            pixel_accu = result[0]

            # single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img)
            IoU = iou_metrics(pred, label_img)
            Dice = 2 * IoU / (1 + IoU)

            single_image_result = (IoU, Dice)
            all_results[name] = tuple([pixel_accu, *single_image_result])
            # 打印每张test的指标
            print('Pixel Acc: {r[0]:.4f}\n'
                  'IoU: {r[1]:.4f}\n'
                  'Dice: {r[2]:.4f}'.format(r=[pixel_accu, IoU, Dice]))

    over_all_iou, over_all_f1 = get_overall_valid_score('{:s}'.format(prob_maps_folder), args.label_dir)

    avg_pq = []
    avg_iou = []
    avg_dice = []

    for name in all_results:
        pq, iou, dice = all_results[name]
        avg_pq += [pq]
        avg_iou += [iou]
        avg_dice += [dice]

    avg_pq = np.nanmean(avg_pq)
    avg_iou = np.nanmean(avg_iou)
    avg_dice = np.nanmean(avg_dice)
    header = ['pixel_acc', 'Iou', 'Dice']
    save_results(header, [avg_pq, avg_iou, avg_dice], all_results,
                 f'{save_dir:s}/test_result_epoch{epoch}_overall_iou_{over_all_iou:.4f}_dice_{2 * over_all_iou / (1 + over_all_iou):.4f}_'
                 f'f1score_{over_all_f1:.4f}.txt')




def chunks(lst, num_workers=None, n=None):
    """
    a helper function for seperate the list to chunks

    Args:
        lst (list): the target list
        num_workers (int, optional): Default is None. When num_workers are not None, the function divide the list into num_workers chunks
        n (int, optional): Default is None. When the n is not None, the function divide the list into n length chunks

    Returns:
        llis: a list of small chunk lists
    """
    chunk_list = []
    if num_workers is None and n is None:
        print("the function should at least pass one positional argument")
        exit()
    elif n == None:
        n = int(np.ceil(len(lst)/num_workers))
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list


def get_overall_valid_score(pred_image_path, groundtruth_path, num_workers=1, num_class=2):
    """
    get the scores with validation groundtruth, the background will be masked out
    and return the score for all photos

    Args:
        pred_image_path (str): the prediction require to test, npy format
        groundtruth_path (str): groundtruth images, png format
        num_workers (int): number of process in parallel, default is 5.
        mask_path (str): the white background, png format
        num_class (int): default is 2.

    Returns:
        float: the mIOU score
    """
    image_names = list(map(lambda x: x.split('.')[0], os.listdir(pred_image_path)))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            cam = np.load(os.path.join(pred_image_path, f"{im_name}.npy"), allow_pickle=True).astype(np.uint8).reshape(-1)

            groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}_anno.bmp")).reshape(-1)
            # groundtruth = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['inst_map']
            groundtruth = np.array(groundtruth != 0, dtype=np.uint8).reshape(-1)

            gt_list.extend(groundtruth)
            pred_list.extend(cam)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in range(num_class):
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                fp = sum(np.logical_and(pred == i, real != i))
                fn = sum(np.logical_or(pred != i, real == i))
                intersection[i] += inter
                union[i] += u
                FP[i] += fp
                FN[i] += fn

    intersection = Array("d", [0] * num_class)
    union = Array("d", [0] * num_class)
    FP = Array("d", [0] * num_class)
    FN = Array("d", [0] * num_class)

    p_list = []
    for i in range(len(image_list)):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    total = 0
    total_f1 = 0
    # for i in range(num_class):
    for i in [1]:
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
        if i == 1:
            class_precision = union[i] / (union[i] + FP[i] + eps)
            class_recall = union[i] / (union[i] + FN[i] + eps)
            total_f1 += (2 * class_precision * class_recall) / (class_precision + class_recall + eps)
    # return total / num_class, total_f1
    return total, total_f1


def get_predmaps(input, model):
    with torch.no_grad():
        o_output = model(input.cuda())

    prob_maps = F.softmax(o_output[0], dim=0).cpu().numpy()
    pred = np.argmin(prob_maps, axis=0)

    return pred


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

