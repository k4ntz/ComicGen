from pytorch_fid.fid_score import parser, calculate_fid_given_paths
import torch
import cv2
import numpy as np
from surface import guided_filter
import generator as G
import os
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from rtpt import RTPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate(path0, path1):
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths([path0, path1],
                                          args.batch_size,
                                          device,
                                          args.dims)
    return fid_value


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_name, shared_pau, rational=False, v=None, d=None):
    model_path = "train_cartoon/saved_models/"+model_name
    if rational:
        approx_func = 'leaky_relu'
        degrees = d
        version = v
        generator = G.Generator_Rational(shared_pau, approx_func, degrees, version, True).to(device)
    else:
        generator = G.Generator(True).to(device)

    generator.eval()
    model = torch.load(model_path, map_location=device)
    generator.load_state_dict(model['generator'])

    name_list = [name for name in os.listdir(load_folder)]

    with torch.no_grad():
        for name in tqdm(name_list):
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)

            image = cv2.imread(load_path)
            image = resize_crop(image)

            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = np.transpose(batch_image, (2, 0, 1))
            batch_image = torch.from_numpy(np.expand_dims(batch_image, axis=0)).to(device)
            output = generator(batch_image)
            output = guided_filter(batch_image, output, r=1, eps=5e-3, device=device)
            output = output.permute(0, 2, 3, 1)
            output = np.squeeze(output.cpu().numpy())
            output = (output + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)


if __name__=="__main__":
    models = ["1000.pth", "2000.pth", "3000.pth", "4000.pth", "5000.pth", "6000.pth", "7000.pth", "8000.pth",
              "9000.pth", "10000.pth", "11000.pth", "12000.pth", "13000.pth", "14000.pth", "15000.pth", "17000.pth",
              "19000.pth", "21000.pth", "23000.pth", "25000.pth", "30000.pth", "35000.pth", "40000.pth"]
    # rows = ["model_v10_f2c"   , "model_v10_f2p"   , "model_v10_s2c"   , "model_v10_s2p"   ,
    #         "rational_v9_f2c" , "rational_v9_f2p" , "rational_v9_s2c" , "rational_v9_s2p" ,
    #         "rational_v10_f2c", "rational_v10_f2p", "rational_v10_s2c", "rational_v10_s2p",
    #         "rational_v11_f2c", "rational_v11_f2p", "rational_v11_s2c", "rational_v11_s2p",
    #         "rational_v12_f2c", "rational_v12_f2p", "rational_v12_s2c", "rational_v12_s2p",
    #         "rational_v13_f2c", "rational_v13_f2p", "rational_v13_s2c", "rational_v13_s2p",
    #         "rational_v14_f2c", "rational_v14_f2p", "rational_v14_s2c", "rational_v14_s2p",
    #         "rational_v15_f2c", "rational_v15_f2p", "rational_v15_s2c", "rational_v15_s2p",
    #         "rational_v16_f2c", "rational_v16_f2p", "rational_v16_s2c", "rational_v16_s2p",
    #         "rational_v18_f2c", "rational_v18_f2p", "rational_v18_s2c", "rational_v18_s2p",
    #         "rational_v22_f2c", "rational_v22_f2p", "rational_v22_s2c", "rational_v22_s2p",
    #         "rational_v26_f2c", "rational_v26_f2p", "rational_v26_s2c", "rational_v26_s2p",
    #         "rational_v30_f2c", "rational_v30_f2p", "rational_v30_s2c", "rational_v30_s2p"]

    # rows = ["model_v10_f2c", "model_v10_f2p", "model_v10_s2c", "model_v10_s2p",
    #         "rational_v9_f2c", "rational_v9_f2p", "rational_v9_s2c", "rational_v9_s2p",
    #         "rational_v10_f2c", "rational_v10_f2p", "rational_v10_s2c", "rational_v10_s2p",
    #         "rational_v11_f2c", "rational_v11_f2p", "rational_v11_s2c", "rational_v11_s2p",
    #         "rational_v12_f2c", "rational_v12_f2p", "rational_v12_s2c", "rational_v12_s2p",
    #         "rational_v13_f2c", "rational_v13_f2p", "rational_v13_s2c", "rational_v13_s2p",
    #         "rational_v14_f2c", "rational_v14_f2p", "rational_v14_s2c", "rational_v14_s2p",
    #         "rational_v15_f2c", "rational_v15_f2p", "rational_v15_s2c", "rational_v15_s2p",
    #         "rational_v16_f2c", "rational_v16_f2p", "rational_v16_s2c", "rational_v16_s2p"
    #         "rational_v18_f2c", "rational_v18_f2p", "rational_v18_s2c", "rational_v18_s2p",
    #         "rational_v22_f2c", "rational_v22_f2p", "rational_v22_s2c", "rational_v22_s2p",
    #         "rational_v26_f2c", "rational_v26_f2p", "rational_v26_s2c", "rational_v26_s2p",
    #         "rational_v30_f2c", "rational_v30_f2p", "rational_v30_s2c", "rational_v30_s2p"]
    rows = ["rational_v18_f2c", "rational_v18_f2p", "rational_v18_s2c", "rational_v18_s2p",
            "rational_v22_f2c", "rational_v22_f2p", "rational_v22_s2c", "rational_v22_s2p"]

    datas = pd.DataFrame(columns=models, index=rows)

    # settings = (("model_v10/model_", None, False, None, None),
    #             ("rational_v9/model_rational_", False, True, "A", (5,4)),
    #             ("rational_v10/model_rational_", False, True, "B", (5,4)),
    #             ("rational_v11/model_rational_", False, True, "C", (5, 4)),
    #             ("rational_v12/model_rational_", False, True, "D", (5, 4)),
    #             ("rational_v13/model_rational_", True, True, "A", (5, 4)),
    #             ("rational_v14/model_rational_", True, True, "B", (5, 4)),
    #             ("rational_v15/model_rational_", True, True, "C", (5, 4)),
    #             ("rational_v16/model_rational_", True, True, "D", (5, 4)),
    #             ("rational_v18/model_rational_", False, True, "B", (3, 2)),
    #             ("rational_v22/model_rational_", True, True, "B", (3, 2)),
    #             ("rational_v26/model_rational_", False, True, "B", (7, 6)),
    #             ("rational_v30/
    settings = (("rational_v18/model_rational_", False, True, "B", (3, 2)),
                ("rational_v22/model_rational_", True, True, "B", (3, 2)))

    rtpt = RTPT(name_initials='SChen', experiment_name="Evaluation",
                max_iterations=len(settings)*len(models))

    rtpt.start()

    for (model_name, shared_pau, rational, v, d) in settings:
        model_ = model_name.split('/')[0]
        # fids = OrderedDict()
        for m in models:
            # cartoonize face
            cartoonize("validation/photo_face", "validation/cartoonized_face", model_name + m, shared_pau, rational=rational, v=v, d=d)
            fid_face2cartoon = calculate("validation/cartoonized_face", "validation/cartoon_face")
            print(m, model_ + "_f2c", fid_face2cartoon)
            fid_face2photo = calculate("validation/cartoonized_face", "validation/photo_face")
            print(m, model_ + "_f2p", fid_face2photo)

            # cartoonize scenery
            cartoonize("validation/photo_scenery", "validation/cartoonized_scenery", model_name + m, shared_pau, rational=rational, v=v, d=d)
            fid_scenery2cartoon = calculate("validation/cartoonized_scenery", "validation/cartoon_scenery")
            print(m, model_ + "_s2c", fid_scenery2cartoon)
            fid_scenery2photo = calculate("validation/cartoonized_scenery", "validation/photo_scenery")
            print(m, model_+"_s2p", fid_scenery2photo)

            datas[m][model_+"_f2c"] = fid_face2cartoon
            datas[m][model_ + "_f2p"] = fid_face2photo
            datas[m][model_ + "_s2c"] = fid_scenery2cartoon
            datas[m][model_ + "_s2p"] = fid_scenery2photo

            rtpt.step()

        datas.to_excel("evaluation2.xlsx")















# "validation/photo_face", "validation/cartoonized_face"
# "validation/photo_scenery", "validation/cartoonized_scenery"