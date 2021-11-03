import matplotlib.pyplot as plt
import torch
from torch import nn
# import generator as G
from rational.torch import Rational


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # settings = (("rational_v9/model_rational_", "A", (5, 4), 15000),
    #             ("rational_v10/model_rational_", "B", (5, 4), 8000),
    #             ("rational_v11/model_rational_", "C", (5, 4), 13000),
    #             ("rational_v12/model_rational_", "D", (5, 4), 12000),
    #             ("rational_v13/model_rational_", "A", (5, 4), 12000),
    #             ("rational_v14/model_rational_", "B", (5, 4), 8000),
    #             ("rational_v15/model_rational_", "C", (5, 4), 14000),
    #             ("rational_v16/model_rational_", "D", (5, 4), 12000),
    #             ("rational_v18/model_rational_", "B", (3, 2), 13000),
    #             ("rational_v22/model_rational_", "B", (3, 2), 11000))
    settings = (("rational_v26/model_rational_", "B", (7, 6), 13000),
                ("rational_v30/model_rational_", "B", (7, 6), 40000))

    # cartoonize face

    for (model_path, version, degrees, iter) in settings:

        print(model_path)

        model_name = model_path.split("/")[0]
        path = f"train_cartoon/saved_models/{model_path}{iter}.pth"

        model = torch.load(path, map_location=device)

        approx_func = 'leaky_relu'

        dictionaries = {}

        for k, v in model['generator'].items():
            if "numerator" in k or "denominator" in k:
                if not "activation" in k:
                    layer, i, x = k.split(".")
                    layer = layer + "." + i
                    if layer in dictionaries:
                        dictionaries[layer].append({x: v})
                    else:
                        dictionaries[layer] = [{x: v}]

    # print(dictionaries)

        value = torch.arange(-1, 1, 0.1).to(device)
        lrelu = nn.LeakyReLU().to(device)

        for k, v in dictionaries.items():
            numerator   = dictionaries[k][0]['numerator']
            denominator = dictionaries[k][1]['denominator']
            rational_function = Rational(approx_func=approx_func, degrees=degrees, version=version)
            rational_function.numerator = nn.Parameter(numerator.to(device))
            rational_function.denominator = nn.Parameter(denominator.to(device))

            plt.plot(value.cpu(), rational_function(value).detach().cpu(), label="rational")
            plt.plot(value.cpu(), lrelu(value).cpu(), label="leaky_relu")

            plt.legend()
            plt.grid()

            plt.savefig("plots/"+model_name+"_"+k+".png")
            plt.close()


