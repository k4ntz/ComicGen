from rational.utils.find_init_weights import find_weights
import torch.nn.functional as F  # To get the tanh function

find_weights(F.leaky_relu)