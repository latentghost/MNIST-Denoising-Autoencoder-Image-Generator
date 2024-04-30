import os
import torch
import random
import argparse
from EncDec import *
from EncDec.autoEncoder import *
from torch.utils.data import DataLoader


P = argparse.ArgumentParser()
P.add_argument("inpDigit", type=int)
A = P.parse_args()
    

if __name__ == "__main__":
    Generator = CVAE_Generator()
    Generator.save_image(digit=A.inpDigit, save_path=SAVEPATH)
