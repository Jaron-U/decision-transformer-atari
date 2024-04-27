import io
import math
import numpy as np

import gzip
import pickle
from PIL import Image
import PIL
from process_data import get_max_timesteps

max_timesteps = get_max_timesteps()
print(f"Max timesteps: {max_timesteps}")

