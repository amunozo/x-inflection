import os

# Paths to external tools and data
EXTERNAL_PATH = 'external/'
UD_COMPATIBILITY_PATH = os.path.join(EXTERNAL_PATH, 'ud-compatibility/')
NEURAL_TRANSDUCER_PATH = os.path.join(EXTERNAL_PATH, 'neural-transducer/')
MARRY_SCRIPT = os.path.join(UD_COMPATIBILITY_PATH, 'UD_UM', 'marry.py')
TRAIN_MORPH_SCRIPT = os.path.join(NEURAL_TRANSDUCER_PATH, 'src', 'train.py')
DECODE_MORPH_SCRIPT = os.path.join(NEURAL_TRANSDUCER_PATH, 'src', 'decode.py')

# Default directories
TEMP_DIR = 'temp_dir/'
DATA_DIR = 'data/'
RESULTS_DIR = 'results/'
CONFIG_DIR = 'config/'
LOGS_DIR = 'logs/'
