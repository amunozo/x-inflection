from main import train_morph, unimorph_training_data
import argparse

parser = argparse.ArgumentParser(description='Train a morphological inflexion system using UniMorph data.')
parser.add_argument('--lang', help='Language 3-letters code as it appears in UniMorph', required='True')
parser.add_argument('--dir', help='Directory of the folder where the data and the model are being stored', required='True')
args = parser.parse_args()

if __name__ == '__main__':
    unimorph_training_data(lang=args.lang, data_dir=args.dir)
    train_morph(lang=args.lang, dir=args.dir)