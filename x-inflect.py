from main import ud_to_um, morph_translation
import argparse

parser = argparse.ArgumentParser(description='Cross-inflect a UD treebank using a seq2seq model trained on UM data')
parser.add_argument('--treebank_folder', help='Directory where the UD treebank files are located', required=True)
parser.add_argument('--post_lang', help='Language dependent postprocessing to improve the feature conversion', required=False, default='')

parser.add_argument('--model_folder', help='Directory where the inflection model is located', required=True)
parser.add_argument('--output', help='Output location of the cross-inflected treebank', required=True)
parser.add_argument('--pos_types', help='POS tags that are cross-inflected', required=False, default=['ADJ','NOUN','VERB'])
args = parser.parse_args()

if __name__ == '__main__':
    ud_to_um(args.post_lang, args.treebank_folder)
    morph_translation(args.treebank_folder, args.model_folder, args.pos_types, args.output, args.post_lang)