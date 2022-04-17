from main import train_morph, unimorph_training_data, ud_morph_data
import sys


lang = sys.argv[1]
arch = sys.argv[2]
source = sys.argv[3]
split = sys.argv[4]

if source == 'ud':
    treebank = sys.argv[4]
    ud_morph_data(treebank, lang)
    try:
        train_morph(lang, arch, source)
    except:
        pass

elif source == 'um':
    if split == 'lemma':
        unimorph_training_data(lang, split)
        train_morph(lang, arch, 'um_lemma')

    elif split == 'form':
        unimorph_training_data(lang, split)
        train_morph(lang, arch, source)