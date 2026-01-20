import os
import random
import src.const as const


# This script trains a model using dep2label and the UD corpus selected using the desired encoding


def ud_to_um(
    post_lang,
    treebank_folder,
):
    """
    Convert UD features to UM. The default output is located in the treebank folder, changing -ud- for -um-.
    """
    # Find the ud-conversion script
    ud_converter = const.MARRY_SCRIPT
    dev_ud_conllu = train_ud_conllu = test_ud_conllu = None

    for filename in os.listdir(treebank_folder):
        if filename.endswith('-ud-dev.conllu'):
            dev_ud_conllu = os.path.join(treebank_folder, filename)
        if filename.endswith('-ud-train.conllu'):
            train_ud_conllu = os.path.join(treebank_folder, filename)
        if filename.endswith('-ud-test.conllu'):
            test_ud_conllu = os.path.join(treebank_folder, filename)
    # Convert both files to unimorph format

    for conllu_file in [dev_ud_conllu, train_ud_conllu, test_ud_conllu]:
        if conllu_file:
            post_lang_arg = f"-l {post_lang}" if post_lang else ""
            os.system(f'python {ud_converter} convert --ud "{conllu_file}" {post_lang_arg}')

def morph_translation(
    treebank_folder, 
    model_folder, 
    pos_types,
    output,
    post_lang='', 
    decode='greedy'
):
    """
    Inflect the lemmas of a UD Treebank using a morphological analyzer trained in a different language to 'translate' the treebank.
    """
    # Find the ud-conversion script
    ud_converter = const.MARRY_SCRIPT
    dev_ud_conllu = train_ud_conllu = test_ud_conllu = None

    # Find the model and the dev and train sets.
    for filename in os.listdir(treebank_folder):
        if filename.endswith('-ud-dev.conllu'):
            dev_ud_conllu = os.path.join(treebank_folder, filename)
        if filename.endswith('-ud-train.conllu'):
            train_ud_conllu = os.path.join(treebank_folder, filename)
        if filename.endswith('-ud-test.conllu'):
            test_ud_conllu = os.path.join(treebank_folder, filename)

    # Convert files to unimorph format
    for conllu_file in [dev_ud_conllu, train_ud_conllu, test_ud_conllu]:
        if conllu_file:
            post_lang_arg = f"-l {post_lang}" if post_lang else ""
            os.system(f'python {ud_converter} convert --ud "{conllu_file}" {post_lang_arg}')

    dev_um_conllu = dev_ud_conllu.replace('-ud-', '-um-')
    train_um_conllu = train_ud_conllu.replace('-ud-', '-um-')
    
    # Create a .tsv object with lemma \t inflected form \t features to apply the morphology model
    with open(dev_um_conllu, 'r') as f:
        dev_list= []
        lines = f.read().replace('\n\n\n','\n').split('\n')
        dev_idx_changes = []
        for idx in range(len(lines)):
            if lines[idx] != '\n' and lines[idx] != '' and not lines[idx].startswith('#'):
                splitted_line = lines[idx].split('\t')
                lemma = splitted_line[2]
                inf_form = splitted_line[1]
                pos_tag = splitted_line[3]
                features = splitted_line[5]
                um_line = lemma + '\t' + inf_form + '\t' + features
                if pos_tag in pos_types:
                    dev_list.append(um_line)
                    dev_idx_changes.append(idx)

        dev_um = '\n'.join(dev_list).replace('\n\n', '\n')

    with open(train_um_conllu, 'r') as f:
        train_list= []
        lines = f.read().replace('\n\n\n','\n').split('\n')
        train_idx_changes = []
        for idx in range(len(lines)):
            if lines[idx] != '\n' and lines[idx] != '' and not lines[idx].startswith('#'):
                splitted_line = lines[idx].split('\t')
                lemma = splitted_line[2]
                inf_form = splitted_line[1]
                pos_tag = splitted_line[3]
                features = splitted_line[5]
                um_line = lemma + '\t' + inf_form + '\t' + features
                if pos_tag in pos_types:
                    train_list.append(um_line)
                    train_idx_changes.append(idx)
                        
        train_um = '\n'.join(train_list).replace('\n\n', '\n')
    
    # Create a temporal directory to place the intermediate files
    temp_dir = const.TEMP_DIR
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with open(temp_dir+'dev.in', 'w') as f:
        f.write(dev_um)
    with open(temp_dir+'train.in', 'w') as f:
        f.write(train_um)

    # Search the model in the folder
        for filename in os.listdir(model_folder):
            if '.nll' in filename:
                model_path = model_folder + '/' + filename

    # Define the morphological analyzer script and run it on the two
    morph_analyzer = const.DECODE_MORPH_SCRIPT

    dev_string = 'python {} --in_file "{}" --out_file "{}" --model "{}" --decode {}'.format(
        morph_analyzer, os.path.join(temp_dir, 'dev.in'), os.path.join(temp_dir, 'dev.out'), model_path, decode
        )
    train_string = 'python {} --in_file "{}" --out_file "{}" --model "{}" --decode {}'.format(
        morph_analyzer, os.path.join(temp_dir, 'train.in'), os.path.join(temp_dir, 'train.out'), model_path, decode
        )
    os.system(dev_string)
    print('Dev file morphology conversion finished')
    os.system(train_string)
    print('Train file morphology conversion finished')
    # Reconstruct the conllu files with the new inflected forms


    # Dev file
    with open(os.path.join(temp_dir, 'dev.out'), 'r', encoding='utf-8') as f1:
        inflected_list = f1.read().split('\n')
        inflected_count = 0
        with open(dev_ud_conllu) as f2:
            dev_conllu_in_lines = f2.read().replace('\n\n\n', '\n').split('\n')
            dev_conllu_out_lines = []
            for idx in range(len(dev_conllu_in_lines)):
                splitted_line = dev_conllu_in_lines[idx].split('\t')
                if idx in dev_idx_changes and '' not in splitted_line:
                    try:
                        org_form = splitted_line[1]
                        inf_form = inflected_list[inflected_count].split('\t')[1]
                        if inf_form[0] != ' ' and inf_form != '':
                            if org_form == org_form.capitalize():
                                splitted_line[1] = inf_form.capitalize()
                            else:
                                splitted_line[1] = inf_form
                        else:
                            splitted_line[1] = org_form
                        inflected_count += 1
                    except IndexError:
                        pass
                dev_conllu_out_lines.append(('\t').join(splitted_line))
            dev_conllu_out_text = '\n'.join(dev_conllu_out_lines)
    
    with open(output + '/dev.conllu', 'w') as f:
        f.write(dev_conllu_out_text)

    # Train file
    with open(os.path.join(temp_dir, 'train.out'), 'r') as f1:
        inflected_list = f1.read().split('\n')
        inflected_count = 0
        with open(train_ud_conllu) as f2:
            train_conllu_in_lines = f2.read().replace('\n\n\n', '\n').split('\n')
            train_conllu_out_lines = []
            for idx in range(len(train_conllu_in_lines)):
                splitted_line = train_conllu_in_lines[idx].split('\t')
                if idx in dev_idx_changes and '' not in splitted_line:
                    try:
                        org_form = splitted_line[1]
                        inf_form = inflected_list[inflected_count].split('\t')[1]
                        if inf_form[0] != ' ' and inf_form != '':
                            if org_form == org_form.capitalize():
                                splitted_line[1] = inf_form.capitalize()
                            else:
                                splitted_line[1] = inf_form
                        else:
                            splitted_line[1] = org_form
                        inflected_count += 1
                    except IndexError:
                        pass
                train_conllu_out_lines.append(('\t').join(splitted_line))
            train_conllu_out_text = '\n'.join(train_conllu_out_lines)
    
    with open(output + '/train.conllu', 'w') as f:
        f.write(train_conllu_out_text)
    
def unimorph_training_data(lang, data_dir):
    """Download the data from the github from UniMorph and prepare it to train a neural transducer morphological inflecter"""
    # Download
    data_url = 'https://github.com/unimorph/{}/'.format(lang)
    lang_dir = data_dir + '/' + lang
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(lang_dir):
        os.mkdir(lang_dir)

    os.system('git clone {} {}'.format(data_url, lang_dir))

    org_file = lang_dir + '/' + lang

    if lang == 'olo' or lang == 'krl': # mix all dialects in a file
        with open(lang_dir + '/' + lang, 'w') as org:
            for filename in os.listdir(lang_dir):
                if filename.startswith(lang+'-'):
                    with open(lang_dir + '/' + filename, 'r') as f:
                        org.write(f.read() + '\n')
    # Shuffle
    try:
        with open(org_file, 'r') as f:
            sentences = f.read().replace('\n\n', '\n').split('\n')
    except FileNotFoundError:
        with open(lang_dir + '/' + lang + '.1') as f:
            sentences_1  = f.read().replace('\n\n', '\n').split('\n')
        with open(lang_dir + '/' + lang + '.2') as f:
            sentences_2  = f.read().replace('\n\n', '\n').split('\n')
        sentences = sentences_1 + sentences_2

    final_sentences = []
    sentences = [sentence for sentence in sentences if sentence != '']
    random.Random(1).shuffle(sentences)
    
    for sentence in sentences:
        if len(sentence.split('\t')) == 3:
            splitted_sentence = sentence.split('\t')
            features = splitted_sentence[2]
            sep_features = features.split(';')
            random.Random(1).shuffle(sep_features)
            features = ';'.join(sep_features)
            splitted_sentence[2] = features
            final_sentences.append('\t'.join(splitted_sentence))
                
    # Split in 80/10/10
    split_1 = int(0.8*len(final_sentences))
    split_2 = int(0.9*len(final_sentences))

    train_sentences = final_sentences[:split_1]
    dev_sentences = final_sentences[split_1:split_2]
    test_sentences = final_sentences[split_2:]
    # Create files
    with open(lang_dir+'/{}.train'.format(lang), 'w') as f:
        for sentence in train_sentences:
            f.write(sentence+'\n')
    with open(lang_dir+'/{}.dev'.format(lang), 'w') as f:
        for sentence in dev_sentences:
            f.write(sentence+'\n')
    with open(lang_dir+'/{}.test'.format(lang), 'w') as f:
        for sentence in test_sentences:
            f.write(sentence+'\n')     

def train_morph(lang, dir):#, arch, source, data='unimorph'):
    """Train a morphological analyzer using the files extracted from UniMorph"""
    # Define paths
    lang_dir = os.path.join(dir, lang) + '/'
    script = const.TRAIN_MORPH_SCRIPT
    train = os.path.join(lang_dir, lang + '.train')
    dev = os.path.join(lang_dir, lang + '.dev')
    test = os.path.join(lang_dir, lang + '.test')
    model = os.path.join(lang_dir, 'model')
    print(model)
    # Call training script from neural-transducer
    os.system('python "{}" --train "{}" --dev "{}" --test  "{}" \
            --model "{}" --arch hmm --dataset unimorph \
            --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
            --src_layer 2 --trg_layer 1 --max_norm 5 --shuffle \
            --estop 1e-8 --epochs 50 --bs 50 --bestacc --indtag --mono'.format(
            script, train, dev, test, model))