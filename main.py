import os
import random


# This script trains a model using dep2label and the UD corpus selected using the desired encoding
# First of all, define those routes that are not going to change depending on the treebank or the encoding we are using
ud_treebanks_main = os.path.expanduser("~/Universal Dependencies 2.7/ud-treebanks-v2.7/")

def combine_treebanks(list_of_treebanks, model_name):
    """Create a folder with the needed files to train a model, joining 
    the data of more than one treebank"""
    # Define the name of the model folder
    output_folder = ud_treebanks_main + model_name + '/'
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        with open(output_folder+'treebanks_used.txt', 'w') as f:
            f.write('Treebanks used:\t')
        for treebank in list_of_treebanks:
            with open(output_folder+'treebanks_used.txt', 'a') as f:
                f.write(treebank + '\t')
    else:
        print('There is already a folder with this name, change the name and try again')
        return None

    training_sentences_list = []
    dev_sentences_list = []
    for treebank in list_of_treebanks:
        ud_treebank_dir = ud_treebanks_main + treebank + '/'
        try:
            # Iterate through the files in the folder to find the .conllu files
            for filename in os.listdir(ud_treebank_dir):
                if filename.endswith('-ud-train.conllu') :
                    train_conllu_udpath = ud_treebank_dir + '/' + filename
                    with open(train_conllu_udpath, 'r') as f:
                        training_sentences = f.read().replace('\n\n\n', '\n').split('\n\n')
                        for sentence in training_sentences:
                            if sentence != '':
                                training_sentences_list.append(sentence)

                elif filename.endswith('-ud-dev.conllu'):
                    dev_conllu_udpath = ud_treebank_dir  + '/' + filename
                    with open(dev_conllu_udpath, 'r') as f:
                        dev_sentences = f.read().replace('\n\n\n', '\n').split('\n\n')
                        for sentence in dev_sentences:
                            if sentence != '':
                                dev_sentences_list.append(sentence)

        except FileNotFoundError:
            print('The directory "{}" was not found. Please select a valid directory and try again.'.format(ud_treebank_dir))
            return None

        # Shuffle the sentences
        random.Random(1).shuffle(training_sentences_list)
        random.Random(1).shuffle(dev_sentences_list)

        with open(output_folder + '/' + '{}-ud-train.conllu'.format(model_name.lower()), 'w') as f:
            for sentence in training_sentences_list:
                f.write(sentence + '\n\n')

        with open(output_folder + '/' + '{}-ud-dev.conllu'.format(model_name.lower()), 'w') as f:
            for sentence in dev_sentences_list:
                f.write(sentence + '\n\n')

def ud_to_um(
    treebank,
    lang=''
):
    """Convert UD features to UM"""
    # Find the model and the dev and train sets.
    ud_treebank_folder = os.path.expanduser('/home/alberto/Universal Dependencies 2.7/ud-treebanks-v2.7/{}/'.format(treebank))
    # Find the ud-conversion script
    ud_converter = os.path.expanduser('~/ud-compatibility/UD_UM/marry.py')
    # Find the model and the dev and train sets.
    ud_treebank_folder = os.path.expanduser('/home/alberto/Universal Dependencies 2.7/ud-treebanks-v2.7/{}/'.format(treebank))
    for filename in os.listdir(ud_treebank_folder):
        if filename.endswith('-ud-dev.conllu'):
            dev_ud_conllu = ud_treebank_folder + filename
            dev_filename = filename
        if filename.endswith('-ud-train.conllu'):
            train_ud_conllu = ud_treebank_folder + filename
            train_filename = filename
        if filename.endswith('-ud-test.conllu'):
            test_ud_conllu = ud_treebank_folder + filename
            train_filename = filename
    # Convert both files to unimorph format
    if lang != '':
        try:
            os.system('python {} convert --ud "{}" -l {}'.format(ud_converter, dev_ud_conllu, lang))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}" -l {}'.format(ud_converter, train_ud_conllu, lang))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}" -l {}'.format(ud_converter, test_ud_conllu, lang))
        except UnboundLocalError:
            pass
    else:
        try:
            os.system('python {} convert --ud "{}"'.format(ud_converter, dev_ud_conllu))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}"'.format(ud_converter, train_ud_conllu))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}"'.format(ud_converter, test_ud_conllu))
        except UnboundLocalError:
            pass

def morph_translation(
    treebank, 
    model, 
    pos_types, 
    lang='', 
    decode='greedy'
):
    """Inflex the lemmas of a UD Treebank using a morphological analyzer trained in a different language to 'translate' the treebank."""
    # Find the model and the dev and train sets.
    ud_treebank_folder = os.path.expanduser('/home/alberto/Universal Dependencies 2.7/ud-treebanks-v2.7/{}/'.format(treebank))
    model_path = os.path.expanduser('~/data_augmentation/morph_models/' + model)
    # Find the ud-conversion script
    ud_converter = os.path.expanduser('~/ud-compatibility/UD_UM/marry.py')
    # Find the model and the dev and train sets.
    ud_treebank_folder = os.path.expanduser('/home/alberto/Universal Dependencies 2.7/ud-treebanks-v2.7/{}/'.format(treebank))
    for filename in os.listdir(ud_treebank_folder):
        if filename.endswith('-ud-dev.conllu'):
            dev_ud_conllu = ud_treebank_folder + filename
            dev_filename = filename
        if filename.endswith('-ud-train.conllu'):
            train_ud_conllu = ud_treebank_folder + filename
            train_filename = filename
        if filename.endswith('-ud-test.conllu'):
            test_ud_conllu = ud_treebank_folder + filename
            test_filename = filename
    # Convert both files to unimorph format
    if lang != '':
        try:
            os.system('python {} convert --ud "{}" -l {}'.format(ud_converter, dev_ud_conllu, lang))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}" -l {}'.format(ud_converter, train_ud_conllu, lang))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}" -l {}'.format(ud_converter, test_ud_conllu, lang))
        except UnboundLocalError:
            pass
    else:
        try:
            os.system('python {} convert --ud "{}"'.format(ud_converter, dev_ud_conllu))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}"'.format(ud_converter, train_ud_conllu))
        except UnboundLocalError:
            pass
        try:
            os.system('python {} convert --ud "{}"'.format(ud_converter, test_ud_conllu))
        except UnboundLocalError:
            pass


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
    temp_dir = os.path.expanduser('~/data_augmentation/temp_dir/')
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    with open(temp_dir+'dev.in', 'w') as f:
        f.write(dev_um)
    with open(temp_dir+'train.in', 'w') as f:
        f.write(train_um)

    # Define the morphological analyzer script and run it on the two
    morph_analyzer = os.path.expanduser('~/neural-transducer/src/decode.py')
    #morph_analyzer = os.path.expanduser('~/neural-transducer/src/sigmorphon19-task1-decode.py')
    dev_string = 'python {} --in_file "{}" --out_file "{}" --model "{}" --decode {}'.format(
        morph_analyzer, temp_dir+'dev.in', temp_dir+'dev.out', model_path, decode
        )
    train_string = 'python {} --in_file "{}" --out_file "{}" --model "{}" --decode {}'.format(
        morph_analyzer, temp_dir+'train.in', temp_dir+'train.out', model_path, decode
        )
    os.system(dev_string)
    print('Dev file morphology conversion finished')
    os.system(train_string)
    print('Train file morphology conversion finished')
    # Reconstruct the conllu files with the new inflected forms
    ud_output_folder = ud_treebanks_main + model.replace('.nll', '') + '-' + treebank + '/'
    if not os.path.exists(ud_output_folder):
        os.mkdir(ud_output_folder)

    dev_ud_conllu_out = ud_output_folder + model.replace('.nll', '') + '-' + dev_filename
    train_ud_conllu_out = ud_output_folder + model.replace('.nll', '') + '-' + train_filename
    # Dev file
    with open(temp_dir + 'dev.out', 'r', encoding='utf-8') as f1:
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
    
    with open(dev_ud_conllu_out, 'w') as f:
        f.write(dev_conllu_out_text)

    # Train file
    with open(temp_dir + 'train.out', 'r') as f1:
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
    
    with open(train_ud_conllu_out, 'w') as f:
        f.write(train_conllu_out_text)
    
def unimorph_training_data(lang, split):
    """Download the data from the github from UniMorph and prepare it to train a neural transducer morphological inflecter"""
    # Download
    if split == 'form':
        um_dir = 'morph_data/um/'
    elif split == 'lemma':
        um_dir = 'morph_data/um_lemma/'
    else:
        print('Split type is not valid. Insert "form" or "lemma".')
        return None

    data_url = 'https://github.com/unimorph/{}/'.format(lang)
    lang_dir = um_dir + lang
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


        sentences = [sentence for sentence in sentences if sentence != '']
        random.Random(1).shuffle(sentences)
        final_sentences = []
        for sentence in sentences:
            if len(sentence.split('\t')) == 3:
                splitted_sentence = sentence.split('\t')
                features = splitted_sentence[2]
                sep_features = features.split(';')
                random.Random(1).shuffle(sep_features)
                features = ';'.join(sep_features)
                splitted_sentence[2] = features
                final_sentences.append('\t'.join(splitted_sentence))
        
    if split == 'form':            
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

    elif split == 'lemma':
        d = read(org_file)
        total_d = defaultdict(list)
        # Unite all the forms of the same lemma under a defaultdict value entry
        for k,v in d.items():
            total_d[k].append(v)

        lemmas = list(total_d.items())

        n = len(lemmas)
        random.seed(1)
        random.shuffle(lemmas)

        # Split in 70/20/10
        train_prop, dev_prop, test_prop = 0.7, 0.2, 0.1
        assert np.isclose(sum([train_prop, dev_prop, test_prop]), 1, atol=1e-08)
        train, test, dev = lemmas[:int(train_prop * n)], lemmas[int(train_prop * n):int((train_prop+dev_prop) * n)], lemmas[int((train_prop+dev_prop) * n):]

        train = dict2lists(train)
        dev = dict2lists(dev)
        test = dict2lists(test)
        
        with open(lang_dir+'/{}.train'.format(lang), 'w') as f:
            for row in train:
                f.write('\t'.join(row)+'\n')

        with open(lang_dir+'/{}.dev'.format(lang), 'w') as f:
            for row in dev:
                f.write('\t'.join(row)+'\n')

        with open(lang_dir+'/{}.test'.format(lang), 'w') as f:
            for row in test:
                f.write('\t'.join(row)+'\n')

def train_morph(lang, arch, source, data='unimorph'):
    """Train a morphological analyzer using the files extracted from UniMorph"""
    # Define paths
    lang_dir= os.path.expanduser('~/data_augmentation/morph_data/{}/'.format(source)) + lang
    script = os.path.expanduser('~/neural-transducer/src/train.py')
    models_dir = os.path.expanduser('~/data_augmentation/morph_models/')
    train = lang_dir + '/' + lang + '.train'
    dev = lang_dir + '/' +lang + '.dev'
    test = lang_dir + '/' + lang + '.test'
    model = models_dir + lang + '-' + arch + '-' + source
    
    # Call training script fron neural-transducer
    if arch == 'hmm':
        os.system('python "{}" --train "{}" --dev "{}" --test  "{}" \
                --model "{}" --arch "{}" --dataset "{}" \
                --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
                --src_layer 2 --trg_layer 1 --max_norm 5 --shuffle \
                --estop 1e-8 --epochs 50 --bs 50 --bestacc --indtag --mono'.format(
                script, train, dev, test, model, arch, data))
    
    elif arch == 'transformer':
        os.system('python "{}" --train "{}" --dev "{}" --test  "{}" \
                --model "{}" --arch "{}" --dataset "{}" \
                --embed_dim 256 --src_hs 1024 --trg_hs 1024 --dropout 0.3 --nb_heads 4 \
                --label_smooth 0.1 --total_eval 50 \
                --src_layer 4 --trg_layer 4 --max_norm 1 --lr 0.001 --shuffle \
                --gpuid 0 --estop 1e-8 --bs 400 --max_steps 20000 \
                --scheduler warmupinvsqr --warmup_steps 4000 --cleanup_anyway --beta2 0.98 --bestacc'.format(
                script, train, dev, test, model, arch, data))

if __name__ == '__main__':
    print('test')