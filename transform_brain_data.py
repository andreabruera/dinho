import numpy
import os
import re

from tqdm import tqdm

words = {1 : list(), 2 : list()}
full_subjects_data = {1 : dict(), 2 : dict()}


for d in words.keys():
    ### words
    with open(os.path.join('bits_n_pieces', 'fernandino{}_words.txt'.format(d))) as i:
        for l_i, l in enumerate(i):
            line = l.strip()
            word = '{}'.format(line)
            if line != '':
                words[d].append(word)

    mapper = dict()
    with open(os.path.join('bits_n_pieces', 'colortable_desikan_killiany.txt')) as i:
        for l in i:
            l = re.sub('\s+', r'\t', l)
            line = l.strip().split('\t')
            assert len(line) > 2
            mapper[line[0]] = 'L_{}'.format(line[1])
            mapper[str(int(line[0])+35)] = 'R_{}'.format(line[1])
    checker = list()
    folder = os.path.join('original_data', 'Study{}_neural_vectors_RSMs'.format(d))
    for brain_area_folder in tqdm(os.listdir(folder)):
        brain_area = re.sub(r'ALE_|DK_|roi|_mask', '', brain_area_folder)
        if brain_area in mapper.keys():
            brain_area = mapper[brain_area]
        else:
            if 'L_' not in brain_area and 'R_' not in brain_area and 'sem' not in brain_area:
                brain_area = 'L_{}'.format(brain_area)
        if brain_area not in checker:
            checker.append(brain_area)
        else:
            raise RuntimeError('There was an issue with area naming!')
        for f in os.listdir(os.path.join(folder, brain_area_folder,)):
            if 'txt' not in f:
                continue
            mtrx = list()
            sub = f.split('_')[-1].replace('.txt', '')
            with open(os.path.join(folder, brain_area_folder, f)) as i:
                for l_i, l in enumerate(i):
                    line = [sim for sim_i, sim in enumerate(l.strip().split('\t'))]
                    assert len(line) == len(words[d])
                    mtrx.append(line)
            ### checks
            assert len(mtrx) == len(words[d])
            for line in mtrx:
                assert len(line) == len(words[d])
            ### adding data
            if brain_area not in full_subjects_data[d].keys():
                full_subjects_data[d][brain_area] = dict()
            for w_one_i, w_one in enumerate(words[d]):
                for w_two_i, w_two in enumerate(words[d]):
                    if w_two_i > w_one_i:
                        ### sims
                        key = tuple(sorted([w_one, w_two]))
                        full_subjects_data[d][brain_area][key] = dict()
                        full_subjects_data[d][brain_area][key][sub] = float(mtrx[w_one_i][w_two_i])
