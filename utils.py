import numpy
import os
import pickle
import random
import re
import scipy

from scipy import stats
from tqdm import tqdm

def divide_binder_ratings(ratings):
    subdivisions = dict()
    with open(os.path.join('data', 'binder_sections.tsv')) as i:
        for l in i:
            line = l.strip().split('\t')
            if line[1] not in ratings['actor'].keys():
                print(line)
                continue
            if line[0] not in subdivisions.keys():
                subdivisions[line[0]] = list()
            subdivisions[line[0]].append(line[1])
    section_vecs = {w : {'{}_section'.format(k) : [ratings[w][dim] for dim in v] for k, v in subdivisions.items()} for w in ratings.keys()}
    return section_vecs

def read_exp48(words):
    vecs = dict()
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'fernandino_experiential_ratings.tsv',
                             )
    assert os.path.exists(file_path)
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                dimensions = [w.strip() for w in line[1:]]
                continue
            vecs[line[0].lower().strip()] = numpy.array(line[1:], dtype=numpy.float64)

    return vecs, list(dimensions)

def read_fernandino_ratings():
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'fernandino_experiential_ratings.tsv',
                             )
    assert os.path.exists(file_path)
    norms = dict()
    with open(file_path) as i:
        counter = 0
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            assert len(line) == len(header)
            if len(line[0].split()) == 1:
                for h_i, h in zip(range(len(header)), header):
                    if h_i == 0:
                        continue
                    val = float(line[h_i])
                    ### minimum is 0, max is 5
                    assert val >= 0. and val <= 6.
                    curr_val = float(val) / 6.
                    if h not in norms.keys():
                        norms[h] = dict()
                    norms[h][line[0].lower().strip()] = curr_val
    ### checking that all went good...
    for k, v in norms.items():
        for w in v.keys():
            for k_two, v_two in norms.items():
                assert w in v_two.keys()
    ### putting the dictionary together
    final_norms = {k : {k_two : v_two[k] for k_two, v_two in norms.items()} for k in norms['Audition'].keys()}

    return final_norms

def read_binder_ratings():
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'binder_ratings.tsv',
                             )
    assert os.path.exists(file_path)
    norms = dict()
    with open(file_path) as i:
        counter = 0
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            assert len(line) == len(header)
            for h_i, h in zip(range(len(header)), header):
                if h_i in [0, 1, 2, 3, 4] or h_i>69:
                    print(h)
                    continue
                if line[h_i] == 'na':
                    curr_val = numpy.nan
                else:
                    val = float(line[h_i])
                    ### minimum is 0, max is 5
                    assert val >= 0. and val <= 6.
                    curr_val = float(val) / 6.
                if h not in norms.keys():
                    norms[h] = dict()
                norms[h][line[1].lower().strip()] = curr_val
    assert len(norms.keys()) == 65
    ### checking that all went good...
    for k, v in norms.items():
        for w in v.keys():
            for k_two, v_two in norms.items():
                assert w in v_two.keys()
    ### putting the dictionary together
    final_norms = {k : {k_two : v_two[k] for k_two, v_two in norms.items()} for k in norms['Audition'].keys()}

    return final_norms

def read_fernandino(vocab, pos, lang='en', trans=dict(), return_dict=False, avg_subjects=False):

    words = {1 : list(), 2 : list()}
    subjects_data = {1 : dict(), 2 : dict()}
    full_subjects_data = {1 : dict(), 2 : dict()}
    pkl_path = os.path.join('data', 'fernandino_rsa.pkl')
    full_pkl_path = os.path.join('data', 'fernandino_pairwise.pkl')
    marker = False
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as i:
            subjects_data = pickle.load(i)
        with open(full_pkl_path, 'rb') as i:
            full_subjects_data = pickle.load(i)
        marker = True

    for d in words.keys():
        missing_idxs = list()
        ### words
        with open(os.path.join('data', 'fernandino{}_words.txt'.format(d))) as i:
            for l_i, l in enumerate(i):
                line = l.strip()
                word = '{}'.format(line)
                if line != '':
                    try:
                        if lang != 'en':
                            line = trans[line]
                        if vocab[line] == 0:
                            missing_idxs.append(l_i)
                            print('missing: {}'.format([line, pos[line]]))
                            continue
                    except KeyError:
                        print('missing: {} - unknown POS'.format(line))
                        missing_idxs.append(l_i)
                        continue
                    words[d].append(word)
        ### similarities
        ### other anterior-frontal areas
        ### reading mapper
        if marker:
            continue

        mapper = dict()
        with open(os.path.join('data', 'colortable_desikan_killiany.txt')) as i:
            for l in i:
                l = re.sub('\s+', r'\t', l)
                line = l.strip().split('\t')
                assert len(line) > 2
                mapper[line[0]] = 'L_{}'.format(line[1])
                mapper[str(int(line[0])+35)] = 'R_{}'.format(line[1])
        folder = 'Study{}_neural_vectors_RSMs'.format(d)
        for brain_area_folder in tqdm(os.listdir(os.path.join('data', folder))):
            brain_area = re.sub(r'ALE_|DK_|roi|_mask', '', brain_area_folder)
            if brain_area in mapper.keys():
                brain_area = mapper[brain_area]
            #print(brain_area)
            for f in os.listdir(os.path.join('data', folder, brain_area_folder,)):
                if 'txt' not in f:
                    continue
                mtrx = list()
                sub = f.split('_')[-1].replace('.txt', '')
                with open(os.path.join('data', folder, brain_area_folder, f)) as i:
                    for l_i, l in enumerate(i):
                        if l_i in missing_idxs:
                            continue
                        line = [sim for sim_i, sim in enumerate(l.strip().split('\t')) if sim_i not in missing_idxs]
                        assert len(line) == len(words[d])
                        mtrx.append(line)
                ### checks
                assert len(mtrx) == len(words[d])
                for line in mtrx:
                    assert len(line) == len(words[d])
                ### adding data
                if brain_area not in subjects_data[d].keys():
                    subjects_data[d][brain_area] = dict()
                    if return_dict:
                        full_subjects_data[d][brain_area] = dict()
                ### RSA
                ### removing diagonal
                subjects_data[d][brain_area][sub] = numpy.array([val for line_i, line in enumerate(mtrx) for val_i, val in enumerate(line) if val_i>line_i], dtype=numpy.float64).tolist()
                if return_dict:
                    full_subjects_data[d][brain_area][sub] = dict()
                    for w_one_i, w_one in enumerate(words[d]):
                        for w_two_i, w_two in enumerate(words[d]):
                            if w_two_i > w_one_i:
                                full_subjects_data[d][brain_area][sub][tuple(sorted([w_one, w_two]))] = float(mtrx[w_one_i][w_two_i])
    if not marker:
        with open(pkl_path, 'wb') as i:
            pickle.dump(subjects_data, i)
        with open(full_pkl_path, 'wb') as i:
            pickle.dump(full_subjects_data, i)
    ### replicating results by fernandino
    if avg_subjects:
        for d in subjects_data.keys():
            for a in subjects_data[d].keys():
                avg_corrs = numpy.average([v for v in subjects_data[d][a].values()], axis=0)
                subjects_data[d][a] = {'all' : avg_corrs}
                if return_dict:
                    avg_sims = dict()
                    for _, tups in full_subjects_data[d][a].items():
                        for t, val in tups.items():
                            try:
                                avg_sims[t] = (avg_sims[t] + val)/2
                            except KeyError:
                                avg_sims[t] = val
                    full_subjects_data[d][a] = {'all' : avg_sims}

    if return_dict:
        return words, subjects_data, full_subjects_data
    else:
        return words, subjects_data
