# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-01-30 11:47:55
# @Filepath: debiaswe/debias.py
from __future__ import print_function, division
import we
import json
import numpy as np
import argparse
import sys
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""
# definitional [['woman', 'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ['daughter', 'son'], ['gal', 'guy'], ['female', 'male'], ['her', 'his'], ['herself', 'himself'], ['Mary', 'John']]

# gender specific words ['he', 'his', 'He', 'her', 'she', 'him', 'She', 'man', 'women', 'men', 'His', 'woman', 'spokesman', 'wife', 'himself', 'son', 'mother', 'father', 'chairman', 'daughter', 'husband', 'guy', 'girls', 'girl', 'Her', 'boy', 'King', 'boys', 'brother', 'Chairman']

# equalize [['monastery', 'convent'], ['spokesman', 'spokeswoman'], ['Catholic_priest', 'nun'], ['Dad', 'Mom'], ['Men', 'Women'], ['councilman', 'councilwoman'], ['grandpa', 'grandma'], ['grandsons', 'granddaughters'], ['prostate_cancer', 'ovarian_cancer'], ['testosterone', 'estrogen'], ['uncle', 'aunt'], ['wives', 'husbands'], ['Father', 'Mother'], ['Grandpa', 'Grandma'], ['He', 'She'], ['boy', 'girl'], ['boys', 'girls'], ['brother', 'sister'], ['brothers', 'sisters'], ['businessman', 'businesswoman'], ['chairman', 'chairwoman'], ['colt', 'filly'], ['congressman', 'congresswoman'], ['dad', 'mom'], ['dads', 'moms'], ['dudes', 'gals'], ['ex_girlfriend', 'ex_boyfriend'], ['father', 'mother'], ['fatherhood', 'motherhood'], ['fathers', 'mothers'], ['fella', 'granny'], ['fraternity', 'sorority'], ['gelding', 'mare'], ['gentleman', 'lady'], ['gentlemen', 'ladies'], ['grandfather', 'grandmother'], ['grandson', 'granddaughter'], ['he', 'she'], ['himself', 'herself'], ['his', 'her'], ['king', 'queen'], ['kings', 'queens'], ['male', 'female'], ['males', 'females'], ['man', 'woman'], ['men', 'women'], ['nephew', 'niece'], ['prince', 'princess'], ['schoolboy', 'schoolgirl'], ['son', 'daughter'], ['sons', 'daughters'], ['twin_brother', 'twin_sister']]


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("definitional_filename", help="JSON of definitional pairs")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
    parser.add_argument("equalize_filename", help="???.bin")
    parser.add_argument("debiased_filename", help="???.bin")

    args = parser.parse_args()
    print(args)

    with open(args.definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(args.equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(args.embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if args.embedding_filename[-4:] == args.debiased_filename[-4:] == ".bin":
        E.save_w2v(args.debiased_filename)
    else:
        E.save(args.debiased_filename)

    print("\n\nDone!\n")
