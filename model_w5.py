import math
import sys
import os
from stemming.porter2 import stem
from F1 import topicInfo
from A2_refwk9 import parse_rcv_coll


def w5(coll, ben, theta):
    T = {}
    # select T from positive documents and r(tk)
    for id, doc in coll.get_docs().items():
        if ben[id] > 0:
            for term, freq in doc.terms.items():
                try:
                    T[term] += 1
                except KeyError:
                    T[term] = 1
    # calculate n(tk)
    ntk = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try:
                ntk[term] += 1
            except KeyError:
                ntk[term] = 1

    # calculate N and R

    num_docs = coll.get_num_docs()
    R = 0
    for id, fre in ben.items():
        if ben[id] > 0:
            R += 1

    for t, rtk in T.items():
        T[t] = ((rtk+0.5) / (R-rtk + 0.5)) / \
            ((ntk[t]-rtk+0.5)/(num_docs-ntk[t]-R+rtk + 0.5))

    # calculate the mean of w4 weights.
    meanW5 = 0
    for id, rtk in T.items():
        meanW5 += rtk
    meanW5 = meanW5/len(T)

    # Features selection
    Features = {t: r for t, r in T.items() if r > meanW5 + theta}
    return Features


def BM25Testing(coll, features):
    ranks = {}
    for id, doc in coll.get_docs().items():
        Rank = 0
        for term in features.keys():
            if term in doc.get_term_list():
                try:
                    ranks[id] += features[term]
                except KeyError:
                    ranks[id] = features[term]
    return ranks


if __name__ == "__main__":
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    topics = topicInfo()
    main_path = os.getcwd()

    for id, topics in topics.items():
        id = id.replace("R", "")
        rel_path = main_path + "/RelevanceFeedback/Dataset" + id + ".txt"
        rel_file = open(rel_path, "r")
        file = rel_file.readlines()

        ben = {}
        features = {}
        for line in file:
            line = line.strip()
            line_list = line.split()
            ben[line_list[1]] = float(line_list[2])
        rel_file.close()

        dataset = "Dataset" + id
        dataset_path = main_path + "/DataCollection/" + dataset
        theta = 3.5
        coll_ = parse_rcv_coll(dataset_path, stop_words)
        bm25_weights = w5(coll_, ben, theta)
        # print(bm25_weights)
        # with open(main_path + "/bm25_weights/" + dataset+".txt", "w") as f_25w:
        #     for(k, v) in sorted(bm25_weights.items(), key=lambda x: x[1], reverse=True):
        #         f_25w.write(k + " " + str(v) + "\n")
        for t, score in bm25_weights.items():
            features[t] = score
        ranks = BM25Testing(coll_, features)
        with open(main_path + "/bm25_ranks/" + dataset+".txt", "w") as f_rank:
            for (d, v) in sorted(ranks.items(), key=lambda x: x[1], reverse=True):
                f_rank.write(d + ' ' + str(v) + '\n')
