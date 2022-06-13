import glob
import os
import string
import sys


def topicInfo():
    id_list = []
    topics = {}
    topics_file = open("Topics.txt", "r")
    topics_file_content = topics_file.readlines()

    for line in topics_file_content:
        line = line.strip()
        if line.startswith("<num>"):
            topic_id = line.replace("<num> Number: ", "")
            # print(topic_id)
            id_list.append(topic_id)

        if line.startswith("<title>"):

            topic_title = line.replace("<title>", "")
    #         topic_title = line.replace("<title> ", "")

            topics[topic_id] = topic_title
    topics_file.close()
    return topics


def F1(rel_path, model_path, rank_path):    # For each doc
    A = {}
    B = {}
    C = {}
    R = {}

    for line in open(rel_path, "r"):
        line = line.strip()
        line1 = line.split()
        A[line1[1]] = int(float(line1[2]))
    for line in open(model_path):
        line = line.strip()
        line1 = line.split()
        B[line1[1]] = int(float(line1[2]))

    for line in open(rank_path):
        line = line.strip()
        line1 = line.split()
        C[line1[0]] = float(line1[1])

    i = 1
    for k, v in sorted(C.items(), key=lambda x: x[1], reverse=True):
        R[i] = k
        i += 1
        if i > 10:
            break
    return (A, B, R)


def F1_result(r_doc, model_doc):
    r = 0.0
    p = 0.0
    F_measure = 0.0

    R = 0
    for k, v in r_doc.items():
        if v == 1:
            R = R + 1

    R1 = 0
    for k, v in model_doc.items():
        if v == 1:
            R1 = R1 + 1

    RR1 = 0
    for k, v in model_doc.items():
        if v == 1 and r_doc[k] == 1:
            RR1 = RR1 + 1

    if R > 0 and R1 > 0:
        r = float(RR1)/float(R)
        p = float(RR1) / float(R1)
        if(r + p) > 0:
            F_measure = (2 * p * r) / (p + r)
        else:
            F_measure = 0
    elif R <= 0:
        r = 0
        F_measure = 0
    elif R1 <= 0:
        p = 0
        F_measure = 0
    return (r, p, F_measure)


if __name__ == '__main__':
    main_path = os.getcwd()
    topics = topicInfo()

    for id, topic in topics.items():
        id = id.replace("R", "")
        rel_file = main_path + "/RelevanceFeedback/Dataset" + id + ".txt"
        bm25_file = main_path + "/bm25BinaryOutput/Dataset" + id + ".txt"
        qlm_file = main_path + "/QlmBinaryOutput/Dataset" + id + ".txt"
        rank_file = main_path + "/bm25_ranks/Dataset" + id + ".txt"

        # Baseline model
        # (rel_doc, b_doc, top_10) = F1(rel_file, bm25_file, rank_file)
        # (r_bm25, p_bm25, f_bm25) = F1_result(rel_doc, b_doc)

        # print(id)
        # print("Recall = " + str(r_bm25))
        # print("Precision = " + str(p_bm25))
        # print("F-Measure = " + str(f_bm25))
        # print()
        ri = 0
        ap1 = 0.0
        # print(top_10)
        # for n, id in sorted(top_10.items(), key=lambda x: int(x[0])):
        #     if(rel_doc[id] == 1):
        #         ri = ri + 1
        #         pi = float(ri) / float(int(n))
        #         ap1 = ap1 + pi
        #         print("At position " + str(int(n)) + " docID: " +
        #               id + ", precision = " + str(pi))
        # print()

        # Language model
        (rel_doc, qlm_doc, top_10) = F1(rel_file, qlm_file, rank_file)
        (r_qlm, p_qlm, f_qlm) = F1_result(rel_doc, qlm_doc)
        # print(id)
        # print("Recall = " + str(r_qlm))
        # print("Precision = " + str(p_qlm))
        # print("F-Measure = " + str(f_qlm))
        # print()
        for n, id in sorted(top_10.items(), key=lambda x: int(x[0])):
            if(rel_doc[id] == 1):
                ri = ri + 1
                pi = float(ri) / float(int(n))
                ap1 = ap1 + pi
                print("At position " + str(int(n)) + " docID: " +
                      id + ", precision = " + str(pi))
        print()
