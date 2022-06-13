import glob
import os
import string
import math
from stemming.porter2 import stem


class BowDoc:
    def __init__(self, docid):
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_term(self, term):
        try:
            self.terms[term] += 1
        except KeyError:
            self.terms[term] = 1

    def get_term_count(self, term):
        try:
            return self.terms[term]
        except KeyError:
            return 0

    def get_term_freq_dict(self):
        return self.terms

    def get_term_list(self):
        return sorted(self.terms.keys())

    def get_docid(self):
        return self.docid

    def get_doc_len(self):
        return self.doc_len

    def set_doc_len(self, doc_len):
        self.doc_len = doc_len


class BowColl:

    def __init__(self):
        self.docs = {}

    def add_doc(self, doc):
        self.docs[doc.get_docid()] = doc

    def get_doc(self, docid):
        return self.docs[docid]

    def get_docs(self):
        return self.docs

    def get_num_docs(self):
        return len(self.docs)


def parse_rcv_coll(inputpath, stop_words):
    coll = BowColl()
    os.chdir(inputpath)

    for file_ in glob.glob("*.xml"):
        curr_doc = None
        start_end = False
        word_count = 0
        for line in open(file_):
            line = line.strip()
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            curr_doc = BowDoc(docid)
                            break
                    continue
                if line.startswith("<text>"):
                    start_end = True
            elif line.startswith("</text>"):
                break
            elif curr_doc is not None:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('', '', string.digits)).translate(
                    str.maketrans(string.punctuation, ' '*len(string.punctuation)))

                for term in line.split():
                    word_count += 1
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        curr_doc.add_term(term)
        if curr_doc is not None:
            curr_doc.set_doc_len(word_count)
            coll.add_doc(curr_doc)
    return coll


def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl = tot_dl + doc.get_doc_len()
    return tot_dl/coll.get_num_docs()


def bm25(coll, q, df):
    bm25s = {}
    avg_dl = avg_doc_len(coll)
    no_docs = coll.get_num_docs()
    for id, doc in coll.get_docs().items():
        query_terms = q.split()
        qfs = {}
        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        k = 1.2 * ((1 - 0.75) + 0.75 * (doc.get_doc_len() / float(avg_dl)))
        bm25_ = 0.0
        for qt in qfs.keys():
            n = 0
            if qt in df.keys():
                n = df[qt]
                f = doc.get_term_count(qt)
                qf = qfs[qt]
                bm = math.log(1.0 / ((n + 0.5) / (no_docs - n + 0.5)), 2) * \
                    (((1.2 + 1) * f) / (k + f)) * \
                    (((100 + 1) * qf) / float(100 + qf))
                bm25_ += bm
        bm25s[doc.get_docid()] = bm25_
    return bm25s


def calc_df(coll):

    df_ = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try:
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    return df_


def coll_len(coll):
    C = 0
    for id, doc in coll.get_docs().items():
        D = doc.get_doc_len()
        C += D
    return C


def coll_q_f(coll, q, df):
    qfs = {}
    C_q = 0
    for id, doc in coll.get_docs().items():
        query_terms = q.split()

        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        for qt in qfs.keys():

            if qt in df.keys():
                f_q_doc = doc.get_term_count(qt)
                C_q += f_q_doc
    return C_q


def QLM(coll, q, df):
    # Lambda value for short queries referenced from textbook chapter 7.
    QLMs = {}
    Lambda = 0.2
    qfs = {}
    C = coll_len(coll)   # total num of words in the coll
    C_q = coll_q_f(coll, q, df)   # num of term from the q in the coll
    QLM_socre = 0
    for id, doc in coll.get_docs().items():
        D = doc.get_doc_len()

        query_terms = q.split()

        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        for qt in qfs.keys():

            if qt in df.keys():
                f_q_doc = doc.get_term_count(qt)
                l_ = (1-Lambda) * (f_q_doc / D)
                r_ = Lambda * (C_q / C)

                if l_ > 0 and r_ > 0:

                    left = math.log(l_, 2)
                    right = math.log(r_, 2)
                    p = left + right
                    QLM_socre += p
        QLMs[doc.get_docid()] = QLM_socre
    return QLMs


def QLM_binary(coll, q, df):
    qfs = {}
    qlm_binary = {}
    for id, doc in coll.get_docs().items():

        query_terms = q.split()

        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] += 1
            except KeyError:
                qfs[term] = 1
        for qt in qfs.keys():

            if qt in df.keys():
                f_q_doc = doc.get_term_count(qt)
        qlm_binary[doc.get_docid()] = f_q_doc
    return qlm_binary


if __name__ == "__main__":
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    # get the topic ID and corresponding topic title
    main_path = os.getcwd()
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
            topics[topic_id] = topic_title
    topics_file.close()
    # print(len(topics))

    # STEP 1 Results
    # with open("A2_RM25_Results_top10.txt", "w") as f:
    #     for topic_id, topic in topics.items():
    #         # print(topic_id)
    #         f.write("Topic " + topic_id + ": \n")
    #         f.write("\t\t DocID \t\t Weight \n")
    #         # Get the DatasetCollection
    #         for id in id_list:
    #             if topic_id == id:
    #                 # f.write(str(id) + "\n")
    #                 id = id.replace("R", "")

    #                 dataset = "Dataset" + id

    #                 dataset_path = main_path + "/DataCollection/" + dataset
    #                 # print(dataset_path)
    #                 coll_ = parse_rcv_coll(dataset_path, stop_words)
    #                 df_ = calc_df(coll_)
    #                 bm25_score = bm25(coll_, topic, df_)
    #                 score = {k: v for (k, v) in sorted(
    #                     bm25_score.items(), key=lambda x: x[1], reverse=True)}
    #                 # with open(main_path + "/bm25BinaryOutput/" + dataset+".txt", "w") as f_rel:
    #                 #     bm25_threshold = 1.0

    #                 #     for k, v in score.items():
    #                 #         if v > bm25_threshold:
    #                 #             f_rel.write(topic_id + " " + k + " 1" + "\n")
    #                 #         else:
    #                 #             f_rel.write(topic_id + " " + k + " 0" + "\n")

    #                 top10 = list(score.items())[:10]
    #                 for i in top10:
    #                     f.write("\t\t " + str(i[0]) +
    #                             " \t\t " + str(i[1]) + "\n")
    #         f.write("\n")

    # STEP 2 Results
    with open("A2_LanguageModel_Results_top10.txt", "w") as m_f:
        for topic_id, topic in topics.items():
            # print(topic_id)
            m_f.write("Topic " + topic_id + ": \n")
            m_f.write("\t\t DocID \t\t Weight \n")
            # Get the DatasetCollection
            for id in id_list:
                if topic_id == id:
                    # m_f.write(str(id) + "\n")
                    id = id.replace("R", "")

                    dataset = "Dataset" + id
                    dataset_path = main_path + "/DataCollection/" + dataset
                    # print(dataset_path)
                    coll_ = parse_rcv_coll(dataset_path, stop_words)
                    df_ = calc_df(coll_)
                    qlm_score = QLM(coll_, topic, df_)
                    qlm_results = {k: v for (k, v) in sorted(
                        qlm_score.items(), key=lambda x: x[1], reverse=True)}
                    qlm_binary_result = QLM_binary(coll_, topic, df_)
                    # with open(main_path + "/QlmBinaryOutput/" + dataset+".txt", "w") as b_f:
                    #     for k, v in qlm_binary_result.items():
                    #         if v > 0:
                    #             b_f.write(topic_id + " " + k + " 1" + "\n")
                    #         else:
                    #             b_f.write(topic_id + " " + k + " 0" + "\n")

                    qlm_top10 = list(qlm_results.items())[:10]
                    for i in qlm_top10:
                        m_f.write("\t\t " + str(i[0]) +
                                  " \t\t " + str(i[1]) + "\n")
            m_f.write("\n")
