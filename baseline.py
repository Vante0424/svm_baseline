import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
import pandas as pd
import sklearn.svm as svm


def getOriginalLabels(filename):
    total_words = []
    total_labels = []
    f = open(filename, 'r', encoding='utf-8')
    for line in f.readlines():
        line = eval(line)

        tokens = line['tokens']
        labels = line['labels']

        for token in tokens:
            total_words.append(token)
        for label in labels:
            total_labels.append(label)
    f.close()
    # print(len(total_words))
    # print(len(total_labels))

    return total_words, total_labels


def getText(filename):
    # print('Getting text...')
    train_text = []
    f = open(filename, 'r', encoding='utf-8')
    for line in f.readlines():
        line = line.strip()
        if len(line) > 0:
            train_text.append(line)
    f.close()
    # print('Text Getting Done!')

    return train_text


def getClassFeatures(S):
    # print('Getting Classification Features...')
    tokens = []
    tags = []
    for i in range(len(S)):
        token = word_tokenize(S[i])
        tokens.append(token)
        tag = pos_tag(token)
        tags.append(tag)
    # print('tokens - {}'.format(tokens))
    # print('tags - {}'.format(tags))
    corpus = TextCollection(tokens)

    ne_chunked_sents = [nltk.ne_chunk(tag) for tag in tags]
    # print(ne_chunked_sents)

    named_entities = []
    for ne_tagged_sentence in ne_chunked_sents:
        for tagged_tree in ne_tagged_sentence:
            if hasattr(tagged_tree, 'label'):
                entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  # get NE name
                entity_type = tagged_tree.label()  # get NE category
                named_entities.append((entity_name, entity_type))
                named_entities = list(set(named_entities))

    entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
    # print(entity_frame)
    # print('named_entities - {}'.format(named_entities))

    '''
    word features:
    1. the average tf-idf score
    2. the number of important ORG
    3. the number of important PER
    4. the number of ORG
    5. the number of PER
    6. the number of stop words
    7. the number of verbs
    8. the number of '!''?'
    '''
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    classification_features = []
    for sentence in S:

        imp_organs = []
        imp_pers = []
        organs = []
        pers = []
        stop_num = 0
        verb_num = 0
        tfidf_sum = 0
        mark_num = 0

        splitted_sentence = sentence.split()
        if len(splitted_sentence) == 0:
            continue
        for i in range(len(splitted_sentence)):
            for named_entity in named_entities:
                if splitted_sentence[i].lower() == named_entity[0].lower() and splitted_sentence[i].lower() in codebook:
                    if named_entity[1] == 'ORGANIZATION':
                        imp_organs.append(splitted_sentence[i].lower())
                    elif named_entity[1] == 'PERSON':
                        imp_pers.append(splitted_sentence[i].lower())

                if splitted_sentence[i].lower() == named_entity[0].lower():
                    if named_entity[1] == 'ORGANIZATION':
                        organs.append(splitted_sentence[i].lower())
                    elif named_entity[1] == 'PERSON':
                        pers.append(splitted_sentence[i].lower())
            if splitted_sentence[i] in stop_words:
                stop_num += 1
            if splitted_sentence[i] == '?' or splitted_sentence[i] == '!':
                mark_num += 1
            tfidf_sum += corpus.tf_idf(splitted_sentence[i], sentence)
        tfidf_aver = tfidf_sum / len(splitted_sentence)

        for i in range(len(tags)):
            for j in range(len(tags[i])):
                if tags[i][j][1] == 'V' or tags[i][j][1] == 'VV':
                    verb_num += 1

        imp_organs = list(set(imp_organs))
        imp_pers = list(set(imp_pers))
        organs = list(set(organs))
        pers = list(set(pers))

        # print('imp_organs - {}'.format(imp_organs))
        # print('imp_persons - {}'.format(imp_pers))
        # print('organs - {}'.format(organs))
        # print('persons - {}'.format(pers))

        # k值
        k1 = len(imp_organs)
        k2 = len(imp_pers)
        k3 = len(organs)
        k4 = len(pers)
        k5 = stop_num
        k6 = tfidf_aver
        k7 = verb_num
        k8 = mark_num

        classification_features.append([k1, k2, k3, k4, k5, k6, k7, k8])
    # print(classification_features)
    # print('Getting Classification Features Done!')
    # print(classification_features)

    return tokens, tags, classification_features


def svmClassifier(train_features, pred_features):
    # print('Training SVM...')
    svm_model = svm.SVC()
    x_train = train_features  # 训练数据
    y_train = svm_labels  # 标签
    svm_model.fit(x_train, y_train)

    # print('SVM Predicting...')
    x_pred = pred_features
    y_pred = svm_model.predict(x_pred)
    # print(y_pred)

    f0 = open('svm_prediction.txt', 'w', encoding='utf-8')
    for item in y_pred:
        f0.write(str(item) + '\n')

    # print('SVM Classifying Done!')
    return y_pred


def BIO(pred_labels):
    f1 = open('train__.txt', 'r', encoding='utf-8')
    f2 = open('result.txt', 'w', encoding='utf-8')

    i = 0
    while i < len(pred_labels):
        for line in f1.readlines():
            line = line.strip()
            final_rst = dict()
            labels = []

            tokens = word_tokenize(line)
            # print(token)
            final_rst['tokens'] = tokens

            for j in range(len(tokens)):
                labels.append(pred_labels[i + j])

            i += len(tokens)

            final_rst['labels'] = labels
            f2.write(str(final_rst) + '\n')
    f1.close()
    f2.close()


if __name__ == "__main__":
    print(0)
    text = 'train__.txt'
    svm_train = 'svm_train__.txt'
    svm_label = 'svm_label__.txt'

    print(1)
    svm_labels = []
    with open(svm_label, 'r', encoding='utf-8') as f:
        for label in f.readlines():
            svm_labels.append(label[:-1])
    # print(svm_labels)
    # print('SVM Labels Ready!')

    with open('codebook.txt', 'r', encoding='utf-8')as f:
        codebook = []
        for line in f.readlines():
            line = line.strip()
            codebook.append(line)

    print(2)
    train_text = getText(text)
    print(train_text)
    total_words, total_labels = getOriginalLabels(svm_train)
    # print(total_words)
    # print(total_labels)
    tokens, tags, classification_features = getClassFeatures(train_text)

    print(3)
    _, _, svm_train_features = getClassFeatures(train_text)
    print(len(svm_train_features))
    print(svm_train_features)

    print(4)
    pred_labels = svmClassifier(svm_train_features, classification_features)
    print(pred_labels)
    exit()

    print(5)
    BIO(pred_labels)

    print('Done!')
