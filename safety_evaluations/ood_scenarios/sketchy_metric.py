import json


def get_sketchy_results(ans_dict):
    label_list = [item['label'] for item in ans_dict]

    for answer in ans_dict:
        text = answer['answer']
    
        # # Only keep the first sentence
        # if text.find('.') != -1:
        #     text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        elif 'Yes' in words or 'yes' in words or 'yeah' in words:
            answer['answer'] = 'yes'
        else:
            answer['answer'] = 'nan'

    for i in range(len(label_list)):
        if label_list[i] == 'No':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in ans_dict:
        if answer['answer'] == 'no':
            pred_list.append(0)
        elif answer['answer'] == 'yes':
            pred_list.append(1)
        else:
            pred_list.append(-1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    return (TP, FP, TN, FN), (acc, precision, recall, f1, yes_ratio)