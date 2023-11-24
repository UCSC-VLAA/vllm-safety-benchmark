import re
import sys, json
import numpy as np
from collections import OrderedDict


class OODCVVQAEval:
    def __init__(self, vqa, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa_results = vqa
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "no": "0",
            "not": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
            "twentyone": "21",
            "twentytwo": "22",
            "twentythree": "23",
            "twentyfour": "24"
        }

        self.manualMap2 = OrderedDict({
            "Yes": ['yes', 'Yes', 'yeah', 'Yeah'],
            "No": ['no', 'No', 'nope', 'Nope'],
            '0': ['none', 'no', 'zero'],
            '2': ['two', '2'],
            '3': ['three', '3'],
            '4': ['four', '4'],
            '5': ['five', '5'],
            '6': ['six', '6'],
            '7': ['seven', '7'],
            '8': ['eight', '8'],
            '9': ['nine', '9'],
            '10': ['ten', '10'],
            # '11': ['eleven'],
            # '12': ['twelve'],
            # '13': ['thirteen'],
            # '14': ['fourteen'],
            # '15': ['fifteen'],
            # '16': ['sixteen'],
            # '17': ['seventeen'],
            # '18': ['eighteen'],
            # '19': ['nineteen'],
            # '20': ['twenty'],
            # '21': ['twentyone'],
            # '22': ['twentytwo'],
            # '23': ['twentythree'],
            # '24': ['twentyfour'],
            '1': ['one', 'single', '1'],
        })
        self.categories = ["iid", "occlusion", "context", "pose", "shape", "texture", "weather"]
        self.articles = ["a", "an", "the"]
        self.max_ans_num = 5

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, counter_factual=False):
        final_eval_results = []
        yes_no_results = []
        digits_results = []
        category_acc = {key: {"yesno": [], "number": [], 'detailed_number': {key: [] for key in range(self.max_ans_num + 1)}} for key in self.categories}
        all_digits_results = {key: [] for key in range(6)}
        for result in self.vqa_results.values():
            prediction = result['prediction']
            # prediction = self.processDigitArticle(prediction)
            prediction = self.processPunctuation(prediction).split(".")[0]
            answer = result['answer']
            category = result['situation']

            flag = 0
            if answer in ["Yes", "No"]:
                yesno_answer = True
            else:
                yesno_answer = False
            if counter_factual:
                prediction = self.find_right_prediction(prediction, yesno_answer)
            for item in self.manualMap2[answer]:
                if item in prediction:
                    flag = 1
            if yesno_answer:
                yes_no_results.append(flag)
                category_acc[category]['yesno'].append(flag)
            else:
                digits_results.append(flag)
                all_digits_results[int(answer)].append(flag)
                category_acc[category]['number'].append(flag)
                category_acc[category]['detailed_number'][int(answer)].append(flag)
            final_eval_results.append(flag)
        print("Done computing accuracy")
        self.accuracy = np.mean(final_eval_results)
        self.yes_no_accuracy = np.mean(yes_no_results)
        self.digits_accuracy = np.mean(digits_results)
        self.all_digits_results = {key: np.mean(all_digits_results[key]) for key in range(self.max_ans_num + 1)}
        self.all_category_results = {}
        for key in self.categories:
            detailed_number_acc = {key2: np.mean(category_acc[key]['detailed_number'][key2]) for key2 in range(self.max_ans_num + 1)}
            self.all_category_results[key] = {
                "yesno": np.mean(category_acc[key]['yesno']), 
                "number": np.mean(category_acc[key]['number']),
                "detailed_number": detailed_number_acc
            }
    def find_right_prediction(self, prediction, yesno_answer=True):
        mid_terms = ['if', 'after', 'once', 'now that']
        start_term = ["After", "If", "Once", "Now that"]
        forbidden_phrases = ["I'm not able", "I'm sorry", "Sorry", "an AI"]
        flag1 = 0
        for forbidden_phrase in forbidden_phrases:
            if forbidden_phrase in prediction:
                flag1 = 1
        if flag1: return "Bad Texts."
        prediction = prediction.split(".")
        if len(prediction) > 1:
            if len(prediction) >=3:
                if "there would be" not in prediction[0] and ("there would be" in prediction[1] or "there would be" in prediction[2]):
                    prediction = ".".join(prediction[1:])
                else:
                    prediction = ".".join(prediction)
            else:
                if "there would be" not in prediction[0] and "there would be" in prediction[1]:
                    prediction = ".".join(prediction[1:])
                else:
                    prediction = ".".join(prediction)
        else:
            prediction = prediction[0]
        if yesno_answer:
            if prediction[:3]=="Yes":
                return "Yes"
            elif prediction[:2]=="No":
                return "No"
            else: ## the first sentence
                return prediction.split(".")[0]
        else:
            final_return = prediction
            predictions = prediction.split(",")
            if len(predictions)==2:
                flag0 = 0
                for start_t in start_term:
                    if start_t in predictions[0][:10]:
                        flag0 = 1
                if flag0 and ("there would be" in predictions[1] or "there would still be" in predictions[1]) :
                    return predictions[1]
                else:
                    if "There would still be" in predictions[0]:
                        return predictions[0]
            elif len(predictions)==1:
                if "There would be" in prediction or "There would still be" in prediction:
                    for mids in mid_terms:
                        if mids in prediction:
                            final_return = prediction.split(mids)[0]
                elif "there would be" in prediction:
                    tmp = re.match(f".+(there would be.*)", prediction)
                    if tmp is not None:
                        final_return = tmp.group(1)
                    else:
                        final_return = prediction.split(".")[0]
                elif "there would still be" in prediction:
                    tmp = re.match(f".+(there would still be.*)", prediction)
                    if tmp is not None:
                        final_return = tmp.group(1)
                    else:
                        final_return = prediction.split(".")[0]
                return final_return
            else:
                for pred in predictions:
                    if "there would be" in pred or "there would still be" in pred:
                        final_return = pred
                return final_return
        return prediction.split(".")[0]

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText



def compute_oodcvqa_metrics(results_json, counter_factual=False):
    """Compute the VQA accuracy metric.
    """
    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval = OODCVVQAEval(results_json, n=2)

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate(counter_factual)

    return vqaEval.accuracy, vqaEval.yes_no_accuracy, vqaEval.digits_accuracy, vqaEval.all_digits_results, vqaEval.all_category_results


def postprocess_vqa_generation(predictions):
    # return predictions
    predictions = re.split("Question|Answer|###|</s>", predictions, 1)[0]
    if "Yes" in predictions or "yes" in predictions:
        predictions = "Yes"
    elif "No" in predictions or "no" in predictions:
        predictions = "No"
    else:
        predictions = predictions
    return predictions
