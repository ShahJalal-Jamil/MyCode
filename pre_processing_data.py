from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from keras.models import Sequential
import numpy

# fix random seed for reproducibility
from keras_preprocessing.text import Tokenizer

numpy.random.seed(7)

'''
Load  Training Data From Excel Sheet. Make two list of Comments and Class.
'''

file = "D:/Test/training_t_c.xlsx"
x1 = pd.ExcelFile(file)

# Print the sheet names
# print(x1.sheet_names)

# Load a sheet into a DataFrame by name: df1
sheet = x1.parse('Sheet1')
list_n = sheet['Comments']
list_class = sheet['Class']

'''

Pre processing the training data. So identify the hashtag, emoji's and plain comments.
list_main contains feature data and list_class_new contains class data.
'''


def isInAlphabeticalOrder(word):
    for i in range(len(word) - 1):
        if word[i] > word[i + 1]:
            return False
    return True


list_of_words = list()

import emoji as emo

print(emo.emoji_lis)


def char_is_emoji(character):
    return character in emo.UNICODE_EMOJI


def emo_is_character(character):
    emoValue = emo.emojize(character)
    print(emoValue)
    return emoValue in emo.EMOJI_ALIAS_UNICODE


def text_has_emoji(text):
    for character in text:
        if character in emo.EMOJI_ALIAS_UNICODE:
            return True
    return False


def char_is_hashTag(character):
    return re.match(r'#', character)


def is_ascii(s):
    print(s)
    return all(ord(c) < 256 for c in s)


def retieveEmo(sentence):
    list_sentence = sentence.split()
    for word in list_sentence:
        retWord = split_count(word)
        if (len(retWord) <= 2 and neagtionCheck(retWord) == False):
            continue
        list_sentence[list_sentence.index(word)] = retWord

    final_sentence = ' '.join(list_sentence)
    # print(final_sentence)
    return final_sentence


import regex


def split_count(text):
    emoji_counter = 0
    data = regex.findall(r'\X', text)
    chunks = []
    for word in data:
        if any(char in emo.UNICODE_EMOJI for char in word):
            emoji_counter += 1
            emoCode = emo.demojize(word)
            emoCode = emoCode.replace("_", "'")
            chunks.append(str(' ') + emoCode + str(' '))
            print(emoCode)
        else:
            chunks.append(word)
    # print(chunks)
    result = ''.join(chunks)
    # print(result)
    return result


def retieveEmoFromWord(word):
    word_item = word.split()


curcialWord = dict()
curcialWordPosition = dict()
# printing the list using loop
import re


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def leastFrequent(List):
    res = min(set(List), key=List.count)
    return res


def mostFrequent(List):
    res = max(set(List), key=List.count)
    return res


def assignPosition(pos, lenght):
    value = 2;
    if (pos == 0):
        value = 1
    elif (pos == 1):
        value = 1
    elif (pos == lenght - 2):
        value = 3
    elif (pos == lenght - 1):
        value = 3
    else:
        value = 2
    return value


def neagtionCheck(s):
    negationList = ['নেই', 'না', 'নয়', 'নি', 'নাই']
    if s in negationList:
        return True
    else:
        return False


def bestThreeSelection(splitItem):
    length = 0
    for item in splitItem:
        if (item[0] != ':' or item[len(item) - 1] != "'"):
            length = length + 1

    pos = 0
    listValue = {}
    listWeight = {}
    trackEmo = 0
    isEmo = False
    for item in splitItem:
        if (item[0] != ':' and item[len(item) - 1] != "'"):
            isEmo = False
            trackEmo = trackEmo + 1
        else:
            isEmo = True
        val, returnItem = hasSimilar(item, trackEmo, length, isEmo)
        # print(returnItem)
        listT = list()
        listT.append(val)
        listT.append(curcialWord[returnItem])
        listT.append(pos)
        listValue[pos] = listT

        # print(hasSimilar(item, pos, len(splitItem)))
        pos = pos + 1
    # sorted_x = sorted(listWeight.items(), key=lambda kv: kv[1], reverse=True)
    # print(listValue)
    sorted_dict = {k: v for k, v in sorted(listValue.items(), key=lambda item: item[0])}
    sortedValuesList = sorted(sorted_dict.values(), reverse=True)
    # print(sortedValuesList)
    # if (len(sortedValuesList) >= 3):
    listFinal = list()
    #     if (len(sortedValuesList) >= 1):
    #         firstVal = sortedValuesList[0][0]
    #         listFinal.append(splitItem[list(listValue.keys())[firstVal]])
    #     if (len(sortedValuesList) >= 2):
    #         secondVal = sortedValuesList[1][0]
    #         listFinal.append(splitItem[list(listValue.keys())[secondVal]])
    #     if (len(sortedValuesList) >= 3):
    #         thirdVal = sortedValuesList[2][0]
    #         listFinal.append(splitItem[list(listValue.keys())[thirdVal]])
    #     # if (len(sortedValuesList) >= 4):
    #     if (len(sortedValuesList) >= 4):
    #         fourth = sortedValuesList[3][0]
    #         listFinal.append(splitItem[list(listValue.keys())[fourth]])

    min = len(sortedValuesList)
    # if (min > 5):
    #     min = 5
    newDict = dict()
    for item in range(min):
        val = sortedValuesList[item][0]
        min = sortedValuesList[item][2]
        if (val >= 0):
            listFinal.append(splitItem[list(listValue.keys())[min]])
        newDict[splitItem[list(listValue.keys())[min]]] = val
    # print(listFinal)
    return listFinal, newDict

    #
    # import operator
    # sorted_Value = sorted(listValue.items(), key=operator.itemgetter(0), reverse=True)
    # print(sorted_Value)
    # sorted_dict = dict(sorted_Value)
    # sorted_dictValue = dict(sorted_Value)
    # print(list(sorted_dict.keys()))


from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def getPanel(value):
    panel = 1
    if (value == 1):
        panel = 1
    elif (value == 3):
        panel = 1
    else:
        panel = 2
    return panel


def hasSimilar(item, pos, length, isEmo):
    value = False
    sum = 0
    i = 0
    listValue = list()
    listValueMin = list()
    returnItem = item
    for im in curcialWord:
        if (item in im or im in item and im != item):
            listValue = listValue + curcialWordPosition[im]
            listValueMin = listValueMin + curcialWordPosition[im]
            # if (len(item) > len(im) and similar(item, im) > 0.90 and neagtionCheck(item) == False):
            #     returnItem = im
            # print(im, leastFrequent(curcialWordPosition[im]))
            sum = 1 + sum
        i = i + 1
    # print(pos, length, assignPosition(7, 8))
    mark = 0

    if (isEmo == True):
        mark = mark + 3

    if (getPanel(assignPosition(pos, length)) == getPanel(mostFrequent(listValue))):
        mark = mark + 1

    if (getPanel(mostFrequent(listValue)) != getPanel(leastFrequent(listValueMin))):
        mark = mark + 1

        # print(item, curcialWord[item], sum, assignPosition(pos, length), mostFrequent(listValue),
        #       leastFrequent(listValueMin))
    return mark, returnItem


def preProcessDataUnsupervised(isPd):
    print(char_is_emoji(':na:'))
    # list_mainT = list()
    # list_main = list_mainT[::-1]
    list_main = list()
    for row in list_n:
        list_main.append(row)

    # print(list_main)

    list_Class_new = list()
    for row in list_class:
        list_Class_new.append(row)

    # print(list_Class_new)

    # Remove the sentence ending statement for better training
    list_final_temp = list()
    for row in list_main:
        # list_final_temp.append(row)
        print(row)
        if ("।" in row):
            row = row.replace("।", " ")
            row = retieveEmo(row)
            # print(row)
            list_final_temp.append(row)
        else:
            row = retieveEmo(row)
            # print(row)
            list_final_temp.append(row)
    print(list_final_temp)

    for x in range(len(list_final_temp)):
        item = list_final_temp[x]
        item = item.replace("", "")
        item = item.replace(".", " ")
        item = item.replace(",", " ")
        item = item.replace("?", " ")
        item = item.replace("!", " ")
        item = item.replace("।", " ")
        splitItem = item.split()
        i = 1
        pos = 0

        for item in splitItem:
            listValue = list()
            if (item in curcialWord):
                value = curcialWord.get(item)
                value = value + 1
                curcialWord[item] = value
                listValue = curcialWordPosition.get(item)
                listValue.append(assignPosition(pos, len(splitItem)))
                curcialWordPosition[item] = listValue
            else:
                curcialWord[item] = 1
                listValue.append(assignPosition(pos, len(splitItem)))
                curcialWordPosition[item] = listValue
                # print(item, len(splitItem), i)
            i = i + 1
            pos = pos + 1

    list_final = list()
    mainDict = {}
    for x in range(len(list_final_temp)):
        item = list_final_temp[x]
        item = item.replace("", "")
        item = item.replace(".", " ")
        item = item.replace(",", " ")
        item = item.replace("?", " ")
        item = item.replace("!", " ")
        item = item.replace("।", " ")
        splitItem = item.split()
        # print(splitItem)
        if (len(splitItem) == 1 or len(splitItem) == 2):
            list_final.append(splitItem)
            dic = {}
            dic[0] = 3
            mainDict[x] = dic
        elif (len(splitItem) == 2):
            list_final.append(splitItem)
            dic = {}
            dic[0] = 3
            dic[1] = 3
            mainDict[x] = dic
        else:
            # print(splitItem)
            res, sort = bestThreeSelection(splitItem)
            # print(sort)
            mainDict[x] = sort
            list_final.append(res)

    # print(len(list_final_temp))
    # print(len(list_final))
    # print("---------------------------------------------------------------")
    # print(mainDict)
    '''
    Divide feature data into two parts. So 80% data will be used as train data and
    other 20% as close test data.
    '''
    # random.shuffle(list_final)

    train_size = int(len(list_final) * .95)

    x_Train = list_final[0:train_size]
    x_Test = list_final[train_size:len(list_final)]

    print(len(x_Train))
    print(len(x_Test))

    y_Train = list_Class_new[0:train_size]
    y_Test = list_Class_new[train_size:len(list_Class_new)]

    print(len(y_Train))
    print(len(y_Test))

    """
    Tokenize the native bangla sentence preparing  for  neural network.
    """

    max_words = 3000
    # print(docs)
    # create the tokenizer

    # print("Final List")
    # print(list_final)

    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(list_final)
    dictionary = t.word_index
    # print(t.word_index)

    X = t.texts_to_matrix(x_Train, mode='count')

    # Data for close test
    XTest = t.texts_to_matrix(x_Test, mode='count')
    # print(len(X), len(XTest))

    # Data for open test
    text = "খুবি ভাল টিশার্ট"
    text = retieveEmo(text)
    list_text = list()
    list_text.append(text)
    data_text = t.texts_to_matrix(list_text, mode='count')
    # print(list_text)

    """
    SVM Configure
    """

    # X = pd.DataFrame(X)
    # XTest = pd.DataFrame(XTest)
    #
    encoder = LabelBinarizer()
    encoder.fit(list_Class_new)
    Y = encoder.transform(y_Train)
    YTest = encoder.transform(y_Test)

    print(X)

    if (isPd):
        X = pd.DataFrame(X)
        XTest = pd.DataFrame(XTest)
        Y = pd.DataFrame(Y)
        YTest = pd.DataFrame(YTest)
        j = 0
        print("Values", X)
    for row in X.index:
        i = 0
        sort = {}
        if row in mainDict:
            sort = mainDict[row]
            # print(sort)
        for im in t.word_index:
            if (im in sort):
                if (sort[im] > 1):
                    X.set_value(row, i, sort[im] * X.get_value(row, i))
                elif (sort[im] == 1):
                    X.set_value(row, i, 1 * X.get_value(row, i))
                else:
                    X.set_value(row, i, -1)
            i = i + 1
    print("Done")
    return X, Y, XTest, YTest, data_text, t


def preProcessData(isPd):
    list_main = list()
    for row in list_n:
        list_main.append(row)

    # print(list_main)

    list_Class_new = list()
    for row in list_class:
        list_Class_new.append(row)

    # print(list_Class_new)

    # Remove the sentence ending statement for better training
    list_final = list()
    for row in list_main:
        # print(row)
        if ("।" in row):
            row = row.replace("।", " ")
            row = retieveEmo(row)
            # print(row)
            list_final.append(row)
        else:
            row = retieveEmo(row)
            # print(row)
            list_final.append(row)
    print(list_final)

    '''
    Divide feature data into two parts. So 80% data will be used as train data and
    other 20% as close test data.
    '''
    import random
    # random.shuffle(list_final)

    train_size = int(len(list_final) * .95)

    x_Train = list_final[0:train_size]
    x_Test = list_final[train_size:len(list_final)]

    print(len(x_Train))
    print(len(x_Test))

    y_Train = list_Class_new[0:train_size]
    y_Test = list_Class_new[train_size:len(list_Class_new)]

    print(len(y_Train))
    print(len(y_Test))

    """
    Tokenize the native bangla sentence preparing  for  neural network.
    """

    max_words = 3000
    # print(docs)
    # create the tokenizer

    print("Final List")
    print(list_final)

    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(list_final)
    dictionary = t.word_index
    print(t.word_index)

    X = t.texts_to_matrix(x_Train, mode='count')

    # Data for close test
    XTest = t.texts_to_matrix(x_Test, mode='count')
    print(len(X), len(XTest))

    # Data for open test
    text = "খুবি ভাল টিশার্ট"
    text = retieveEmo(text)
    list_text = list()
    list_text.append(text)
    data_text = t.texts_to_matrix(list_text, mode='count')
    print(list_text)

    """
    SVM Configure
    """

    # X = pd.DataFrame(X)
    # XTest = pd.DataFrame(XTest)
    #
    encoder = LabelBinarizer()
    encoder.fit(list_Class_new)
    Y = encoder.transform(y_Train)
    YTest = encoder.transform(y_Test)

    if (isPd):
        X = pd.DataFrame(X)
        XTest = pd.DataFrame(XTest)
        Y = pd.DataFrame(Y)
        YTest = pd.DataFrame(YTest)

    return X, Y, XTest, YTest, data_text, t


def preProcessDataNN():
    list_main = list()
    for row in list_n:
        list_main.append(row)

    # print(list_main)

    list_Class_new = list()
    for row in list_class:
        list_Class_new.append(row)

    # print(list_Class_new)

    # Remove the sentence ending statement for better training
    list_final = list()
    for row in list_main:
        # print(row)
        if ("।" in row):
            row = row.replace("।", " ")
            row = retieveEmo(row)
            print(row)
            list_final.append(row)
        else:
            row = retieveEmo(row)
            print(row)
            list_final.append(row)
    print(list_final)

    '''
    Divide feature data into two parts. So 80% data will be used as train data and
    other 20% as close test data.
    '''

    train_size = int(len(list_final) * .9)

    x_Train = list_final[0:train_size]
    x_Test = list_final[train_size:len(list_final)]

    print(len(x_Train))
    print(len(x_Test))

    y_Train = list_Class_new[0:train_size]
    y_Test = list_Class_new[train_size:len(list_Class_new)]

    print(len(y_Train))
    print(len(y_Test))

    """
    Tokenize the native bangla sentence preparing  for  neural network.
    """

    max_words = 3000
    # print(docs)
    # create the tokenizer

    print("Final List")
    print(list_final)

    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(list_final)
    dictionary = t.word_index
    print(t.word_index)

    X = t.texts_to_matrix(x_Train, mode='count')

    # Data for close test
    XTest = t.texts_to_matrix(x_Test, mode='count')
    print(len(X), len(XTest))

    # Data for open test
    text = "টি-শার্টগুলো খুবই সুন্দর "
    text = retieveEmo(text)
    list_text = list()
    list_text.append(text)
    data_text = t.texts_to_matrix(list_text, mode='count')
    print(list_text)

    """
    SVM Configure
    """

    # X = pd.DataFrame(X)
    # XTest = pd.DataFrame(XTest)
    #
    encoder = LabelBinarizer()
    encoder.fit(list_Class_new)
    Y = encoder.transform(y_Train)
    YTest = encoder.transform(y_Test)

    X = pd.DataFrame(X)
    XTest = pd.DataFrame(XTest)
    Y = pd.DataFrame(Y)
    YTest = pd.DataFrame(YTest)

    return X, Y, XTest, YTest, t.word_index, encoder


print("He", char_is_emoji("👌"))
print("He", emo_is_character(":ok_hand:"))

print(char_is_emoji("👌"))
print(emo_is_character(":ok_hand:"))
print(char_is_emoji(":ok_hand:"))


def char_is_emoji(character):
    return character in emo.UNICODE_EMOJI
