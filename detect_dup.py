import nltk
from nlp.tokenizer import GeneralTokenizer
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from fuzzywuzzy import fuzz


def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))

STOP_WORDS = set("a,about,above,after,again,against,all,am,an,and,any,are,aren't,as,at,be,because,been,before,being,below," \
             "between,both,but,by,can't,cannot,could,couldn't,did,didn't,do,does,doesn't,doing,don't,down,during,each," \
             "few,for,from,further,had,hadn't,has,hasn't,have,haven't,having,he,he'd,he'll,he's,her,here,here's,hers," \
             "herself,him,himself,his,how,how's,i,i'd,i'll,i'm,i've,if,in,into,is,isn't,it,it's,its,itself,let's,me," \
             "more,most,mustn't,my,myself,no,nor,not,of,off,on,once,only,or,other,ought,our,ours,out,over,own,same," \
             "shan't,she,she'd,she'll,she's,should,shouldn't,so,some,such,than,that,that's,the,their,theirs,them," \
             "themselves,then,there,there's,these,they,they'd,they'll,they're,they've,this,those,through,to,too," \
             "under,until,up,very,was,wasn't,we,we'd,we'll,we're,we've,were,weren't,what,what's,when,when's,where," \
             "where's,which,while,who,who's,whom,why,why's,with,won't,would,wouldn't,you,you'd,you'll,you're,you've," \
             "your,yours,yourself,yourselves,ourselves".split(','))

tokenizer = GeneralTokenizer()

stop_words = nltk_stopwords().union(STOP_WORDS)


def generate_ngram(words, min_ngram, max_ngram):
    result = []
    for i in range(min_ngram, max_ngram + 1):
        result += (' '.join(_) for _ in nltk.ngrams(words, i))

    return result


def check_dup2((ques_id, text1, text2)):
    text1 = str(text1) if type(text1) is float or type(text1) is int else text1
    text2 = str(text2) if type(text2) is float or type(text2) is int else text2
    return ques_id, round(float(fuzz.ratio(text1, text2)) / 100, 1)


def check_dup3((ques_id, text1, text2)):
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    ret = (ques_id, round(float(fuzz.token_set_ratio(' '.join(tokens1), ' '.join(tokens2))) / 100, 1))
    print '- Dup result for question id: %s, %s' % ret
    return ret


def check_dup4((ques_id, text1, text2)):
    return ques_id, 0.6


def check_dup((ques_id, text1, text2)):
    tokens1 = set(generate_ngram([w for w in tokenizer.tokenize(text1) if w], 1, 3))
    tokens2 = set(generate_ngram([w for w in tokenizer.tokenize(text2) if w], 1, 3))
    union = len(tokens1 | tokens2)
    ret = (ques_id, round(float(len(tokens1 & tokens2)) / union, 1)) if union else (ques_id, 0.0)
    print '- Dup result for question id: %s, %s' % ret
    return ret


if __name__ == '__main__':
    file_test = '/home/diepdt/Dropbox/STUDY/DATA-SCIENCE/kaggle/quora-duplicated-questions/test.csv'
    file_test_out = '/home/diepdt/Dropbox/STUDY/DATA-SCIENCE/kaggle/quora-duplicated-questions/test.csv.out4'
    print 'Start read file...'
    df_in = pd.read_csv(file_test, encoding='utf-8')
    print 'Finish read file...'
    pairs = []
    print 'Start load data...'
    pairs = df_in.values.tolist()
    print 'Finish load data...'
    print 'Start check dup...'
    pool = Pool(cpu_count() * 1)
    result = pool.map(check_dup4, pairs)
    print 'Finish check dup...'

    df_out = pd.DataFrame(result, columns=['test_id', 'is_duplicate'])
    df_out.to_csv(file_test_out, index=False, encoding='utf-8')
