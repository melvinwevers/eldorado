import os
#import numpy as np
import pandas as pd
from tqdm import tqdm
import stanfordnlp
import unidecode
from nltk import sent_tokenize
import re
import csv
from collections import Counter
from nltk.util import ngrams


regex_pat = re.compile(r'[^a-zA-Z\s]', flags=re.IGNORECASE)

def digit_perc(x):
    return round(sum(c.isdigit() for c in str(x)) / len(str(x)), 4)

class Corpus(object):
    """ A basic corpus class for reading documents ('content') from vanilla or tabular files
    Attributes:
        name: str representing the corpus' name
        path: str representing the relative path
    """

    def __init__(self, path, title, type_):
        self.path = path
        self.title = title
        self.type = type_


    def read(self):
        self.fpath = os.path.join(self.path, self.title, self.type)
        self.fnames = [f for f in os.listdir(self.fpath) if f.endswith('.tsv')]
        self.fpaths = [os.path.join(self.fpath, fname) for fname in self.fnames]
        self.metadata = list()

        for f in tqdm(self.fpaths):
            print(f)
            #df = pd.read_csv(f,
            #                 index_col=None, delimiter='\t', encoding='unicode-escape')
            df = pd.read_csv(f,
                             index_col=None, delimiter='\t')
            if df.empty:
                print('Nothing in here!')
                pass
            else:

                df['ocr'] = df['ocr'].str.replace(regex_pat, '')  # only words
                df['ocr'] = df['ocr'].str.findall(
                    r'\w{2,}').str.join(' ').str.lower()
                df = df[~df['ocr'].isna()]

                self.year = df['date'].values[0][:4]

                self.date = df['date'].values

                #self.identifier = df['identifier'].values
                self.text_content = df['ocr'].values



    def process(self):
        """ Reads filenames and content of file(s) on path
        Attributes:
            fnames: list of str containing filename on files on path 
            fpaths: list of str containing relative path and filename to files on path
            content: list of str containing content of files on path
        """
        self.fpath = os.path.join(self.path, self.title, self.type)
        self.fnames = [f for f in os.listdir(self.fpath) if f.endswith('.tsv')]
        self.fpaths = [os.path.join(self.fpath, fname) for fname in self.fnames]
        self.metadata = list()


        for f in tqdm(self.fpaths):
            print(f)
            #df = pd.read_csv(f,
            #                 index_col=None, delimiter='\t', encoding='unicode-escape')
            df = pd.read_csv(f,
                             index_col=None, delimiter='\t')
            if df.empty:
                print('Nothing in here!')
                pass
            else:

                df['identifier'] = df['ocr_url'].apply(
                    lambda x: x.split('/')[:][3][12:-4]) #create identifier
                #df['ocr'] = df['ocr'].apply(lambda x: x.encode('latin1').decode('utf8'))
                df['perc_digits'] = df['ocr'].apply(lambda x: digit_perc(x))
                df['ocr'] = df['ocr'].apply(lambda x: x[1:])
                df['ocr'] = df['ocr'].str.replace(regex_pat, '')  # only words
                df['ocr'] = df['ocr'].str.findall(
                    r'\w{2,}').str.join(' ').str.lower()
                df = df[~df['ocr'].isna()]

                self.year = df['date'].values[0][:4]

                self.date = df['date'].values
                
                self.identifier = df['identifier'].values
                self.text_content = df['ocr'].values

                df.drop('ocr', axis=1, inplace=True)
                df['newspaper_title'] = self.title
                self.metadata.append(df)
        
                output_f = os.path.join(self.fpath, self.title + '_' + self.date + '.tsv')
                with open(output_f, mode='w') as output_csv:
                    csv_writer = csv.writer(output_csv, delimiter='\t')
                    csv_writer.writerow(['identifier', 'text_content'])
                    csv_writer.writerows(zip(self.identifier, self.text_content))
        
        self.bigframe = pd.concat(self.metadata, axis=0, ignore_index=True)
        self.bigframe.to_csv(os.path.join(self.fpath, self.title + '_metadata.tsv'), sep='\t', index=None)

                            
    def subcorpus(self):


        words = ['cigaret*\w+', 'sigaret*\w+']
        pattern = '|'.join(words)

        self.fpath = os.path.join(self.path, self.title, self.type)
        self.fnames = [f for f in os.listdir(self.fpath) if f.endswith('.tsv')]
        self.fpaths = [os.path.join(self.fpath, fname)
                       for fname in self.fnames]
        self.metadata = list()
        self.text_content = list()
        self.date = list()

        for f in tqdm(self.fpaths):
            print(f)
            #df = pd.read_csv(f,
            #                 index_col=None, delimiter='\t', encoding='unicode-escape')
            df = pd.read_csv(f,
                             index_col=None, delimiter='\t')
            if df.empty:
                print('Nothing in here!')
                pass
            else:

                df['ocr'] = df['ocr'].str.replace(regex_pat, '')  # only words
                df['ocr'] = df['ocr'].str.findall(
                    r'\w{2,}').str.join(' ').str.lower()
                df = df[~df['ocr'].isna()]


                df = df[df['ocr'].str.contains(pattern, na=False)]
                self.date.append(df['date'].values)
                self.text_content.append(df['ocr'].values)


    

    def normalize(self, lang="nl"):
        """ linguistic normalization
        Attributes:
            lemma: list of str containing lemmas of content whitespace tokenized
        """
        self.read()
        self.tokens = list()
        self.bigrams = list()
        self.frequencies = list()
        nlp = stanfordnlp.Pipeline(
            processors='tokenize', lang=lang)
        for text in tqdm(self.text_content[:100]): 
            try:
                doc = nlp(text)
                tokens = [
                    token.text for sent in doc.sentences for token in sent.tokens]
            except:
                tokens = 'nan'
            tokens = list(filter(None, tokens))
            self.tokens.extend(tokens)

        self.bigrams = list(ngrams(self.tokens, 2))
        self.frequencies =Counter(self.bigrams)
