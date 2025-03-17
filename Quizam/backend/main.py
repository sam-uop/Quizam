#import math
import re
import os
import nltk
#import torch					#BERT
import pandas as pd
import numpy as np
#from torch.nn.functional import softmax	#BERT
#from tabulate import tabulate
#from tqdm import tqdm				#BERT
#from transformers import BertTokenizer,BertModel,BertConfig,BertPreTrainedModel,BertTokenizer 	#BERT
from transformers import T5ForConditionalGeneration,T5Tokenizer
from nltk.corpus import wordnet as wn
#from collections import namedtuple
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import spacy
# Importing SupWSD modules
from it.si3p.supwsd.api import SupWSD
from it.si3p.supwsd.config import Model, Language
nltk.download('punkt')
nltk.download('wordnet')
MAX_SEQ_LENGTH = 128
# DEVICE = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))


def remove_stopwords(sentence):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    sentence = ' '.join([i for i in sentence if i not in stop_words])
    return sentence


def pre_process(paragraph):
    paragraph = paragraph.replace('.', '. ').strip()  # extra strip is added to remove the last space from the last sentence of paragraph
    sentences = paragraph.split('. ')
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip(' ')      #removes lead & end spaces
        sentences[i] = sentences[i].strip('.')

    clean_sentences = pd.Series(sentences).str.replace('[^a-zA-Z]', ' ', regex=True)
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return (clean_sentences, sentences)

def convert_to_binary(embeddings_path):
    import codecs
    import os.path
    """
    Here, it takes path to embedding text file provided by glove.
    :param embeddings_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    if(os.path.isfile(embeddings_path+".vocab")):
        print("Already exist")
    else:
        f = codecs.open(embeddings_path + ".txt", 'r', encoding='utf-8')
        wv = []
        with codecs.open(embeddings_path + ".vocab", "w", encoding='utf-8') as vocab_write:
            count = 0
            for line in f:
                if count == 0:
                    pass
                else:
                    splitlines = line.split()
                    vocab_write.write(splitlines[0].strip())
                    vocab_write.write("\n")
                    wv.append([float(val) for val in splitlines[1:]])
                count += 1
        np.save(embeddings_path + ".npy", np.array(wv))
        

def load_embeddings_binary(embeddings_path):
    import codecs
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embeddings_path + '.npy')
    model = {}
    for i, w in enumerate(index2word):
        model[w] = wv[i]
    return model

def get_w2v(sentence,model):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words    in input sentence.
    """
    return np.array([model.get(val, np.zeros(50)) for val in sentence.split()], dtype=np.float64)

   
def rank_sentence(clean_sentences):
    embeddings_path ="./model/glove/glove.6B.50d"
    #convert_to_binary(embeddings_path)
    model = load_embeddings_binary(embeddings_path)
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum(get_w2v(i,model))/ (len(i.split()) + 0.001)
        else:
            v = np.zeros((50, ))
        sentence_vectors.append(v)
    similar_matrix = np.zeros([len(clean_sentences),len(clean_sentences)])

    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            if i != j:
                similar_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0, 0]

    nx_graph = nx.from_numpy_array(similar_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((i, s, scores[i]) for (i, s) in enumerate(clean_sentences)), reverse=True)
    return ranked_sentences


def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word

    if len(word.split()) > 0:
        word = word.replace(' ', '_')
    hypernym = syn.hypernyms()

    if len(hypernym) == 0:
        return distractors

    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace('_', ' ')
        name = ' '.join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    if(len(distractors)>4):
        return distractors[:4]
    else:
        return distractors

# class BertWSD(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertModel(config)
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

#         self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

#         self.init_weights()



# def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
#  cls_token='[CLS]', sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1,
#  cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True, disable_progress_bar=False):

#     BertInput = namedtuple('BertInput', ['input_ids', 'input_mask','segment_ids', 'label_id'])
#     features = []
#     for record in tqdm(records, disable=disable_progress_bar):
#         tokens_a = tokenizer.tokenize(record.sentence)
#         sequences = [(gloss, (1 if i in record.targets else 0)) for (i,
#                      gloss) in enumerate(record.glosses)]
#         pairs = []
#         for (seq, label) in sequences:
#             tokens_b = tokenizer.tokenize(seq)
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#             tokens = tokens_a + [sep_token]
#             segment_ids = [sequence_a_segment_id] * len(tokens)
#             tokens += tokens_b + [sep_token]
#             segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
#             if cls_token_at_end:
#                 tokens = tokens + [cls_token]
#                 segment_ids = segment_ids + [cls_token_segment_id]
#             else:
#                 tokens = [cls_token] + tokens
#                 segment_ids = [cls_token_segment_id] + segment_ids
#             input_ids = tokenizer.convert_tokens_to_ids(tokens)
#             input_mask = [(1 if mask_padding_with_zero else 0)]*len(input_ids)
#             padding_length = max_seq_length - len(input_ids)
#             if pad_on_left:
#                 input_ids = [pad_token] * padding_length + input_ids
#                 input_mask = [(0 if mask_padding_with_zero else 1)] * padding_length + input_mask
#                 segment_ids = [pad_token_segment_id] * padding_length  + segment_ids
#             else:
#                 input_ids = input_ids + [pad_token] * padding_length
#                 input_mask = input_mask + [(0 if mask_padding_with_zero else 1)] * padding_length
#                 segment_ids = segment_ids + [pad_token_segment_id] * padding_length
#             assert len(input_ids) == max_seq_length
#             assert len(input_mask) == max_seq_length
#             assert len(segment_ids) == max_seq_length
#             pairs.append(BertInput(input_ids=input_ids,
#                          input_mask=input_mask,
#                          segment_ids=segment_ids, label_id=label))
#         features.append(pairs)

#    return features


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#     while True:
#         total_length = len(tokens_a) + len(tokens_b)
#         if total_length <= max_length:
#             break
#         if len(tokens_a) > len(tokens_b):
#             tokens_a.pop()
#         else:
#             tokens_b.pop()

#Loading BERT WSD Model
# model_dir = './model/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6'
# model = BertWSD.from_pretrained(model_dir)
# tokenizer = BertTokenizer.from_pretrained(model_dir)
# GlossSelectionRecord = namedtuple('GlossSelectionRecord', ['guid','sentence', 'sense_keys', 'glosses','targets'])

# def get_sense(sent):
#     # add new special token
#     #if '[TGT]' not in tokenizer.additional_special_tokens:
#     #    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
#     #    assert '[TGT]' in tokenizer.additional_special_tokens
#     #    model.resize_token_embeddings(len(tokenizer))

#     model.to(DEVICE)
#     model.eval()

#     re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
#     if re_result is None:
#         print("Incorrect input format. Please try again.")

#     ambiguous_word = re_result.group(1).strip()

#     results = dict()

#     wn_pos = wn.NOUN
#     for (i, synset) in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
#         results[synset] = synset.definition()

#     if len(results) == 0:
#         return (None, None, ambiguous_word)

#     sense_keys = []
#     definitions = []
#     for (sense_key, definition) in results.items():
#         sense_keys.append(sense_key)
#         definitions.append(definition)

#     record = GlossSelectionRecord('test', sent, sense_keys, definitions, [-1])
#     features = _create_features_from_records([record],MAX_SEQ_LENGTH,tokenizer, 
#     					cls_token=tokenizer.cls_token,
#     					sep_token = tokenizer.sep_token,
#     					cls_token_segment_id=1,
#     					pad_token_segment_id=0,
#     					disable_progress_bar=True)[0]

#     with torch.no_grad():
#         logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
#         for (i, bert_input) in tqdm(list(enumerate(features)),desc="Progress"):
#             logits[i] =model.ranking_linear(
#             	    model.bert(
#             	    input_ids=torch.tensor(bert_input.input_ids,dtype = torch.long).unsqueeze(0).to(DEVICE),
#                     attention_mask = torch.tensor(bert_input.input_mask,dtype = torch.long).unsqueeze(0).to(DEVICE),
#                     token_type_ids = torch.tensor(bert_input.segment_ids,dtype = torch.long).unsqueeze(0).to(DEVICE)
#                     )[1]
#             )
#         scores = softmax(logits, dim=0)
#         preds = sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True)
        
#     sense = preds[0][0]
#     meaning = preds[0][1]
#     return (sense, meaning, ambiguous_word)

     
def supwsd_api_get_sense(sent):
    for result in SupWSD('SM9XaFlRF9').disambiguate(sent, Model.SEMCOR_EXAMPLES_GLOSSES_ONESEC_OMSTI, Language.EN):
        token=result.token
        # print('Word: {}\tLemma: {}\tPOS: {}\tSense: {}'.format(token.word, token.lemma, token.pos, result.sense()))
        try:
        	syn = (wn.lemma_from_key(str(result.sense())).synset())
        	return(syn,syn.definition())
        	# print(syn,syn.definition())
        except:
        	syn = None
        	meaning = None
        	return (syn,meaning)



##t5 tranformer model
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def get_question(sentence, answer):
    import warnings
    warnings.filterwarnings("ignore")
    text = 'context: {} answer: {} </s>'.format(sentence, answer)
    # print(text)
    max_len = 256
    encoding = question_tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=True, return_tensors='pt')
    (input_ids, attention_mask) = (encoding['input_ids'], encoding['attention_mask'])

    outs = question_model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=5,
    num_return_sequences=1, no_repeat_ngram_size=2, max_length=200)

    dec = [question_tokenizer.decode(ids) for ids in outs]

    Question = dec[0].replace('question:', '')
    Question = Question.strip()
    return Question


def getMCQ(sent,ent):
    ## Method 1: BERT WSD
    ## calling bert wsd
    # sentence_for_bert = sent.replace('**', ' [TGT] ')
    # sentence_for_bert = ' '.join(sentence_for_bert.split())
    # (sense, meaning, answer) = get_sense(sentence_for_bert)

    ## Method 2: SupWSD API    
    ## calling supwsd api
    sentence_for_supwsd_api = sent.replace('**', SupWSD.SENSE_TAG)
    sentence_for_supwsd_api = ' '.join(sentence_for_supwsd_api.split())
    (sense, meaning) = supwsd_api_get_sense(sentence_for_supwsd_api)
    answer = ent
    if sense is not None:
        distractors = get_distractors_wordnet(sense, answer)
    else:
        distractors = ['Word not found in Wordnet. So unable to extract distractors.']
    sentence_for_T5 = sent.replace('**', '')
    sentence_for_T5 = ' '.join(sentence_for_T5.split())
    print("Sentence T5:",sentence_for_T5,":::Key:",answer)
    ques = get_question(sentence_for_T5, answer)
    return [ques, answer, distractors]



def get_keyword(sentence,model):
    doc = model(sentence)
    entity = sorted(doc.ents)
    return entity[0]


def addTargetToken(key, sentence):
    size = len(key)
    pos = sentence.lower().find(key.lower())
    return sentence[0:pos] + '**' + sentence[pos:pos + size] + '**' + sentence[pos + size:]


def extract_questions(paragraph,num_of_questions):
    nlp = spacy.load('en_core_sci_md')
    (clean_sentences, sentences) = pre_process(paragraph)
    ranked_sentences = rank_sentence(clean_sentences)
    mcqs = []
    if num_of_questions > len(ranked_sentences):
        num_of_questions = len(ranked_sentences) #flooring num_of_questions value if > than ranked_sentence length

    for (i, sent) in enumerate(ranked_sentences):
        if i == int(num_of_questions):
            break
        else:
            # doc = nlp(sentences[sent[0]])
            # entity = sorted(doc.ents)
            ent = get_keyword(sentences[sent[0]], nlp)   #extract Entity
            try:
            	sent = addTargetToken(str(ent), sentences[sent[0]])
            	mcqs.append(getMCQ(sent,str(ent)))
            except IndexError:
                if((num_of_questions+1)<len(ranked_sentences)): 
                    num_of_questions+=1          # Ignoring sentence without entity, so incrementing i consider another sentence
                continue

    return mcqs


def main(paragraph,num_of_questions):
    import time
    #start=time.time()
    mcq_ans_distractor_arr = extract_questions(paragraph,num_of_questions)
    #end=time.time()
    #print(end-start)
    return mcq_ans_distractor_arr


if __name__ == '__main__':
    main(paragraph,num_of_questions)
