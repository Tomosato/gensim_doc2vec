import gensim
import nagisa
import smart_open
import random

# 形態素解析しながら、ドキュメントをタグ付けする。
def tagging_document(file_name, tokens_only = False):
    with smart_open.smart_open(file_name) as f :
        for i, line in enumerate(f):
            text = nagisa.tagging(line)
            text_list = text.words
            text_postags = text.postags
            listed_text = []
            for text_list_elements, text_postags_element in zip(text_list, text_postags):
                if text_postags_element == "名詞" :
                    listed_text.append(text_list_elements)
                elif text_postags_element == "動詞" : 
                    listed_text.append(text_list_elements)
                elif text_postags_element == "形状詞" :
                    listed_text.append(text_list_elements)
            
            if tokens_only :
                yield listed_text
            else :
                yield gensim.models.doc2vec.TaggedDocument(listed_text, [i])


train_corpus = list(tagging_document("train_file.txt")) # TODO: コマンドライン引数で取りたい
test_corpus = list(tagging_document("test_file.txt", tokens_only = True)) # TODO: コマンドライン引数で取りたい。


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

