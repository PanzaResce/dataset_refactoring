import re
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from collections import defaultdict

aggregation_mapping = {
    "fair": ["not_clause", "a1", "ch1", "cr1", "j1", "law1", "ltd1", "ter1", "use1", "pinc1"],
    "a": ["a2", "a3"],
    "ch": ["ch2", "ch3"],
    "cr": ["cr2", "cr3"],
    "j": ["j2", "j3"],
    "law": ["law2", "law3"],
    "ltd": ["ltd2", "ltd3"],
    "ter": ["ter2", "ter3"],
    "use": ["use2", "use3"],
    "pinc": ["pinc2", "pinc3"]    
}

def get_tags_id(list_tags_file):
    """
        Get id and tags association according to original corpus
        list_tags_file: file containing the list of all tags (unfair + fair)
    """
    tags_to_id = {"not_clause": 0}
    id_to_tags = {0: "not_clause"}
    with open(list_tags_file) as f:
        for idx, line in enumerate(f):
            values = line.strip().split(" ")
            for v in values:
                tags_to_id[v] = idx+1
                id_to_tags[idx+1] = v
    
    return tags_to_id, id_to_tags

def load_document_list(doc_list_file):
    doc_list = []      

    with open(doc_list_file) as f:
        for line in f:
            doc_list.append(line.strip())
    return doc_list

def retrieve_labels_per_document(document_list, labels_dir, tags_to_id):
    labels = {}         
    for doc in document_list:
        labels[doc] = []
        with open(os.path.join(labels_dir, doc)) as f:
            for line in f:
                tags = line.strip().split()
                tags_ids = [tags_to_id[tag] for tag in tags]
                if len(tags_ids) > 0:
                    labels[doc].append(tags_ids)
                else:
                    labels[doc].append([0])
    return labels

def count_unfair_labels(all_labels, tags_to_id):
    fair_ids = [tags_to_id[tag] for tag in aggregation_mapping["fair"]]
    total = 0
    for labels in all_labels:
        total += sum([1 for l in labels if l not in fair_ids])
    return total

def count_unfair_clauses(all_labels, tags_to_id):
    fair_ids = [tags_to_id[tag] for tag in aggregation_mapping["fair"]]
    total = 0
    for labels in all_labels:
        num_fair = sum([1 for l in labels if l in fair_ids])
        total+=bool(len(labels)-num_fair)
    return total

def retrieve_clauses_per_document(document_list, sentences_dir):
    doc_sentences = {} 
    for doc in document_list:
        doc_sentences[doc] = []
        with open(os.path.join(sentences_dir, doc)) as f:
            for line in f:
                line = re.sub('[0-9][0-9.,-]*', 'SPECIALNUMBER', line.strip().lower())
                doc_sentences[doc].append(line)
    return doc_sentences

def is_label_fair(label, tags_to_id):
    """ List of labels contains only fair tags """
    fair_ids = [tags_to_id[tag] for tag in aggregation_mapping["fair"]]
    for l in label:
        if l not in fair_ids:
            return False
    return True


def number_of_tags_per_clause(all_labels, tags_to_id, only_unfair=False):
    counts = {count: 0 for count in range(1,5)}
    for labels in all_labels:
        if only_unfair:
            if not is_label_fair(labels, tags_to_id):
                counts[len(labels)] += 1
        else:
            counts[len(labels)] += 1

    return counts 


def get_sentences_labels(doc_list_file, sentences_dir, labels_dir, tags_to_id):
    """
        Get all sentences and labels from the corpus
    """

    # List of documents by file name
    doc_list = load_document_list(doc_list_file)

    # key = document name, value = list of sentences
    doc_sentences = retrieve_clauses_per_document(doc_list, sentences_dir)
    
    # key = document name, value = list of tags for each sentence
    labels = retrieve_labels_per_document(doc_list, labels_dir, tags_to_id) 


    doc_to_id = {}
    id_to_doc = {}

    documents_id = []
    all_sentences = []
    all_labels = []
    mask_train = []

    count_documents = 0
    for doc in doc_list:
        for s in doc_sentences[doc]:
            doc_to_id[doc] = count_documents
            id_to_doc[count_documents] = doc
            all_sentences.append(s)
            documents_id.append(count_documents)
            # Do not train with sentences of len <= 5
            if len(s.split()) <= 5:
                mask_train.append(False)
            else:
                mask_train.append(True)
        for l in labels[doc]:
            all_labels.append(l)
        count_documents += 1
    
    return all_sentences, all_labels, documents_id, id_to_doc

def get_full_dataset(sentences, labels, documents, aggregate_tags=False, id_to_tags={}, multi_class=False):
    if aggregate_tags:
        if id_to_tags == {}:
            print("Provide also id_to_tags if aggregate_tags is True")
            return

        aggreg_label_to_id = {
            "fair": 0,
            "a": 1,
            "ch": 2,
            "cr": 3,
            "j": 4,
            "law": 5,
            "ltd": 6,
            "ter": 7,
            "use": 8,
            "pinc": 9 
        }

        aggreg_labels = []

        for label_list in labels:
            new_tags = []
            for l in label_list:
                for k,v in aggregation_mapping.items():
                    if id_to_tags[l] in v:
                        new_tags.append(aggreg_label_to_id[k])
            aggreg_labels.append(new_tags)

        # Some clauses may be tagged as fair for one category and unfair for another (e.g. <law1><a2> ... clause ... <a2><law1>)
        # If the task does not consider the multi-class, then these clauses have to be marked as unfair
        if not multi_class:
            for tags in aggreg_labels:
                if 0 in tags and len(tags) != 1:
                    del(tags[tags.index(0)])

        return Dataset.from_dict({'text': sentences, 'labels': aggreg_labels, 'doc': documents}), aggreg_label_to_id
    else:
        return Dataset.from_dict({'text': sentences, 'labels': labels, 'doc': documents}), id_to_tags

def train_val_test_split(dataset, seed=666):
    """
        Split the dataset by keeping each document in the same split.
        Return also the document's indices associated with their split. 
    """

    np.random.seed(seed)
    
    # Group elements by 'doc' 
    doc_groups = defaultdict(list)
    for i, example in enumerate(dataset):
        doc_groups[example['doc']].append(i)

    # Shuffle the groups
    doc_ids = list(doc_groups.keys())
    np.random.shuffle(doc_ids)

    # Define the split ratios among documents
    train_ratio = 0.6
    val_ratio = 0.25
    test_ratio = 0.15

    # Split the doc based on the ratios
    num_docs = len(doc_ids)
    train_end = int(train_ratio * num_docs)
    val_end = train_end + int(val_ratio * num_docs)

    train_docs = doc_ids[:train_end]
    val_docs = doc_ids[train_end:val_end]
    test_docs = doc_ids[val_end:]

    # Get indices
    train_indices = [idx for doc in train_docs for idx in doc_groups[doc]]
    val_indices = [idx for doc in val_docs for idx in doc_groups[doc]]
    test_indices = [idx for doc in test_docs for idx in doc_groups[doc]]

    print(f"#train docs: {len(train_docs)}")
    print(f"#val docs: {len(val_docs)}")
    print(f"#test docs: {len(test_docs)}")

    # Create the new splits
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    return dataset_dict, train_indices, val_indices, test_indices

def unfair_clauses_per_document(all_labels, all_documents, id2doc, ordered=True):
    unfair_doc = {id2doc[doc]: 0 for doc in all_documents}
    for lab, doc in zip(all_labels, all_documents):
        if 0 not in lab:
            unfair_doc[id2doc[doc]] += 1
    if ordered:
        return {k: v for k, v in sorted(unfair_doc.items(), key=lambda item: item[1], reverse=True)}
    else:
        return unfair_doc

def unfair_document_per_categories(labels_per_doc, id_to_tags):
    unfair_doc_freqs = {k:0 for k in aggregation_mapping.keys()}

    def find_aggregated_tag(tag):
        for k,v in aggregation_mapping.items():
            if tag in v:
                return k

    for doc in labels_per_doc.keys():
        already_counted = {k:False for k in aggregation_mapping.keys()}
        labels_in_doc = labels_per_doc[doc]
        for lbs in labels_in_doc:
            for l in lbs:
                tag = id_to_tags[l]
                aggregated_tag = find_aggregated_tag(tag)
                if not already_counted[aggregated_tag]:
                    unfair_doc_freqs[aggregated_tag] += 1
                    already_counted[aggregated_tag] = True
    return unfair_doc_freqs

def get_tags_frequencies(dataset, tags_to_id, split=None):
    tags_freq = {k:0 for k,v in tags_to_id.items()}
    id_to_tags = {v:k for k,v in tags_to_id.items()}

    if split == None:
        for el in dataset:
            for label in el["labels"]:
                tags_freq[id_to_tags[label]] += 1
        return {k: v for k, v in sorted(tags_freq.items(), key=lambda item: item[0])}
    
    else:
        for el in dataset[split]:
            for label in el["labels"]:
                tags_freq[id_to_tags[label]] += 1
        return {k: v for k, v in sorted(tags_freq.items(), key=lambda item: item[1], reverse=True)}

def print_label_distribution(dataset, tags_to_id, exclude_fair):
    train_freqs = get_tags_frequencies(dataset, tags_to_id, "train")
    val_freqs = get_tags_frequencies(dataset, tags_to_id, "validation")
    test_freqs = get_tags_frequencies(dataset, tags_to_id, "test")

    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(15, 5)

    print(f"Train: {train_freqs}")
    print(f"Validation: {val_freqs}")
    print(f"Test: {test_freqs}")

    # Filter "fair" into bar plots
    if exclude_fair:
        train_freqs = {k:v for k,v in train_freqs.items() if k!="fair"}
        val_freqs = {k:v for k,v in val_freqs.items() if k!="fair"}
        test_freqs = {k:v for k,v in test_freqs.items() if k!="fair"}

    axs[0].bar(*zip(*train_freqs.items()))
    axs[0].set_title("Train")

    axs[1].bar(*zip(*val_freqs.items()))
    axs[1].set_title("Validation")

    axs[2].bar(*zip(*test_freqs.items()))
    axs[2].set_title("Test")

    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.show()  


def print_label_ratio(dataset, tags_to_id):
    train_freqs = get_tags_frequencies(dataset, tags_to_id, "train")
    val_freqs = get_tags_frequencies(dataset, tags_to_id, "validation")
    test_freqs = get_tags_frequencies(dataset, tags_to_id, "test")

    print("TRAIN")
    for k,v in train_freqs.items():
        print(f"{k}: {v} ({v*100/len(dataset["train"]):.2f})")
    
    print("VALIDATION")
    for k,v in val_freqs.items():
        print(f"{k}: {v} ({v*100/len(dataset["validation"]):.2f})")
    
    print("TEST")
    for k,v in test_freqs.items():
        print(f"{k}: {v} ({v*100/len(dataset["test"]):.2f})")

def unfair_pie_chart(dataset, tags_to_id):
    data_freqs = get_tags_frequencies(dataset, tags_to_id)
    dataset_labels = [el for el in data_freqs.keys() if el !="fair"]

    plt.figure(figsize=(13, 9))
    data_val = [val for k, val in data_freqs.items() if k !="fair"]
    plt.pie(data_val, labels=dataset_labels, autopct=lambda x: '{:.1f}%\n({:.0f})'.format(x, sum(data_val)*x/100))

def compare_lexglue(dataset, lex_dataset, tags_to_id, lex_mapping, split):
    data_freqs = get_tags_frequencies(dataset, tags_to_id, split)

    lex_tag_to_id = {v:k for k,v in lex_mapping.items()}
    lex_freqs = get_tags_frequencies(lex_dataset, lex_tag_to_id, split)

    dataset_labels = [el for el in data_freqs.keys() if el !="fair"]
    lex_labels = [el for el in lex_freqs.keys() if el !="fair" and el !="pinc"]

    fig, ax = plt.subplots(1,2, figsize=(12, 8))
    data_val = [val for k, val in data_freqs.items() if k !="fair"]
    ax[0].pie(data_val, labels=dataset_labels, autopct=lambda x: '{:.1f}%\n({:.0f})'.format(x, sum(data_val)*x/100))
    ax[0].set_title(f"Dataset {split}")

    lex_val = [val for k, val in lex_freqs.items() if k !="fair" and  k !="pinc"]
    ax[1].pie(lex_val, labels=lex_labels, autopct=lambda x: '{:.1f}%\n({:.0f})'.format(x, sum(lex_val)*x/100))
    ax[1].set_title(f"Lex {split}")

def compare_dataset_splits(dataset, tags_to_id):
    train_freqs = get_tags_frequencies(dataset, tags_to_id, "train")
    val_freqs = get_tags_frequencies(dataset, tags_to_id, "validation")
    test_freqs = get_tags_frequencies(dataset, tags_to_id, "test")


    # Filter "fair" into bar plots
    train_freqs = {k:v for k,v in train_freqs.items() if k!="fair"}
    val_freqs = {k:v for k,v in val_freqs.items() if k!="fair"}
    test_freqs = {k:v for k,v in test_freqs.items() if k!="fair"}
    
    labels = list(train_freqs.keys())

    train_values = list(train_freqs.values())
    val_values = list(val_freqs.values())
    test_values = list(test_freqs.values())

    x = np.arange(len(labels))
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, train_values, width, label='Train Split')
    ax.bar(x, val_values, width, label='Validation Split')
    ax.bar(x + width, test_values, width, label='Test Split')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution Across Dataset Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()