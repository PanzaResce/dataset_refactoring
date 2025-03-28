from datasets import concatenate_datasets
from utils.dataset_utils import *

def docs_correctly_distributed_among_splits(dataset_dict, n_test_docs):
    # All the clauses of a document must be in the same split
    train_docs = set(el["doc"] for el in dataset_dict["train"])
    val_docs = set(el["doc"] for el in dataset_dict["validation"])
    test_docs = set(el["doc"] for el in dataset_dict["test"])

    tot_docs = n_test_docs

    if len(train_docs) + len(val_docs) + len(test_docs) != tot_docs:
        print(f"FOUND ONLY {len(train_docs) + len(val_docs) + len(test_docs)} documents out of {tot_docs}")
        return

    train_val_intersect = train_docs.intersection(val_docs)
    train_test_intersect = train_docs.intersection(test_docs)
    val_test_intersect = val_docs.intersection(test_docs)
    if len(train_val_intersect) != 0:
        print(f"docs_correctly_distributed_among_splits FAILED (X): some documents ({len(train_val_intersect)}) shared between train and validation")
        return

    if len(train_test_intersect) != 0:
        print(f"docs_correctly_distributed_among_splits FAILED (X): some documents ({len(train_test_intersect)}) shared between train and test")
        return

    if len(val_test_intersect) != 0:
        print(f"docs_correctly_distributed_among_splits FAILED (X): some documents ({len(val_test_intersect)}) shared between test and validation")
        return

    print("docs_correctly_distributed_among_splits OK (V)")

def check_label_integrity(dataset_dict, multi_class):
    # if the task is not multi-class, then there can't be fair and unfair label mixed
    if multi_class:
        print("This test covers only the case where the task is NOT multi-class (?)")
        return
    
    def loop_split(dataset, split):
        for el in dataset[split]:
            if 0 in el["labels"] and len(el["labels"]) != 1:
                print(f"check_label_integrity FAILED (X): found fair label with more than one element in the {split} split ({el})")
                return False
        return True

    if loop_split(dataset_dict, "train") and loop_split(dataset_dict, "validation") and loop_split(dataset_dict, "test"):
        print(f"check_label_integrity OK (V)")

def check_documents_are_142(doc_list_file):
    doc_list = load_document_list(doc_list_file)
    
    if len(doc_list) != 142:
        print(f"check_documents_are_142 FAILED (X): Number of documents mismatching: {len(doc_list)} instead of 142")
        return False
    else:
        print(f"check_documents_are_142 OK (V): Documents are exactly {len(doc_list)}/142")
        return True

def check_tags_to_id_mappings_match(list_tags_file):
    tags_to_id, id_to_tags = get_tags_id(list_tags_file)

    for id in id_to_tags.keys():
        rev_tags_to_id = {v:k for k,v in tags_to_id.items()}
        if id_to_tags[id] != rev_tags_to_id[id]:
            print(f"check_tags_to_id_mappings_match FAILED (X): Found mismatch between tags_to_id and id_to_tags ({id_to_tags[id]} != {rev_tags_to_id[id]})")
            return False
    print(f"check_tags_to_id_mappings_match OK (V): Matching tags_to_id and id_to_tags")
    