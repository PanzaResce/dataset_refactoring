DATASET='unfair_tos'
TASK_TYPE='multi_label'

python models/tfidf_svm.py --dataset ${DATASET} --task_type ${TASK_TYPE}