#!pip install simpletransformers
#!pip install pandas
#!pip install -U scikit-learn

import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='macro')

def precision_multiclass(labels, preds):
    return precision_score(labels, preds, average='macro')

def recall_multiclass(labels, preds):
    return recall_score(labels, preds, average='macro')
    
def evaluate_model(model, df):
    features = df['text'].to_list()
    labels = df['labels'].to_list()
    predictions, raw_outputs = model.predict(features)
    report = classification_report(labels, predictions, digits=4)    
    return report, None, None

mod_agenda = pd.read_json("data/mod-agenda/mod-agenda.json", orient="records")
mod_agenda = mod_agenda.loc[:,["text","label"]]
mod_agenda = mod_agenda.rename(columns = {"label":"labels"})

num_classes = 15

# AS
as_train_df = pd.read_json("data/AS/train.json", orient="records")
as_val_df = pd.read_json("data/AS/validation.json", orient="records")
as_test_df = pd.read_json("data/AS/test.json", orient="records")

as_train_df = as_train_df.loc[:,["problem","label"]]
as_train_df = as_train_df.rename(columns = {"problem":"text", "label":"labels"})
as_val_df = as_val_df.loc[:,["problem","label"]]
as_val_df = as_val_df.rename(columns = {"problem":"text", "label":"labels"})
as_test_df = as_test_df.loc[:,["problem","label"]]
as_test_df = as_test_df.rename(columns = {"problem":"text", "label":"labels"})

as_weights_dict = {i: ((len(as_train_df)/num_classes)/as_train_df["labels"].value_counts()[i]) for i in as_train_df["labels"].value_counts().index}
as_weights = [0]*num_classes
for i in as_weights_dict:
    as_weights[i] = as_weights_dict[i]
    
as_args = {
    "train_batch_size": 32,
    "num_train_epochs": 21,
    "learning_rate": 4e-5,
#     "weight": weights,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
    "reprocess_input_data": False,
    'evaluate_during_training': True,
    "eval_batch_size": 32,
    "no_cache": True,
}

# Create a ClassificationModel
as_model = ClassificationModel('bert', 'allenai/scibert_scivocab_uncased', weight=as_weights, num_labels=num_classes, args=as_args, use_cuda=True)

as_model.train_model(as_train_df, eval_df=as_val_df, f1=f1_multiclass, acc=accuracy_score)

as_result, as_model_outputs, as_wrong_predictions = evaluate_model(as_model, as_test_df)
# as_result, as_model_outputs, as_wrong_predictions = as_model.eval_model(as_test_df, f1=f1_multiclass, acc=accuracy_score, prec=precision_multiclass, rec=recall_multiclass)
print(as_result)

# as_mod_result, as_mod_model_outputs, as_mod_wrong_predictions = as_model.eval_model(mod_agenda, f1=f1_multiclass, acc=accuracy_score, prec=precision_multiclass, rec=recall_multiclass)
as_mod_result, as_mod_model_outputs, as_mod_wrong_predictions = evaluate_model(as_model, mod_agenda)
print(as_mod_result)

# KE
ke_train_df = pd.read_json("data/KE/train.json", orient="records")
ke_val_df = pd.read_json("data/KE/validation.json", orient="records")
ke_test_df = pd.read_json("data/KE/test.json", orient="records")

ke_train_df = ke_train_df.loc[:,["problem","label"]]
ke_train_df = ke_train_df.rename(columns = {"problem":"text", "label":"labels"})
ke_val_df = ke_val_df.loc[:,["problem","label"]]
ke_val_df = ke_val_df.rename(columns = {"problem":"text", "label":"labels"})
ke_test_df = ke_test_df.loc[:,["problem","label"]]
ke_test_df = ke_test_df.rename(columns = {"problem":"text", "label":"labels"})

ke_weights_dict = {i: ((len(ke_train_df)/num_classes)/ke_train_df["labels"].value_counts()[i]) for i in ke_train_df["labels"].value_counts().index}
ke_weights = [0]*num_classes
for i in ke_weights_dict:
    ke_weights[i] = ke_weights_dict[i]
    
ke_args = {
    "train_batch_size": 32,
    "num_train_epochs": 17,
    "learning_rate": 1e-4,
#     "weight": weights,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "overwrite_output_dir": True,
    "reprocess_input_data": False,
    'evaluate_during_training': True,
    "eval_batch_size": 32,
    "no_cache": True,
}

# Create a ClassificationModel
ke_model = ClassificationModel('bert', 'allenai/scibert_scivocab_uncased', weight=ke_weights, num_labels=num_classes, args=ke_args, use_cuda=True)

ke_model.train_model(ke_train_df, eval_df=ke_val_df, f1=f1_multiclass, acc=accuracy_score)

# ke_result, ke_model_outputs, ke_wrong_predictions = ke_model.eval_model(ke_test_df, f1=f1_multiclass, acc=accuracy_score, prec=precision_multiclass, rec=recall_multiclass)
ke_result, ke_model_outputs, ke_wrong_predictions = evaluate_model(ke_model, ke_test_df)
print(ke_result)
# ke_mod_result, ke_mod_model_outputs, ke_mod_wrong_predictions = ke_model.eval_model(mod_agenda, f1=f1_multiclass, acc=accuracy_score, prec=precision_multiclass, rec=recall_multiclass)
ke_mod_result, ke_mod_model_outputs, ke_mod_wrong_predictions = evaluate_model(ke_model, mod_agenda)
print(ke_mod_result)

f = open('eval_results.txt', 'w+')
f.write('AS: \n')
f.write(str(as_result))
f.write('\nKE: \n')
f.write(str(ke_result))
f.write('\nAS on mod-AGENDA: \n')
f.write(str(as_mod_result))
f.write('\nKE on mod-AGENDA: \n')
f.write(str(ke_mod_result))
f.close()

