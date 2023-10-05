from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed
from torch.utils.data import DataLoader

class ClimateDataset:

    def __init__(self, model_to_train, model_checkpoint, batch_size, dataset_url = "https://raw.githubusercontent.com/fzanart/FliccClasf/main/", seed=42):
        set_seed(seed)
        # map_labels = None, # map labels to a different scenario -> pass a list of desired labels,
        # filter_labels = False, # exclude labels from dataset if they are not in labelmap
        # model_1 -> binary classfication, model_2 -> bknow labels, model_3 -> struct labels, model_4 -> all labels (one model)
        MODEL_TO_TRAIN= {1:{'labelmap':{'ad hominem':1, 'anecdote':1, 'cherry picking':1, 'conspiracy theory':1, 'fake experts':1, 'false choice':1, 'false equivalence':1, 'impossible expectations':1, 'misrepresentation':0, 'oversimplification':0, 'single cause':1, 'slothful induction':0},
                            'filter_labels':False,
                            'map_labels':['bknow','struct']},
                         2:{'labelmap':{'misrepresentation':0, 'oversimplification':1, 'slothful induction':2},
                            'filter_labels':['misrepresentation','oversimplification','slothful induction'],
                            'map_labels':None},
                         3:{'labelmap':{'ad hominem':0,'anecdote':1,'cherry picking':2,'conspiracy theory':3,'fake experts':4,'false choice':5,'false equivalence':6,'impossible expectations':7,'single cause':8},
                            'filter_labels':['ad hominem','anecdote','cherry picking','conspiracy theory','fake experts','false choice','false equivalence','impossible expectations','single cause'],
                            'map_labels':None},
                         4:{'labelmap':{'ad hominem':0, 'anecdote':1, 'cherry picking':2, 'conspiracy theory':3, 'fake experts':4, 'false choice':5, 'false equivalence':6, 'impossible expectations':7, 'misrepresentation':8, 'oversimplification':9, 'single cause':10, 'slothful induction':11},
                            'filter_labels':False,
                            'map_labels':None}}
        self.dataset_url = dataset_url
        self.labelmap = MODEL_TO_TRAIN[model_to_train]['labelmap']
        if MODEL_TO_TRAIN[model_to_train]['map_labels']:
            self.labels = MODEL_TO_TRAIN[model_to_train]['map_labels']
        else:
            self.labels = list(self.labelmap.keys())
        self.num_labels = len(self.labels)

        self.model_checkpoint = model_checkpoint
        if any(k in self.model_checkpoint for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, padding_side=padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.load_dataset(MODEL_TO_TRAIN[model_to_train]['filter_labels'])

    def load_dataset(self, filter_labels):
        self.train_dataset = load_dataset('csv', data_files=self.dataset_url+'Data/fallacy_train.csv', split='train')
        self.test_dataset = load_dataset('csv', data_files=self.dataset_url+'Data/fallacy_test.csv', split='train')
        self.val_dataset = load_dataset('csv', data_files=self.dataset_url+'Data/fallacy_val.csv', split='train')
        self.dataset = DatasetDict({'train': self.train_dataset,'test': self.test_dataset,'val': self.val_dataset})
        self.dataset = self.dataset.rename_column("label", "labels")
        if filter_labels:
            # filter dataset and only keep desired_labels
            self.dataset = self.dataset.filter(lambda sample: sample['labels'] in list(self.labelmap.keys()))

    def preprocess_data(self, batch):
        tokenized_batch = self.tokenizer(batch["text"], padding=True, truncation=True)
        tokenized_batch["labels"] = [self.labelmap.get(label) for label in batch["labels"]]
        return tokenized_batch
    
    def setup_dataloaders(self):
        self.dataset_encoded = self.dataset.map(self.preprocess_data, batched=True, batch_size=None)
        try:
            self.dataset_encoded = self.dataset_encoded.select_columns(['labels', 'input_ids', 'token_type_ids', 'attention_mask'])
        except ValueError:
            self.dataset_encoded = self.dataset_encoded.select_columns(['labels', 'input_ids', 'attention_mask'])
        self.dataset_encoded.set_format("torch")
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataloader = DataLoader(self.dataset_encoded["train"], batch_size=self.batch_size, collate_fn=self.data_collator)
        self.eval_dataloader = DataLoader(self.dataset_encoded["val"], batch_size=self.batch_size, collate_fn=self.data_collator)
        self.test_dataloader = DataLoader(self.dataset_encoded["test"], batch_size=self.batch_size, collate_fn=self.data_collator)