import os
import argparse
import logging

from utils.data_utils import *
from utils.eval_utils import *

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
    )
from transformers.trainer_callback import EarlyStoppingCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=True)
    parser.add_argument('--data_dir',
                        type=str,
                        required=True)
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=3e-5)
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.1)
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1)
    parser.add_argument('--warmup_steps',
                        type=int,
                        default=1000)
    parser.add_argument('--save_total_limit',
                        type=int,
                        default=3)
    parser.add_argument('--evaluation_strategy',
                        type=str,
                        default='steps')
    parser.add_argument('--logging_strategy',
                        type=str,
                        default='steps')
    parser.add_argument('--save_strategy',
                        type=str,
                        default='steps')
    
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)

    experiment_name = args.model_name.split('/')[-1]
    model_dir = f'./experiments/{experiment_name}'

    if os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label2id = {'positive': 2, 'neutral': 1, 'negative': 0}
    id2label = {2: 'positive', 1: 'neutral', 0: 'negative'}
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    dataset_dict = process(args.data_dir, tokenizer)

    total_steps_epoch = len(dataset_dict['train']) // (args.batch_size * args.gradient_accumulation_steps)
    logging_steps = total_steps_epoch
    eval_steps = logging_steps
    save_steps = logging_steps
    load_best_model_at_end = True
    folder_model = 'e' + str(args.epochs) + '_lr' + str(args.learning_rate)
    output_dir = model_dir + 'results'
    logging_dir = model_dir + 'results'
    # get best model through a metric
    metric_for_best_model = 'eval_BalancedAccuracy'
    if metric_for_best_model == 'eval_BalancedAccuracy':
        greater_is_better = True
    elif metric_for_best_model == 'eval_loss':
        greater_is_better = False

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size*2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        #warmup_ratio=warmup_ratio,
        save_total_limit=args.save_total_limit,
        logging_steps=logging_steps,
        eval_steps=logging_steps,
        load_best_model_at_end = load_best_model_at_end,
        metric_for_best_model = metric_for_best_model,
        greater_is_better = greater_is_better,
        gradient_checkpointing=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy=args.evaluation_strategy,
        logging_dir=logging_dir,
        logging_strategy=args.logging_strategy,
        save_strategy=args.save_strategy,
        save_steps=save_steps,
        fp16=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    early_stopping_patience = args.save_total_limit

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()

    preds, _, _ = trainer.predict(dataset_dict['test'])
    preds = np.argmax(preds, axis=-1)
    preds -= 1
    pd.DataFrame({'id': [_id.item() for _id in dataset_dict['test']['id']], 'sentiment': preds.tolist()}).to_csv('submission.csv', index=False, header=False, sep=';')

    results = trainer.evaluate()
    print("***** Dev results *****")
    for key in sorted(results.keys()):
        print("  %s = %s", key, str(results[key]))

    print(trainer.evaluate())
    