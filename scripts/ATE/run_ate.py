import os
import argparse
import logging

from utils.data_utils import *
from utils.eval_utils import *

from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification
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
    parser.add_argument('--sentence_level',
                        action='store_true')
    
    parser.add_argument('--do_lower_case',
                        action='store_true')
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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))
    dataset_dict = None
    dataset_dict = process(args.data_dir, tokenizer, 'sentence' if args.sentence_level else 'document')

    total_steps_epoch = len(dataset_dict['train']) // (args.batch_size * args.gradient_accumulation_steps)
    logging_steps = total_steps_epoch
    eval_steps = logging_steps
    save_steps = logging_steps
    load_best_model_at_end = True
    folder_model = 'e' + str(args.epochs) + '_lr' + str(args.learning_rate)
    output_dir = model_dir + '/results'
    logging_dir = model_dir + '/results'
    # get best model through a metric
    metric_for_best_model = 'eval_f1'
    if metric_for_best_model == 'eval_f1':
        greater_is_better = True
    elif metric_for_best_model == 'eval_loss':
        greater_is_better = False

    args_ = TrainingArguments(
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

    data_collator = DataCollatorForTokenClassification(tokenizer)

    early_stopping_patience = args.save_total_limit

    trainer = Trainer(
        model,
        args_,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()

    dev_preds, dev_labels, ner_results = predict(trainer, dataset_dict['validation'])
    ner_results.to_csv(os.path.join(model_dir, 'ner_results.csv'), index=False)

    dev_preds_df = pd.DataFrame({'id': list(range(len(dev_preds))), 'tokens': dataset_dict['validation']['tokens'], 'ner_prediction': dev_preds, 'ner_labels': dev_labels})
    dev_preds_df['predicted_aspects'] = dev_preds_df.apply(lambda row: extract_aspect(row['tokens'], row['ner_prediction']), axis=1)
    dev_preds_df['true_aspects'] = dev_preds_df.apply(lambda row: extract_aspect(row['tokens'], row['ner_labels']), axis=1)
    dev_preds_df.to_csv(os.path.join(model_dir, 'ATE_dev_preds.csv'), index=False)

    test_preds = predict(trainer, dataset_dict['test'], inference=True)
    test_preds_df = pd.DataFrame({'id': dataset_dict['test']['id'], 'tokens': dataset_dict['test']['tokens'], 'ner_prediction': test_preds})
    test_preds_df['aspects'] = test_preds_df.apply(lambda row: extract_aspect(row['tokens'], row['ner_prediction']), axis=1)
    if args.sentence_level:
        test_preds_df = test_preds_df[['id', 'aspects']].groupby('id').agg({
            'aspects': lambda x: concat_lists(x)
        }).reset_index()
    test_preds_df[['id', 'aspects']].to_csv(os.path.join(model_dir, f'ATE_test_preds.csv'), index=False, sep=';')
    
    results = eval_ate(dev_preds_df.predicted_aspects, dev_preds_df.true_aspects)
    logger.info("***** Dev results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s".format(key, str(results[key])))


    # dev_preds_df.rename(columns={'true_aspects': 'aspects'})[['id', 'aspects']].to_csv('ATE_dev_gt.csv', index=False, sep=';')
    # dev_preds_df.rename(columns={'predicted_aspects': 'aspects'})[['id', 'aspects']].to_csv('ATE_dev_preds.csv', index=False, sep=';')