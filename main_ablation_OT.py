from model_ablation_OT import Patient_DA_prediction
from data import prepare_dataloader
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--da_embedding_dimension', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--encoder_attention_heads', type=int, default=8)
    parser.add_argument('--decoder_attention_heads', type=int, default=8)
    parser.add_argument('--encoder_layers', type=int, default=4)
    parser.add_argument('--decoder_layers', type=int, default=4)
    parser.add_argument('--encoder_ffn_dim', type=int, default=128)
    parser.add_argument('--decoder_ffn_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--moving_average', type=int, default=5)
    parser.add_argument('--autocorrelation_factor', type=int, default=5)
    args = parser.parse_args()


    hyper_params = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "da_embedding_dimension" : args.da_embedding_dimension,
        "d_model" :args.d_model,
        "encoder_attention_heads" : args.encoder_attention_heads,
        "decoder_attention_heads" : args.decoder_attention_heads,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "encoder_ffn_dim": args.encoder_ffn_dim,
        "decoder_ffn_dim": args.decoder_ffn_dim,
        "seed" : 42,
        "dropout":args.dropout,
        "moving_average": args.moving_average,
        "autocorrelation_factor": args.autocorrelation_factor
    }

    model_config = {
    "prediction_length": 1,
    "context_length": 20,
    "num_time_features": 1,
    "da_embedding_dimension": hyper_params['da_embedding_dimension'],
    "n_da": 24,
    "input_size": hyper_params['da_embedding_dimension'],
    "num_static_categorical_features": 1,
    "num_static_real_features": 54,
    "cardinality": [3],  # number of DA from the patient, number of DA form the therapist
    "embedding_dimension": [3],
    "d_model": hyper_params['d_model'],
    "n_da_client": 8,
    "lags_sequence": [1, 2, 3],
    "encoder_attention_heads": hyper_params['encoder_attention_heads'],
    "decoder_attention_heads": hyper_params['decoder_attention_heads'],
    "encoder_layers": hyper_params['decoder_layers'],
    "decoder_layers": hyper_params['decoder_layers'],
    "encoder_ffn_dim": hyper_params['encoder_ffn_dim'],
    "decoder_ffn_dim": hyper_params['encoder_ffn_dim'],
    "activation_function": 'gelu',
    "dropout": hyper_params['dropout'],
    "encoder_layerdrop": hyper_params['dropout'],
    "decoder_layerdrop": hyper_params['dropout'],
    "attention_dropout": hyper_params['dropout'],
    "activation_dropout": hyper_params['dropout'],
    "num_parallel_samples": 100,
    "init_std": 0.02,
    "use_cache": True,
    "is_encoder_decoder": True,
    "label_length": 10,
    "moving_average": hyper_params['moving_average'],
    "autocorrelation_factor": hyper_params['autocorrelation_factor']
    }
    torch.manual_seed(hyper_params['seed'])

    model = Patient_DA_prediction(model_config)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(batch_size = hyper_params['batch_size'])

    n_epochs = hyper_params['epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyper_params['learning_rate'], steps_per_epoch=len(train_dataloader), epochs=n_epochs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(n_epochs):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        train_loss = 0
        train_accuracy = 0
        n_batches = 0
        train_f1 = 0
        train_f1_dict = {}
        preds = []
        targets = []
        model.train()
        for i, batch in pbar:
            past_values_da = batch['past_values_da'].to(device)
            past_time_features = batch['past_time_features'].to(device)
            past_observed_mask = batch['past_observed_mask'].to(device)
            static_categorical_features = batch['static_categorical_features'].to(device)
            future_values_da = batch['future_values_da'].to(device)
            future_time_features = batch['future_time_features'].to(device)
            text = batch['text'].to(device)


            outputs = model(past_values_da, past_time_features, past_observed_mask, static_categorical_features, future_values_da, future_time_features,text)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            n_batches += 1

            pbar.set_description(f"epoch {epoch} loss {train_loss/n_batches} accuracy {train_accuracy/n_batches} f1 {train_f1/n_batches}")

        # Validation
        val_loss = 0
        val_accuracy = 0
        val_f1 = 0
        preds = []
        targets = []
        n_batches = 0
        val_f1_dict = {}
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        model.eval()
        for i, batch in pbar:
            past_values_da = batch['past_values_da'].to(device)
            past_time_features = batch['past_time_features'].to(device)
            past_observed_mask = batch['past_observed_mask'].to(device)
            static_categorical_features = batch['static_categorical_features'].to(device)
            future_values_da = batch['future_values_da'].to(device)
            future_time_features = batch['future_time_features'].to(device)
            text = batch['text'].to(device)

            outputs = model(past_values_da, past_time_features, past_observed_mask, static_categorical_features, future_values_da,future_time_features,text)
            val_loss += outputs['loss'].item()
            outputs_pred = model.generate(past_values_da, past_time_features, past_observed_mask, static_categorical_features, future_time_features,text)
            preds.append(outputs_pred['pred'].squeeze())
            targets.append(future_values_da[:,0])

            val_f1 = model.f1(outputs_pred['pred'].squeeze(), future_values_da[:,0])
            for i,da in enumerate(model.da_names):
                if da not in val_f1_dict:
                    val_f1_dict[da] = 0
                val_f1_dict[da] += val_f1[i]
            val_f1 = model.f1_macro(outputs_pred['pred'].squeeze(), future_values_da[:,0])
            val_accuracy += torch.mean((outputs_pred['pred']==future_values_da[:,0]).float()).item()
            n_batches += 1
            pbar.set_description(f"epoch {epoch} val_loss {val_loss/n_batches} val_accuracy {val_accuracy/n_batches} val_f1 {val_f1/n_batches}")
        preds = torch.cat(preds)
        targets = torch.cat(targets)

        val_f1 = model.f1(preds, targets)
        val_f1_dict = {}
        for i in range(len(model.da_names)):
            val_f1_dict[model.da_names[i]] = val_f1[i]
        val_f1 = model.f1_macro(preds, targets)



        print(f"epoch {epoch} train_loss {train_loss/len(train_dataloader)} val_loss {val_loss/len(val_dataloader)} train_accuracy {train_accuracy/len(train_dataloader)} val_accuracy {val_accuracy/len(val_dataloader)}")
        log_dict = {
            "train_loss": train_loss/len(train_dataloader),
            "val_loss": val_loss/len(val_dataloader),
            "train_accuracy": train_accuracy/len(train_dataloader),
            "val_accuracy": val_accuracy/len(val_dataloader),
            "train_f1": train_f1,
            "val_f1": val_f1
        }
        for da in model.da_names:

            log_dict[f"val_f1_{da}"] = val_f1_dict[da]

        np.save(f"predictions/preds_ablation_OT_test.npy", preds.cpu().numpy())
        np.save(f"predictions/targets_ablation_OT_test.npy", targets.cpu().numpy())
        torch.save(model,f"models/ablation_OT.pt")
