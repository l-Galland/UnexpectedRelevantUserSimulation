
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AutoformerConfig, AutoformerForPrediction
# import F1
from torchmetrics import F1Score

class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size

    def forward(self, target_word, context_word, negative_example):
        emb_target = self.embeddings_target(target_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))
        emb_negative = self.embeddings_context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))
        return -out

    def get_closest_word(self, batch_word):
        # min distance between batch_word and all words in the embedding

        batch_word = batch_word.unsqueeze(1)

        batch_word = batch_word.repeat(1, 8, 1)
        all_words = torch.arange(8).unsqueeze(0).repeat(batch_word.size(0), 1).to(batch_word.device)
        all_words = self.embeddings_target(all_words)
        distances = torch.norm(batch_word - all_words, dim=-1)

        return torch.argmin(distances, dim=-1)

class Patient_DA_prediction(nn.Module):
    def __init__(self,config):
        super(Patient_DA_prediction, self).__init__()
        da_embedding_dimension = 8
        self.config = AutoformerConfig(
                prediction_length=config['prediction_length'],
                context_length=config['context_length'],
                num_time_features=config['num_time_features'],
                da_embedding_dimension=da_embedding_dimension,
                n_da=config['n_da'],
                input_size=da_embedding_dimension,
                num_static_categorical_features=config['num_static_categorical_features'],
                num_static_real_features=0,
                cardinality=config['cardinality'],
                embedding_dimension=config['embedding_dimension'],
                d_model=config['d_model'],
                n_da_client=config['n_da_client'],
                lags_sequence=config['lags_sequence'],
                encoder_attention_heads=config['encoder_attention_heads'],
                decoder_attention_heads=config['decoder_attention_heads'],
                encoder_layers=config['encoder_layers'],
                decoder_layers=config['decoder_layers'],
                encoder_ffn_dim=config['encoder_ffn_dim'],
                decoder_ffn_dim=config['decoder_ffn_dim'],
                activation_function=config['activation_function'],
                dropout=config['dropout'],
                encoder_layerdrop=config['encoder_layerdrop'],
                decoder_layerdrop=config['decoder_layerdrop'],
                attention_dropout=config['attention_dropout'],
                activation_dropout=config['activation_dropout'],
                num_parallel_samples=config['num_parallel_samples'],
                init_std=config['init_std'],
            moving_average=config['moving_average'],
            autocorrelation_factor=config['autocorrelation_factor'],
            output_hidden_states=True,
        )
        self.model = AutoformerForPrediction(self.config)
        fusion_dim = 156
        self.da_linear = nn.Linear(da_embedding_dimension, fusion_dim)
        self.da_embedding = Word2Vec(da_embedding_dimension, config['n_da'])
        self.da_embedding.load_state_dict(torch.load('word2vec.pth'))
        self.dropout = nn.Dropout(config['activation_dropout'])
        self.da_prediction_head = nn.Linear(fusion_dim, fusion_dim//2)
        self.da_prediction_head2 = nn.Linear(fusion_dim//2, self.config.n_da_client)
        self.f1 = F1Score(num_classes=self.config.n_da_client, average='none',task='multiclass')
        self.f1_macro = F1Score(num_classes=self.config.n_da_client, average='macro',task='multiclass')
        self.da_names = ["Sharing personal information or Describe past event",
"Changing unhealthy behavior in the future",
"Sustaining unhealthy behavior in the future",
"Sharing negative feeling or emotion",
"Sharing positive feeling or emotion",
"Understanding or New Perspective",
"Greeting or Closing",
"Backchannel"]

        self.loss_function = nn.CrossEntropyLoss()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_linear = nn.Linear(768, fusion_dim)

        #freeze da embedding
        for param in self.da_embedding.parameters():
            param.requires_grad = False


    def forward(self, past_value_da, past_time_features, past_observed_mask, static_categorical_features, future_values_da,future_time_features,text):

        text_feature = self.text_linear(text)
        out = self.da_prediction_head(text_feature)
        out = self.dropout(out)
        out = F.gelu(out)
        out = self.da_prediction_head2(out)

        loss =  self.loss_function(out, future_values_da.long().squeeze())

        return {'loss': loss}

    def generate(self, past_value_da, past_time_features, past_observed_mask, static_categorical_features, future_time_features,text):
        text_feature = self.text_linear(text)

        out = self.da_prediction_head(text_feature)
        out = self.dropout(out)
        out = F.gelu(out)
        out = self.da_prediction_head2(out)
        out = torch.argmax(out, dim=-1)



        return {"pred" : out}
