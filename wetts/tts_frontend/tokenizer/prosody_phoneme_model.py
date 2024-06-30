import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from wetts.tts_frontend.tokenizer.torchcrf import CRF
from wetts.tts_frontend.tokenizer.constants import polyphone_counter
from wetts.tts_frontend.utils import get_mask_from_lengths


class ProsodyPhonemeModel(nn.Module):
    def __init__(self, tagset_size, hidden_dim, pretrained_model_name_or_path):
        super().__init__()
        self.counter = polyphone_counter
        self.polyphone_size = len(self.counter)
        self.max_tagset_size = max(len(self.counter[char]) for char in self.counter)

        self.auto_model = AutoModel.from_pretrained(pretrained_model_name_or_path)

        self.embedding_dim = self.auto_model.config.hidden_size

        # prosody_model
        self.prosody_model = nn.ModuleDict()
        self.prosody_model["pm_fc"] = nn.Linear(self.embedding_dim, hidden_dim)
        self.prosody_model["pm_layer_norm"] = nn.LayerNorm(hidden_dim)
        self.prosody_model["lstm"] = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.prosody_model["lstm_layer_norm"] = nn.LayerNorm(hidden_dim)
        for i in [1, 2, 3]:
            self.prosody_model[f"hidden2p{i}"] = nn.Linear(hidden_dim * 2, tagset_size)
            self.prosody_model[f"p{i}_crf"] = CRF(num_tags=tagset_size)

        # phoneme_model
        self.phoneme_model = nn.ModuleDict()
        self.phoneme_model["pm_fc2"] = nn.Linear(self.embedding_dim, hidden_dim)
        self.phoneme_model["pm_layer_norm2"] = nn.LayerNorm(hidden_dim)
        self.phoneme_model["lstm2"] = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.phoneme_model["lstm_layer_norm2"] = nn.LayerNorm(hidden_dim)
        self.phoneme_model["polyphone_embedding"] = nn.Embedding(
            self.polyphone_size, hidden_dim
        )
        self.phoneme_model["embed_fc"] = nn.Linear(hidden_dim * 3, hidden_dim)
        self.phoneme_model["embed_batch_norm"] = nn.BatchNorm1d(hidden_dim)
        self.phoneme_model["last_fc"] = nn.Linear(hidden_dim, self.max_tagset_size)
        self.phoneme_model["log_softmax"] = nn.LogSoftmax(dim=-1)

    def seq_embed(self, input_ids, token_type_ids=None, attention_mask=None):
        embeds = self.auto_model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        return embeds

    def prosody_embed(self, embeds):
        embeds_fc = self.prosody_model["pm_fc"](embeds)
        lstm_in = F.relu(self.prosody_model["pm_layer_norm"](embeds_fc))
        lstm_out, _ = self.prosody_model["lstm"](lstm_in)
        lstm_out = F.relu(self.prosody_model["lstm_layer_norm"](lstm_out))
        out = torch.cat((lstm_in, lstm_out), dim=2)
        return out

    def phoneme_embed(self, embeds):
        embeds_fc = self.phoneme_model["pm_fc2"](embeds)
        lstm_in = F.relu(self.phoneme_model["pm_layer_norm2"](embeds_fc))
        lstm_out, _ = self.phoneme_model["lstm2"](lstm_in)
        lstm_out = F.relu(self.phoneme_model["lstm_layer_norm2"](lstm_out))
        out = torch.cat((lstm_in, lstm_out), dim=2)
        return out

    def _get_emission_scores(self, embeds):
        seq_embed = self.prosody_embed(embeds)
        p1_emissions = self.prosody_model["hidden2p1"](seq_embed)
        p2_emissions = self.prosody_model["hidden2p2"](seq_embed)
        p3_emissions = self.prosody_model["hidden2p3"](seq_embed)
        return p1_emissions, p2_emissions, p3_emissions

    def prosody_forward(
            self,
            input_ids,
            p1_tags,
            p2_tags,
            p3_tags,
            token_type_ids=None,
            attention_mask=None,
    ):
        seq_embed = self.seq_embed(input_ids, token_type_ids, attention_mask)
        p1_emissions, p2_emissions, p3_emissions = self._get_emission_scores(seq_embed)
        p1_loss = -1 * self.prosody_model["p1_crf"](
            p1_emissions, p1_tags, mask=attention_mask.byte()
        )
        p2_loss = -1 * self.prosody_model["p2_crf"](
            p2_emissions, p2_tags, mask=attention_mask.byte()
        )
        p3_loss = -1 * self.prosody_model["p3_crf"](
            p3_emissions, p3_tags, mask=attention_mask.byte()
        )
        return p1_loss, p2_loss, p3_loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None):
        seq_embed = self.seq_embed(input_ids, token_type_ids, attention_mask)
        return self.seq_decode(seq_embed, attention_mask=attention_mask)

    def seq_decode(self, seq_embed, attention_mask=None):
        p1_emissions, p2_emissions, p3_emissions = self._get_emission_scores(seq_embed)
        p1_tags = self.prosody_model["p1_crf"].decode(
            p1_emissions, mask=attention_mask.byte()
        )
        p2_tags = self.prosody_model["p2_crf"].decode(
            p2_emissions, mask=attention_mask.byte()
        )
        p3_tags = self.prosody_model["p3_crf"].decode(
            p3_emissions, mask=attention_mask.byte()
        )
        return p1_tags, p2_tags, p3_tags

    def phoneme_forward(
            self,
            input_ids,
            position_ids,
            polyphone_ids,
            phoneme_sizes,
            token_type_ids=None,
            attention_mask=None,
    ):
        seq_embed = self.seq_embed(input_ids, token_type_ids, attention_mask)
        out = self.log_softmax_out(
            seq_embed, position_ids, polyphone_ids, phoneme_sizes
        )
        return out

    def log_softmax_out(self, seq_embed, position_ids, polyphone_ids, phoneme_sizes):
        seq_embed = self.phoneme_embed(seq_embed)
        char_embed = torch.gather(
            seq_embed,
            1,
            position_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, seq_embed.size(2)),
        )
        char_embed = char_embed.squeeze(1)
        polyphone_embed = self.phoneme_model["polyphone_embedding"](polyphone_ids)
        embed = torch.cat((char_embed, polyphone_embed), dim=1)
        embed = F.relu(
            self.phoneme_model["embed_batch_norm"](
                self.phoneme_model["embed_fc"](embed)
            )
        )
        fc_out = self.phoneme_model["last_fc"](embed)
        mask = get_mask_from_lengths(phoneme_sizes, self.max_tagset_size)
        fc_out = fc_out.masked_fill(~mask, float("-inf"))
        out = self.phoneme_model["log_softmax"](fc_out)
        return out


def test():
    pp_model = ProsodyPhonemeModel(
        tagset_size=5,
        hidden_dim=128,
        pretrained_model_name_or_path="/data1/xiepengyuan/workspace/audio/wetts/resources/tts_frontend/chinese-electra-180g-small-discriminator"
    )


if __name__ == '__main__':
    test()
