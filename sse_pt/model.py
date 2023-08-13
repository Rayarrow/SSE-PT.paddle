import numpy as np
import paddle
import paddle.nn as nn


class SSEPT(paddle.nn.Layer):
    def __init__(self, user_num, item_num, args):
        super(SSEPT, self).__init__()
        # params
        self.item_hidden_units = args.hidden_units
        self.user_hidden_units = args.hidden_units
        self.hidden_units = self.item_hidden_units + self.user_hidden_units

        # layers
        self.item_emb = nn.Embedding(item_num + 1, self.item_hidden_units)  # [pad] is 0
        self.user_emb = nn.Embedding(user_num + 1, self.user_hidden_units)  # [pad] is 0
        self.pos_emb = nn.Embedding(args.maxlen, self.hidden_units)
        self.emb_dropout = paddle.nn.Dropout(p=args.dropout)

        self.subsequent_mask = (paddle.triu(paddle.ones((args.maxlen, args.maxlen))) == 0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_units,
                                                        nhead=args.num_heads,
                                                        dim_feedforward=self.hidden_units,
                                                        dropout=args.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args.num_blocks)


    def add_positional_embed(self, seq_embed):
        """
        Add position embeddings to item embeddings.
        """
        positions = np.tile(np.array(range(seq_embed.shape[1])), [seq_embed.shape[0], 1])
        position_embed = self.pos_emb(paddle.to_tensor(positions, dtype='int64'))
        return self.emb_dropout(seq_embed + position_embed)

    def comb_user_item_embedding(self, user_ids, log_seqs):
        # process user embedding
        user_embed = self.user_emb(user_ids).unsqueeze(axis=1)  # (batch_size, 1, user_hidden_units)
        user_embed = user_embed.tile([1, log_seqs.shape[1], 1])  # (batch_size, max_len, user_hidden_units)

        # process item embedding
        item_embed = self.item_emb(log_seqs)  # (batch_size, max_len, item_hidden_units)
        comb_embed = paddle.concat([item_embed, user_embed], 2)  # (batch_size, max_len, hidden_units)

        comb_embed = self.add_positional_embed(comb_embed)  # (batch_size, max_len, hidden_units)

        return comb_embed, user_embed, item_embed

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        comb_embed, user_embed, item_embed = self.comb_user_item_embedding(user_ids, log_seqs)

        # all input seqs: (batch_size, max_len)
        log_feats = self.encoder(comb_embed, self.subsequent_mask)  # (batch_size, max_len, hidden_units)

        pos_embed = self.item_emb(pos_seqs) # (batch_size, max_len, item_hidden_units)
        pos_comb_embed = paddle.concat([pos_embed, user_embed], 2) # (batch_size, max_len, hidden_units)
        neg_embed = self.item_emb(neg_seqs) # (batch_size, max_len, item_hidden_units)
        neg_comb_embed = paddle.concat([neg_embed, user_embed], 2) # (batch_size, max_len, hidden_units)

        pos_logits = (log_feats * pos_comb_embed).sum(axis=-1)
        neg_logits = (log_feats * neg_comb_embed).sum(axis=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        """

        :param user_ids (tensor, shape: [1]):
        :param log_seqs (tensor, shape: [max_len]):
        :param item_indices: [
        :return:
        """
        comb_embed, _, _ = self.comb_user_item_embedding(user_ids, log_seqs)

        log_feats = self.encoder(comb_embed, self.subsequent_mask)  # (batch_size, max_len, hidden_units)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(paddle.to_tensor(item_indices, dtype='int64'))
        user_embs = self.user_emb(user_ids).squeeze(0).tile([item_embs.shape[0], 1]) # (batch_size, user_hidden_units)

        candidate_item_embs = paddle.concat([item_embs, user_embs], 1) # (candidate_num, hidden_units)

        logits = candidate_item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
