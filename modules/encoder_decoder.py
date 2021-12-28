import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LSTMAttention(nn.Module):
    def __init__(self, encoder_dim):
        super(LSTMAttention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt)
        # return self.decode(src, tgt)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
        # return src

    def decode(self, img_features, captions):
        return self.decoder(img_features, captions)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, x, mask):
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # return self.sublayer[1](x, self.feed_forward)
        residual = x
        x = self.self_attn(x, x, x, mask)
        x += residual
        x = self.norm1(x)
        residual = x
        x = self.feed_forward(x)
        x += residual
        x = self.norm2(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 encoder_dim,
                 position_encoding,
                 max_seq_length,
                 max_gen_length,
                 hidden_size,
                 tokenizer,
                 sqrt=True
                 ):
        super(Decoder, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(hidden_size, vocabulary_size)
        self.dropout = nn.Dropout()
        self.max_seq_length = max_seq_length
        self.max_gen_length = max_gen_length

        self.tokenizer = tokenizer

        self.attention = LSTMAttention(encoder_dim)

        if not sqrt:
            self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        else:
            self.embedding = Embeddings(hidden_size, vocabulary_size)
        self.lstm = nn.LSTMCell(hidden_size + encoder_dim, hidden_size)
        self.pos_encoding = position_encoding
    
    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c

    def forward(self, img_features, captions=None):
        batch_size = img_features.size(0)
        if self.training:
            max_timespan = max([len(caption) for caption in captions]) - 1
        else:
            max_timespan = max([len(caption) for caption in captions])

        h, c = self.get_init_lstm_state(img_features)

        embedding = self.embedding(captions)

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).cuda()
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).cuda()
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

        return preds

    def caption(self, img_features, beam_size):

        bs = img_features.shape[0]
        ret_seqs = torch.zeros((bs, beam_size, self.max_gen_length)).to(img_features.device)
        ret_logprobs = torch.zeros((bs, beam_size)).to(img_features.device)

        for i in range(bs):

            current_beam_size = beam_size

            prev_words = torch.zeros(beam_size, 1).long().to(img_features.device)
            img_feature = img_features[i].repeat([beam_size, 1, 1])

            sentences = prev_words
            top_preds = torch.zeros(beam_size, 1).to(img_features.device)
            alphas = torch.ones(beam_size, 1, img_feature.size(1)).to(img_features.device)

            completed_sentences = []
            completed_sentences_alphas = []
            completed_sentences_preds = []

            step = 1
            h, c = self.get_init_lstm_state(img_feature)

            while True:
                embedding = self.embedding(prev_words).squeeze(1)
                context, alpha = self.attention(img_feature, h)
                gate = self.sigmoid(self.f_beta(h))
                gated_context = gate * context

                lstm_input = torch.cat((embedding, gated_context), dim=1)
                h, c = self.lstm(lstm_input, (h, c))
                output = F.log_softmax(self.deep_output(self.dropout(h)), -1)
                output = top_preds.expand_as(output) + output

                if step == 1:
                    top_preds, top_words = output[0].topk(current_beam_size, 0, True, True)
                else:
                    top_preds, top_words = output.view(-1).topk(current_beam_size, 0, True, True)
                prev_word_idxs = top_words // output.size(1)
                next_word_idxs = top_words % output.size(1)

                sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
                alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

                incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 0]
                if step == self.max_gen_length:
                    incomplete = []
                complete = list(set(range(len(next_word_idxs))) - set(incomplete))

                if len(complete) > 0:
                    completed_sentences.extend(sentences[complete].tolist())
                    completed_sentences_alphas.extend(alphas[complete].tolist())
                    completed_sentences_preds.extend(top_preds[complete])
                current_beam_size -= len(complete)

                if current_beam_size == 0:
                    break
                sentences = sentences[incomplete]
                alphas = alphas[incomplete]
                h = h[prev_word_idxs[incomplete]]
                c = c[prev_word_idxs[incomplete]]
                img_feature = img_feature[prev_word_idxs[incomplete]]
                top_preds = top_preds[incomplete].unsqueeze(1)
                prev_words = next_word_idxs[incomplete].unsqueeze(1)

                if step >= self.max_gen_length:
                    break
                step += 1

            for j in range(len(completed_sentences)):
                ret_seqs[i][j][:completed_sentences[j].shape[-1]] = completed_sentences[j]
                ret_logprobs[i][j] = completed_sentences_preds[j]
            
        return ret_seqs, ret_logprobs



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class EncoderDecoder(nn.Module):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(tgt_vocab, self.d_model, c(position), self.max_seq_length, self.max_gen_length, self.lstm_dim, self.tokenizer, self.sqrt),
            lambda x: x,
            # nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position))
            lambda x: x
            )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.max_seq_length = args.max_seq_length
        self.max_gen_length = args.max_gen_length
        self.drop_prob_lm = args.drop_prob_lm
        self.sqrt = args.sqrt
        self.lstm_dim = args.lstm_dim
        self.vocab_size = len(tokenizer.idx2token)
        self.tokenizer = tokenizer
        self.beam_size = args.beam_size
        self.use_bn = args.use_bn
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(args.d_vf),) if self.use_bn else ()) +
                (nn.Linear(args.d_vf, args.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(args.d_model),) if self.use_bn == 2 else ())))

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        # self.logit = nn.Linear(args.d_model, tgt_vocab)

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        # att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = self.att_embed(att_feats)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = F.log_softmax(out, dim=-1)
        return outputs

    def _sample(self, att_feats, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        img_features = self.model.encode(att_feats, att_masks)
        sentences, log_probs = self.model.decoder.caption(img_features, self.beam_size)
        return sentences, log_probs

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

