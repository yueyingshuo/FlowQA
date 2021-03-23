import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from allennlp.nn.util import remove_sentence_boundaries
from . import layers

class FlowQA(nn.Module):
    """Network for the FlowQA Module."""
    def __init__(self, opt, embedding=None, padding_idx=0):
        super(FlowQA, self).__init__()

        # Input size to RNN: word emb + char emb + question emb + manual features
        doc_input_size = 0
        que_input_size = 0
        question_input_size=0

        layers.set_my_dropout_prob(opt['my_dropout_p'])
        layers.set_seq_dropout(opt['do_seq_dropout'])

        if opt['use_wemb']:
            # Word embeddings
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)
            if embedding is not None:
                self.embedding.weight.data = embedding
                if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                    opt['fix_embeddings'] = True
                    opt['tune_partial'] = 0
                    for p in self.embedding.parameters():
                        p.requires_grad = False
                else:
                    assert opt['tune_partial'] < embedding.size(0)
                    fixed_embedding = embedding[opt['tune_partial']:]
                    # a persistent buffer for the nn.Module
                    self.register_buffer('fixed_embedding', fixed_embedding)
                    self.fixed_embedding = fixed_embedding
            embedding_dim = opt['embedding_dim']
            doc_input_size += embedding_dim
            que_input_size += embedding_dim
            question_input_size+=embedding_dim
        else:
            opt['fix_embeddings'] = True
            opt['tune_partial'] = 0

        if opt['CoVe_opt'] > 0:
            self.CoVe = layers.MTLSTM(opt, embedding)
            CoVe_size = self.CoVe.output_size
            doc_input_size += CoVe_size
            que_input_size += CoVe_size
            question_input_size+=CoVe_size

        if opt['use_elmo']:
            options_file = "/home/yueying/pycharm_workspace/FlowQA/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
            weight_file = "/home/yueying/pycharm_workspace/FlowQA/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            doc_input_size += 1024
            que_input_size += 1024
        if opt['use_pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            doc_input_size += opt['pos_dim']
        if opt['use_ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            doc_input_size += opt['ner_dim']

        if opt['do_prealign']:
            self.pre_align = layers.GetAttentionHiddens(embedding_dim, opt['prealign_hidden'], similarity_attention=True)
            doc_input_size += embedding_dim
        if opt['no_em']:
            doc_input_size += opt['num_features'] - 3
        else:
            doc_input_size += opt['num_features']

        # Setup the vector size for [doc, question]
        # they will be modified in the following code
        doc_hidden_size, que_hidden_size = doc_input_size, que_input_size
        print('Initially, the vector_sizes [doc, query] are', doc_hidden_size, que_hidden_size)

        flow_size = opt['hidden_size']

        # RNN document encoder
        self.doc_rnn1 = layers.StackedBRNN(doc_hidden_size, opt['hidden_size'], num_layers=1)
        self.dialog_flow1 = layers.StackedBRNN(opt['hidden_size'] * 2, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        self.doc_rnn2 = layers.StackedBRNN(opt['hidden_size'] * 2 + flow_size + CoVe_size, opt['hidden_size'], num_layers=1)
        self.dialog_flow2 = layers.StackedBRNN(opt['hidden_size'] * 2, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        doc_hidden_size = opt['hidden_size'] * 2

        # RNN question encoder
        self.question_rnn, que_hidden_size = layers.RNN_from_opt(que_hidden_size, opt['hidden_size'], opt,
        num_layers=2, concat_rnn=opt['concat_rnn'], add_feat=CoVe_size)
        #last question encoder
        self.lq_rnn, lq_hidden_size = layers.RNN_from_opt(question_input_size, opt['hidden_size'], opt,
                                                                    num_layers=2, concat_rnn=opt['concat_rnn'],
                                                                 add_feat=CoVe_size)



        # Output sizes of rnn encoders
        print('After Input LSTM, the vector_sizes [doc, query] are [', doc_hidden_size, que_hidden_size, '] * 2')

        # Deep inter-attention
        self.deep_attn = layers.DeepAttention(opt, abstr_list_cnt=2, deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], do_similarity=opt['deep_inter_att_do_similar'], word_hidden_size=embedding_dim+CoVe_size, no_rnn=True)

        self.deep_attn_rnn, doc_hidden_size = layers.RNN_from_opt(self.deep_attn.att_final_size + flow_size, opt['hidden_size'], opt, num_layers=1)
        self.dialog_flow3 = layers.StackedBRNN(doc_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)

        # Question understanding and compression
        self.high_lvl_qrnn, que_hidden_size = layers.RNN_from_opt(que_hidden_size * 2, opt['hidden_size'], opt, num_layers = 1, concat_rnn = True)
        #last question understadning
        self.lq_high_lvl_qrnn, lq_hidden_size = layers.RNN_from_opt(lq_hidden_size * 2, opt['hidden_size'], opt,
                                                                  num_layers=1, concat_rnn=True)
        # Self attention on context
        att_size = doc_hidden_size + 2 * opt['hidden_size'] * 2

        if opt['self_attention_opt'] > 0:
            self.highlvl_self_att = layers.GetAttentionHiddens(att_size, opt['deep_att_hidden_size_per_abstr'])
            self.high_lvl_crnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size * 2 + flow_size, opt['hidden_size'], opt, num_layers = 1, concat_rnn = False)
            print('Self deep-attention {} rays in {}-dim space'.format(opt['deep_att_hidden_size_per_abstr'], att_size))
        elif opt['self_attention_opt'] == 0:
            self.high_lvl_crnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size + flow_size, opt['hidden_size'], opt, num_layers = 1, concat_rnn = False)

        print('Before answer span finding, hidden size are', doc_hidden_size, que_hidden_size)

        # Question merging
        self.self_attn = layers.LinearSelfAttn(que_hidden_size)
        #last_question
        self.lq_self_attn = layers.LinearSelfAttn(lq_hidden_size)
        if opt['do_hierarchical_query']:
            self.hier_query_rnn = layers.StackedBRNN(que_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
            que_hidden_size = opt['hidden_size']

            self.lq_hier_query_rnn = layers.StackedBRNN(lq_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU,
                                                     bidir=False)
            # lq_hidden_size = opt['hidden_size']

        # Attention for span start/end
        # doc_hidden_size 250  que_hidden_size 125
        self.get_answer = layers.GetSpanStartEnd(doc_hidden_size, que_hidden_size, opt,
        opt['ptr_net_indep_attn'], opt["ptr_net_attn_type"], opt['do_ptr_update'])
        #doc_hidden_size 250 lq_hidden_size 125
        self.lq_get_answer = layers.GetSpanStartEnd(doc_hidden_size, lq_hidden_size, opt,
                                                 opt['ptr_net_indep_attn'], opt["ptr_net_attn_type"],
                                                 opt['do_ptr_update'])



        self.ans_type_prediction = layers.BilinearLayer(doc_hidden_size * 2, que_hidden_size, opt['answer_type_num'])

        self.lq_type_prediction = layers.BilinearLayer(doc_hidden_size * 2, lq_hidden_size, opt['answer_type_num'])

        # Store config
        self.opt = opt

    def forward(self, x1, x1_c, x1_f, x1_pos, x1_ner, x1_mask, x2_full, x2_c, x2_full_mask,x3_full,x3_c,x3_full_mask,qa):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d * len_w] or [1]
        x1_f = document word features indices  [batch * q_num * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x3_full = question word indices        [batch * q_num * len_q]
        x3_c = question char indices           [(batch * q_num) * len_q * len_w]
        x3_full_mask = question padding mask   [batch * q_num * len_q]
        qa=last_qa
        x3_full=history_id
        x3_c=history_cid,
        x3_full_mask=history_mask
        """
        question_input=qa[0]
        answer_input=qa[1]
        question_mask = torch.eq(question_input, 0)
        answer_mask = torch.eq(answer_input, 0)
        lq_mask=x1_mask
        # precomputing ELMo is only for context (to speedup computation)
        if self.opt['use_elmo'] and self.opt['elmo_batch_size'] > self.opt['batch_size']: # precomputing ELMo is used
            if x1_c.dim() != 1: # precomputation is needed
                precomputed_bilm_output = self.elmo._elmo_lstm(x1_c)
                self.precomputed_layer_activations = [t.detach().cpu() for t in precomputed_bilm_output['activations']]
                self.precomputed_mask_with_bos_eos = precomputed_bilm_output['mask'].detach().cpu()
                self.precomputed_cnt = 0

            # get precomputed ELMo
            layer_activations = [t[x1.size(0) * self.precomputed_cnt: x1.size(0) * (self.precomputed_cnt + 1), :, :] for t in self.precomputed_layer_activations]
            mask_with_bos_eos = self.precomputed_mask_with_bos_eos[x1.size(0) * self.precomputed_cnt: x1.size(0) * (self.precomputed_cnt + 1), :]
            if x1.is_cuda:
                layer_activations = [t.cuda() for t in layer_activations]
                mask_with_bos_eos = mask_with_bos_eos.cuda()

            representations = []
            for i in range(len(self.elmo._scalar_mixes)):
                scalar_mix = getattr(self.elmo, 'scalar_mix_{}'.format(i))
                representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                        representation_with_bos_eos, mask_with_bos_eos
                )
                representations.append(self.elmo._dropout(representation_without_bos_eos))

            x1_elmo = representations[0][:, :x1.size(1), :]
            self.precomputed_cnt += 1

            precomputed_elmo = True
        else:
            precomputed_elmo = False

        """
        x1_full = document word indices        [batch * q_num * len_d]
        x1_full_mask = document padding mask   [batch * q_num * len_d]
        """
        x1_full = x1.unsqueeze(1).expand(x3_full.size(0), x3_full.size(1), x1.size(1)).contiguous()
        x1_full_mask = x1_mask.unsqueeze(1).expand(x3_full.size(0), x3_full.size(1), x1.size(1)).contiguous()

        drnn_input_list, qrnn_input_list,question_list = [], [],[]

        x3 = x3_full.view(-1, x3_full.size(-1))
        x3_mask = x3_full_mask.view(-1, x3_full.size(-1))

        if self.opt['use_wemb']:
            # Word embedding for both document and question
            emb = self.embedding if self.training else self.eval_embed
            x1_emb = emb(x1)
            x3_emb = emb(x3)
            question_emb=emb(question_input)
            # Dropout on embeddings
            if self.opt['dropout_emb'] > 0:
                x1_emb = layers.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
                x3_emb = layers.dropout(x3_emb, p=self.opt['dropout_emb'], training=self.training)
                # question_emb = layers.dropout(question_emb, p=self.opt['dropout_emb'], training=self.training)
            #[bs,context_size,300] [question_num,question_len,300]
            drnn_input_list.append(x1_emb)
            qrnn_input_list.append(x3_emb)
            question_list.append(question_emb)

        if self.opt['CoVe_opt'] > 0:
            x1_cove_mid, x1_cove_high = self.CoVe(x1, x1_mask)
            x3_cove_mid, x3_cove_high = self.CoVe(x3, x3_mask)
            question_cove_mid, question_cove_high = self.CoVe(question_input, question_mask)
            # Dropout on contexualized embeddings
            if self.opt['dropout_emb'] > 0:
                x1_cove_mid = layers.dropout(x1_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                x1_cove_high = layers.dropout(x1_cove_high, p=self.opt['dropout_emb'], training=self.training)
                x3_cove_mid = layers.dropout(x3_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                x3_cove_high = layers.dropout(x3_cove_high, p=self.opt['dropout_emb'], training=self.training)
                # question_cove_mid = layers.dropout( question_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                # question_high = layers.dropout( question_cove_high, p=self.opt['dropout_emb'], training=self.training)

            drnn_input_list.append(x1_cove_mid)
            qrnn_input_list.append(x3_cove_mid)
            question_list.append(question_cove_mid)

        if self.opt['use_elmo']:
            if not precomputed_elmo:
                x1_elmo = self.elmo(x1_c)['elmo_representations'][0]#torch.zeros(x1_emb.size(0), x1_emb.size(1), 1024, dtype=x1_emb.dtype, layout=x1_emb.layout, device=x1_emb.device)
            x3_elmo = self.elmo(x3_c)['elmo_representations'][0]#torch.zeros(x3_emb.size(0), x3_emb.size(1), 1024, dtype=x3_emb.dtype, layout=x3_emb.layout, device=x3_emb.device)
            # Dropout on contexualized embeddings
            if self.opt['dropout_emb'] > 0:
                x1_elmo = layers.dropout(x1_elmo, p=self.opt['dropout_emb'], training=self.training)
                x3_elmo = layers.dropout(x3_elmo, p=self.opt['dropout_emb'], training=self.training)

            drnn_input_list.append(x1_elmo)
            qrnn_input_list.append(x3_elmo)

        if self.opt['use_pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            drnn_input_list.append(x1_pos_emb)

        if self.opt['use_ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            drnn_input_list.append(x1_ner_emb)
        #x1_input[batch_size,context_len,1946]  x3_input:[bs*q_num,question_num,1924]
        #question_input[bs,question_len,900]
        x1_input = torch.cat(drnn_input_list, dim=2)
        x3_input = torch.cat(qrnn_input_list, dim=2)
        question_input = torch.cat(question_list, dim=2)

        def expansion_for_doc(z):
            return z.unsqueeze(1).expand(z.size(0), x3_full.size(1), z.size(1), z.size(2)).contiguous().view(-1, z.size(1), z.size(2))
        #expand to[que_num,ctx_len,*] [que_num,ctx_len,*]
        # 但是这个时候还没有和question产生联系，只是形状变了而已
        x1_emb_expand = expansion_for_doc(x1_emb)
        x1_cove_high_expand = expansion_for_doc(x1_cove_high)
        #x1_elmo_expand = expansion_for_doc(x1_elmo)
        if self.opt['no_em']:
            x1_f = x1_f[:, :, :, 3:]
        #[que_num,ctx_len,*] x1_f已经说了context的词是否出现在某一个question里面
        x1_input = torch.cat([expansion_for_doc(x1_input), x1_f.view(-1, x1_f.size(-2), x1_f.size(-1))], dim=2)
        x1_mask = x1_full_mask.view(-1, x1_full_mask.size(-1))

        if self.opt['do_prealign']:
            #[que_num,ctx_len,*]初步计算context和question的相似度了
            x1_atten = self.pre_align(x1_emb_expand, x3_emb, x3_mask)
            x1_input = torch.cat([x1_input, x1_atten], dim=2)

        # === Start processing the dialog ===
        # cur_h: [batch_size * max_qa_pair, context_length, hidden_state]
        # flow : fn (rnn)
        # x1_full: [batch_size, max_qa_pair, context_length]
        def flow_operation(cur_h, flow):
            flow_in = cur_h.transpose(0, 1).view(x1_full.size(2), x1_full.size(0), x1_full.size(1), -1)
            flow_in = flow_in.transpose(0, 2).contiguous().view(x1_full.size(1), x1_full.size(0) * x1_full.size(2), -1).transpose(0, 1)
            # [bsz * context_length, max_qa_pair, hidden_state]
            flow_out = flow(flow_in)
            # [bsz * context_length, max_qa_pair, flow_hidden_state_dim (hidden_state/2)]
            if self.opt['no_dialog_flow']:
                flow_out = flow_out * 0

            flow_out = flow_out.transpose(0, 1).view(x1_full.size(1), x1_full.size(0), x1_full.size(2), -1).transpose(0, 2).contiguous()
            flow_out = flow_out.view(x1_full.size(2), x1_full.size(0) * x1_full.size(1), -1).transpose(0, 1)
            # [bsz * max_qa_pair, context_length, flow_hidden_state_dim]
            return flow_out

        # Encode document with RNN
        doc_abstr_ls = []
        #[batch_size*q_num,context_len,*]
        doc_hiddens = self.doc_rnn1(x1_input, x1_mask)
        #[bs*qa_num,context_len,*]
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow1)

        doc_abstr_ls.append(doc_hiddens)

        #doc_diddens[bs*question_num,context_len,*] doc_abstr_ls:list(batch_size [question_num,context_len])
        doc_hiddens = self.doc_rnn2(torch.cat((doc_hiddens, doc_hiddens_flow, x1_cove_high_expand), dim=2), x1_mask)
        #doc_hiddens = self.doc_rnn2(torch.cat((doc_hiddens,  x1_cove_high_expand), dim=2), x1_mask)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow2)
        doc_abstr_ls.append(doc_hiddens)

        # Encode question with RNN

        #que_abstr_ls=list(batch_size[question_num,question_len,*])

        _, que_abstr_ls = self.question_rnn(x3_input, x3_mask, return_list=True, additional_x=x3_cove_high)
        _, lq_abstr_ls = self.lq_rnn(question_input, question_mask, return_list=True, additional_x=question_cove_high)

        # Final question layer
        #[question_num,question_len,*]
        question_hiddens = self.high_lvl_qrnn(torch.cat(que_abstr_ls, 2), x3_mask)
        que_abstr_ls += [question_hiddens]

        lq_hiddens = self.lq_high_lvl_qrnn(torch.cat(lq_abstr_ls, 2), question_mask)
        lq_abstr_ls += [lq_hiddens]


        # Main Attention Fusion Layer
        doc_info = self.deep_attn([torch.cat([x1_emb_expand, x1_cove_high_expand], 2)], doc_abstr_ls,
        [torch.cat([x3_emb, x3_cove_high], 2)], que_abstr_ls, x1_mask, x3_mask)

        doc_hiddens = self.deep_attn_rnn(torch.cat((doc_info, doc_hiddens_flow), dim=2), x1_mask)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow3)

        doc_abstr_ls += [doc_hiddens]

        # Self Attention Fusion Layer
        x1_att = torch.cat(doc_abstr_ls, 2)

        if self.opt['self_attention_opt'] > 0:
            highlvl_self_attn_hiddens = self.highlvl_self_att(x1_att, x1_att, x1_mask, x3=doc_hiddens, drop_diagonal=True)
            doc_hiddens = self.high_lvl_crnn(torch.cat([doc_hiddens, highlvl_self_attn_hiddens, doc_hiddens_flow], dim=2), x1_mask)
        elif self.opt['self_attention_opt'] == 0:
            doc_hiddens = self.high_lvl_crnn(torch.cat([doc_hiddens, doc_hiddens_flow], dim=2), x1_mask)

        doc_abstr_ls += [doc_hiddens]
        # Merge the question hidden vectors
        #[question_num,*]
        q_merge_weights = self.self_attn(question_hiddens, x3_mask)
        question_avg_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        lq_merge_weights = self.lq_self_attn(lq_hiddens, question_mask)
        lq_avg_hidden = layers.weighted_avg(lq_hiddens, lq_merge_weights)

        if self.opt['do_hierarchical_query']:
            question_avg_hidden = self.hier_query_rnn(question_avg_hidden.view(x1_full.size(0), x1_full.size(1), -1))
            question_avg_hidden = question_avg_hidden.contiguous().view(-1, question_avg_hidden.size(-1))



        # Get Start, End span
        #doc_hiddens [question_num,context_len,250] question_avg_hidden[question_num,125]

        start_scores, end_scores = self.get_answer(doc_hiddens, question_avg_hidden, x1_mask)
        print(start_scores[0])


        bs=x1_full.size(1)
        qa_doc_hiddens = torch.FloatTensor(x1_full.size(0), x1_full.size(2),doc_hiddens.size(2)).fill_(0).cuda(non_blocking=True)
        for batch_i in range(x1_full.size(0)):
            i = bs * batch_i
            tmp = doc_hiddens[i]
            while True:
                tmp = torch.add(tmp, doc_hiddens[i + 1])
                i += 1
                if (i + 1) >= bs * (batch_i + 1):
                    break
            qa_doc_hiddens[batch_i,] = torch.div(tmp,bs)
        #[bs,ctx_len]
        lq_start_scores, lq_end_scores = self.lq_get_answer(qa_doc_hiddens, lq_avg_hidden, lq_mask)
        print(lq_start_scores[0])

        # lq_start=lq_start_scores.view_as(x1_full)
        all_start_scores = start_scores.view_as(x1_full)     # batch x q_num x len_d
        all_end_scores = end_scores.view_as(x1_full)         # batch x q_num x len_d

        # Get whether there is an answer
        # doc hiddens=[bs*q_num,context_len,250] qa_doc_hidden[bs,ctxt_len,250]
        #[33,500]按照max and mean把第二个维度加起来了
        doc_avg_hidden = torch.cat((torch.max(doc_hiddens, dim=1)[0], torch.mean(doc_hiddens, dim=1)), dim=1)
        #[33,500][33,125]=>
        class_scores = self.ans_type_prediction(doc_avg_hidden, question_avg_hidden)
        all_class_scores = class_scores.view(x1_full.size(0), x1_full.size(1), -1)      # batch x q_num x class_num
        all_class_scores = all_class_scores.squeeze(-1) # when class_num = 1 【3,11】
        return lq_start_scores,lq_end_scores,all_start_scores

        # return all_start_scores, all_end_scores , all_class_scores
