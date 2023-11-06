"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Decomposing transformer neural network corresponding to the first
             configuration of events (1 muon and 1-5 protons) described in the paper.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from .embedding import EmbeddingSource, PositionalEncoding


# TransformerConf1 network
class TransformerConf1(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 num_head: int,
                 img_size: int,
                 tgt_size: int,
                 dropout: float = 0.1,
                 max_len: int = 10,
                 device: object = None
                 ):
        super(TransformerConf1, self).__init__()
        # Initialise the Transformer model
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=num_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=emb_size * 4,
                                       dropout=dropout)
        # Linear layer for target embedding
        self.tgt_emb = nn.Linear(tgt_size, emb_size)
        # Embedding source module
        self.src_emb = EmbeddingSource(emb_size=emb_size, img_size=img_size, dropout=dropout, device=device)
        # Positional encoding for target
        self.pos_encoding_tgt = PositionalEncoding(emb_size=emb_size, dropout=dropout, max_len=max_len)
        # Linear layer to map encoder memory to vertex position
        self.memory2vertex = nn.Linear(emb_size, 3)
        # Linear layer for output parameters
        self.output_params = nn.Linear(emb_size, tgt_size)
        # Linear layer to determine whether to keep iterating
        self.keep_iterating = nn.Linear(emb_size, 2)
        # First token as a learnable parameter
        self.first_token = nn.Parameter(torch.randn(1, tgt_size))

    def _init_weights(self, module):
        """
        Initialise the model weights
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialise weights using a normal distribution
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialise layer normalization parameters
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, src: Tensor, exit_muon: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        """
        Encode the source sequence.

        Args:
            src (Tensor): Input source sequence.
            exit_muon (Tensor): Exit muon information.
            src_mask (Tensor): Source mask.
            src_padding_mask (Tensor): Source padding mask.

        Returns:
            memory (Tensor): Encoded memory.
            vertex_pos (Tensor): Vertex position.
        """
        # Extract source information
        src_indexes = src[:, :, :3].long()
        src_charges = src[:, :, 3:4]

        # Embed the source sequence
        src_emb = self.src_emb(src_indexes, src_charges, exit_muon)

        # Encode the source sequence
        memory = self.transformer.encoder(src=src_emb, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)

        return memory, self.memory2vertex(memory[0])

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor,
               tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        """
        Decode the target sequence.

        Args:
            tgt (Tensor): Target sequence.
            memory (Tensor): Encoded memory.
            tgt_mask (Tensor): Target mask.
            tgt_padding_mask (Tensor): Target padding mask.
            memory_key_padding_mask (Tensor): Memory padding mask.

        Returns:
            outs_params (Tensor): Output parameters.
            keep_iterating (Tensor): Whether to keep iterating.
        """
        # Embed the target sequence
        tgt_emb = self.pos_encoding_tgt(self.tgt_emb(tgt))

        # Decode the target sequence
        outs = self.transformer.decoder(tgt=tgt_emb,
                                        memory=memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

        return self.output_params(outs), self.keep_iterating(outs)

    def forward(self,
                src: Tensor,
                exit_muon: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor):
        """
        Forward pass of the Transformer model.

        Args:
            src (Tensor): Input source sequence.
            exit_muon (Tensor): Exit muon information.
            tgt (Tensor): Target sequence.
            src_mask (Tensor): Source mask.
            tgt_mask (Tensor): Target mask.
            src_padding_mask (Tensor): Source padding mask.
            tgt_padding_mask (Tensor): Target padding mask.

        Returns:
            vertex_pos (Tensor): Vertex position.
            outs_params (Tensor): Output parameters.
            keep_iterating (Tensor): Whether to keep iterating.
        """
        # Encode the source sequence
        memory, vertex_pos = self.encode(src, exit_muon, src_mask, src_padding_mask)

        # Concatenate the first token with the target sequence
        tgt = torch.cat((self.first_token.repeat(1, tgt.size(1), 1), tgt), dim=0)

        # Decode the target sequence
        outs_params, keep_iterating = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)

        return vertex_pos, outs_params, keep_iterating
