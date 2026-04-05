import torch
import torch.nn as nn


class ATSSModel(nn.Module):
    def __init__(self, num_frames=8, feat_dim=256):
        super().__init__()
        self.num_frames = num_frames
        self.feat_dim = feat_dim

        # Three separate encoders for different similarity matrices
        self.encoder_image = self._build_vit()
        self.encoder_text = self._build_vit()
        self.encoder_cross = self._build_vit()

        # Cross-attention layers
        self.img_txt_attn = nn.MultiheadAttention(embed_dim=num_frames, num_heads=4)
        self.txt_img_attn = nn.MultiheadAttention(embed_dim=num_frames, num_heads=4)
        self.cross_attn = nn.MultiheadAttention(embed_dim=num_frames, num_heads=4)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_frames * 3),
            nn.Linear(num_frames * 3, 2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(2)
        )

        self._initialize_weights()

    def _build_vit(self):
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_frames, nhead=4, dim_feedforward=4 * self.num_frames)
        return nn.TransformerEncoder(encoder_layer, num_layers=2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.TransformerEncoderLayer):
                for name, param in m.named_parameters():
                    if 'weight' in name and param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, sim_img, sim_txt, sim_cross):
        # Process each similarity matrix through its own encoder
        feat_img = self.encoder_image(sim_img)
        feat_txt = self.encoder_text(sim_txt)
        feat_cross = self.encoder_cross(sim_cross)

        # 1. Visual-Textual Cross-Attention
        img_txt_output, _ = self.img_txt_attn(feat_img.transpose(0, 1),
                                              feat_txt.transpose(0, 1),
                                              feat_txt.transpose(0, 1))

        # 2. Textual-Visual Cross-Attention
        txt_img_output, _ = self.txt_img_attn(feat_txt.transpose(0, 1),
                                              feat_img.transpose(0, 1),
                                              feat_img.transpose(0, 1))

        # 3. Cross-Modal Cross-Attention
        cross_output, _ = self.cross_attn(feat_cross.transpose(0, 1),
                                          torch.cat([feat_img, feat_txt], dim=1).transpose(0, 1),
                                          torch.cat([feat_img, feat_txt], dim=1).transpose(0, 1))

        # Transpose back to the Original Dimension
        img_txt_output = img_txt_output.transpose(0, 1)
        txt_img_output = txt_img_output.transpose(0, 1)
        cross_output = cross_output.transpose(0, 1)

        # Average pooling over time dimension
        feat_img_txt = img_txt_output.mean(dim=1)
        feat_txt_img = txt_img_output.mean(dim=1)
        feat_cross = cross_output.mean(dim=1)

        # Concatenate all features
        fused = torch.cat([feat_img_txt, feat_txt_img, feat_cross], dim=-1)

        return self.classifier(fused)
