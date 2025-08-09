import torch
import torch.nn as nn
import torch.nn.functional as func


class TenderEncoder(nn.Module):
    def __init__(self, bert_dim=768, output_dim=128, hl=0, dr=False):
        super().__init__()
        self.hl = hl
        self.hidden_layers = nn.ModuleList()

        self.dr = dr
        self.dropout_layers = nn.ModuleList()

        # lat e lon della provincia
        self.geo_layer = nn.Linear(2, 16)

        # categoria oggetto gara
        self.cat_layer = nn.Embedding(3, 16)

        # importo
        self.budget_layer = nn.Linear(1, 16)

        # descrizione cpv (embedding BERT)
        self.cpv_desc_layer = nn.Linear(bert_dim, 64)

        # descrizione oggetto (embedding BERT)
        self.ogg_desc_layer = nn.Linear(bert_dim, 64)

        # hidden layers
        if self.hl > 0:
            for _ in range(self.hl):
                self.hidden_layers.append(nn.Linear(16 + 16 + 16 + 64 + 64, 16 + 16 + 16 + 64 + 64))
                if dr:
                    self.dropout_layers.append(nn.Dropout(0.3))

        self.output_layer = nn.Linear(16 + 16 + 16 + 64 + 64, output_dim)

    def forward(self, tender):
        # lat e lon della provincia
        geo = torch.stack([tender["lat"], tender["lon"]], dim=1)
        geo_emb = func.relu(self.geo_layer(geo))

        # categoria oggetto gara
        cat_emb = func.relu(self.cat_layer(tender["cat"]))

        # importo
        budget_emb = func.relu(self.budget_layer(tender["budget"])).unsqueeze(0)

        # descrizione cpv (embedding BERT)
        cpv_emb = func.relu(self.cpv_desc_layer(tender["cpv"])).unsqueeze(0)

        # descrizione oggetto (embedding BERT)
        ogg_emb = func.relu(self.ogg_desc_layer(tender["ogg"])).unsqueeze(0)

        features = torch.cat([geo_emb, cat_emb, budget_emb, cpv_emb, ogg_emb], dim=1)

        # hidden layers
        if self.hl > 0:
            for i, layer in enumerate(self.hidden_layers):
                features = func.relu(layer(features))
                if self.dr:
                    features = self.dropout_layers[i](features)

        output = self.output_layer(features)

        return output
