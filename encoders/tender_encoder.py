import torch
import torch.nn as nn
import torch.nn.functional as func


class TenderEncoder(nn.Module):
    def __init__(self, bert_dim=768, output_dim=128):
        super().__init__()
        # lat e lon della provincia
        self.geo_layer = nn.Linear(2, 16)

        # categoria oggetto gara
        self.cat_layer = nn.Linear(1, 16)

        # importo
        self.budget_layer = nn.Linear(1, 16)

        # descrizione cpv (embedding BERT)
        self.cpv_desc_layer = nn.Linear(bert_dim, 64)

        # descrizione oggetto (embedding BERT)
        self.ogg_desc_layer = nn.Linear(bert_dim, 64)

        self.output_layer = nn.Linear(16 + 16 + 16 + 64 + 64, output_dim)

    def forward(self, tender):
        # lat e lon della provincia
        geo = torch.stack([tender["lat"], tender["lon"]], dim=1)
        geo_emb = func.relu(self.geo_layer(geo))

        # categoria oggetto gara
        cat_emb = func.relu(self.cat_layer(tender["cat"]))

        # importo
        budget_emb = func.relu(self.budget_layer(tender["budget"]))

        # descrizione cpv (embedding BERT)
        cpv_emb = func.relu(self.cpv_desc_layer(tender["cpv_desc"]))

        # descrizione oggetto (embedding BERT)
        ogg_emb = func.relu(self.ogg_desc_layer(tender["ogg_desc"]))

        features = torch.cat([geo_emb, cat_emb, budget_emb, cpv_emb, ogg_emb], dim=1)
        output = self.output_layer(features)

        return output
